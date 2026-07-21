"""
Resolve reviewed CSV rows to all-JPEG image sources.

RAW rows (Sony a7III .ARW) are converted with `sips` into EXTRACTED_DIR by
default, even when a camera JPEG sits next to the RAW — uniform provenance
prevents the model from learning in-camera-processing artifacts that
correlate with class (pass prefer_paired=True to restore the old behavior).
Conversions are cached and validated on every hit (PIL-verified, pixel size
equal to the source RAW's native size — a mismatch means an embedded-preview
fallback or partial decode). Conversion is atomic: the output is renamed
into place only after validation, EXIF copy, and the timestamp check all
pass, so a cache hit implies a fully processed file. A params sidecar in
EXTRACTED_DIR invalidates the whole cache if conversion settings change.
Anything unresolvable is a hard error listing every failed basename.

Grouping-relevant EXIF tags are copied from the source RAW with exiftool and
the capture timestamp is verified equal on both files. No exifread import
here: callers inject `metadata_loader` (normally exif_analyzer.extract_metadata).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from PIL import Image

RAW_EXTENSIONS = {".arw", ".cr2", ".nef", ".dng"}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}
PAIRED_SUFFIXES = (".JPG", ".jpg", ".JPEG", ".jpeg")
MIN_LONGEST_SIDE = 5000  # fallback floor, used only when native size is unreadable
SIPS_QUALITY = "95"
CONVERSION_PARAMS = f"format=jpeg,quality={SIPS_QUALITY}"
PARAMS_SIDECAR = ".sips_params"
SESSION_PREFIX = "raw-import-"
KNOWN_DECISIONS = {"confirm", "reclassify", "reject", "skip", ""}
EXIF_COPY_TAGS = [
    "-DateTimeOriginal",
    "-SubSecTimeOriginal",
    "-ExposureTime",
    "-FNumber",
    "-ISO",
    "-FocalLength",
]


class ResolutionError(RuntimeError):
    """Fatal dataset-resolution failure; never resolved silently."""


class AmbiguousDecisionError(ResolutionError):
    """human_decision contains a value with unknown keep/drop semantics."""


class LabelConflictError(ResolutionError):
    """The same basename appears with conflicting target_class labels."""


@dataclass(frozen=True)
class ResolvedJpeg:
    jpeg_path: Path
    jpeg_source: str  # "paired_jpeg" | "sips_from_raw"
    source_session: str  # raw-import-* folder in the source path, else ""
    pixel_size: str  # "WxH" of the JPEG, e.g. "6000x4000" (or "3936x2624" in crop mode)


def enumerate_and_validate_decisions(rows: Iterable[dict[str, str]]) -> Counter:
    """Print human_decision value counts; hard-error on unknown values."""
    counts = Counter((row.get("human_decision") or "").strip() for row in rows)
    print("Rows per human_decision value:")
    for value, count in counts.most_common():
        print(f"  {value or '<empty>'}: {count}")
    unknown = set(counts) - KNOWN_DECISIONS
    if unknown:
        raise AmbiguousDecisionError(
            "human_decision contains values with unknown keep/drop semantics: "
            f"{sorted(unknown)}. Extend KNOWN_DECISIONS (and row_is_selected) "
            "deliberately instead of guessing."
        )
    return counts


def _jpeg_size(path: Path) -> str:
    try:
        with Image.open(path) as image:
            return f"{image.size[0]}x{image.size[1]}"
    except Exception:
        return ""


def source_session_of(path: Path) -> str:
    return next((part for part in path.parts if part.startswith(SESSION_PREFIX)), "")


class JpegResolver:
    """Resolves one source path at a time; owns the conversion cache and a
    lazily built case-insensitive basename index used as a warned fallback."""

    def __init__(
        self,
        archive_root: Path,
        extracted_dir: Path,
        *,
        metadata_loader: Callable[[str], object],
        runner: Callable = subprocess.run,
        binary_checker: Callable[[str], Optional[str]] = shutil.which,
        prefer_paired: bool = False,
        min_longest_side: int = MIN_LONGEST_SIDE,
        progress_every: int = 10,
    ) -> None:
        self.archive_root = Path(archive_root)
        self.extracted_dir = Path(extracted_dir)
        self.metadata_loader = metadata_loader
        self.runner = runner
        self.binary_checker = binary_checker
        self.prefer_paired = prefer_paired
        self.min_longest_side = min_longest_side
        self.progress_every = progress_every
        self.n_paired = 0
        self.n_converted = 0
        self.n_cached = 0
        self._index: Optional[dict[str, list[Path]]] = None
        self._native_sizes: dict[Path, Optional[tuple[int, int]]] = {}
        self._params_stale: Optional[bool] = None

    def preflight(self) -> None:
        if not self.archive_root.exists():
            raise ResolutionError(
                f"Archive root {self.archive_root} not found — check that T7 is mounted."
            )
        for binary in ("sips", "exiftool"):
            if self.binary_checker(binary) is None:
                raise ResolutionError(f"Required external binary {binary!r} not on PATH.")

    def run_canary(self, source_paths: Iterable[Path]) -> None:
        """Convert the first RAW-only source end-to-end before batching."""
        for path in source_paths:
            if path.suffix.lower() not in RAW_EXTENSIONS:
                continue
            if self.prefer_paired and any(
                path.with_suffix(s).exists() for s in PAIRED_SUFFIXES
            ):
                continue
            print(f"Canary conversion: {path.name}")
            self.resolve(path)
            print("Canary OK (PIL-verified, native size, EXIF timestamp equality).")
            return
        print("Canary skipped: no RAW-only sources found.")

    def resolve(self, source_path: Path) -> ResolvedJpeg:
        source_path = self._locate(Path(source_path))
        session = source_session_of(source_path)

        if source_path.suffix.lower() in JPEG_EXTENSIONS:
            self.n_paired += 1
            return ResolvedJpeg(source_path, "paired_jpeg", session, _jpeg_size(source_path))
        if self.prefer_paired:
            for suffix in PAIRED_SUFFIXES:
                paired = source_path.with_suffix(suffix)
                if paired.exists():
                    self.n_paired += 1
                    return ResolvedJpeg(paired, "paired_jpeg", session, _jpeg_size(paired))

        if source_path.suffix.lower() not in RAW_EXTENSIONS:
            raise ResolutionError(f"{source_path}: unsupported extension.")

        output = self.extracted_dir / f"{source_path.stem}.jpg"
        if self._cache_valid(source_path, output):
            self.n_cached += 1
        else:
            self._convert(source_path, output)
            self.n_converted += 1
            if self.n_converted % self.progress_every == 0:
                print(f"  converted {self.n_converted} RAW files so far...")
        # Validation runs on cache hits too, so a stale preview-sized output
        # from an older pipeline can never sneak through.
        width, height = self._validate_jpeg(output, source_path)
        return ResolvedJpeg(output, "sips_from_raw", session, f"{width}x{height}")

    def summary(self) -> str:
        return (
            f"Resolution: {self.n_paired} paired / "
            f"{self.n_converted} converted / {self.n_cached} cache hits"
        )

    def _locate(self, source_path: Path) -> Path:
        if source_path.exists():
            return source_path
        hits = self._lookup_raw(source_path.stem)
        if not hits:
            raise ResolutionError(
                f"UNRESOLVED {source_path.name}: source_path {source_path} is "
                "missing and no archive hit."
            )
        chosen = hits[0] if len(hits) == 1 else self._disambiguate(source_path, hits)
        print(
            f"WARNING: {source_path.name}: source_path missing; resolved via "
            f"archive index -> {chosen}",
            file=sys.stderr,
        )
        return chosen

    def _lookup_raw(self, stem: str) -> list[Path]:
        if self._index is None:
            print(
                "WARNING: building recursive basename index (fallback path in use)",
                file=sys.stderr,
            )
            self._index = {}
            for path in self.archive_root.rglob("*"):
                if path.is_file() and path.suffix.lower() in RAW_EXTENSIONS:
                    self._index.setdefault(path.stem.upper(), []).append(path)
        return sorted(self._index.get(stem.upper(), []))

    def _disambiguate(self, source_path: Path, hits: list[Path]) -> Path:
        stamps: dict[Path, object] = {}
        for hit in hits:
            try:
                stamps[hit] = self.metadata_loader(str(hit)).capture_datetime
            except Exception as exc:
                raise ResolutionError(
                    f"Ambiguous basename {source_path.name}: {len(hits)} archive "
                    f"hits and EXIF read failed for {hit}: {exc}"
                ) from exc
        if len(set(stamps.values())) == 1:
            return hits[0]  # identical timestamps => duplicate copies of one shot
        raise ResolutionError(
            f"Ambiguous basename {source_path.name}: multiple archive files "
            f"with differing EXIF timestamps: "
            f"{ {str(path): str(ts) for path, ts in stamps.items()} }"
        )

    def _run_tool(self, cmd: list[str], action: str) -> None:
        result = self.runner(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ResolutionError(
                f"{action} failed (rc={result.returncode}): {result.stderr.strip()}"
            )

    def _cache_valid(self, source_path: Path, output: Path) -> bool:
        """True only if the cached output is complete and current. Misses when:
        the source is newer, conversion params changed since the cache was
        written, or the output never finished (atomic rename guarantees a
        present file is fully processed)."""
        if self._params_stale is None:
            sidecar = self.extracted_dir / PARAMS_SIDECAR
            recorded = sidecar.read_text().strip() if sidecar.exists() else None
            self._params_stale = recorded is not None and recorded != CONVERSION_PARAMS
            if self._params_stale:
                print(
                    f"WARNING: conversion params changed ({recorded} -> "
                    f"{CONVERSION_PARAMS}); ignoring all cached conversions.",
                    file=sys.stderr,
                )
            self.extracted_dir.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(CONVERSION_PARAMS + "\n")
        if self._params_stale:
            return False
        return output.exists() and output.stat().st_mtime > source_path.stat().st_mtime

    def _convert(self, source_path: Path, output: Path) -> None:
        """Atomic: work on a .tmp file; rename into place only after
        validation, the EXIF tag copy, and the timestamp check all pass."""
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_name(output.name + ".tmp")
        self._run_tool(
            ["sips", "-s", "format", "jpeg", "-s", "formatOptions", SIPS_QUALITY,
             str(source_path), "--out", str(tmp)],
            f"sips conversion of {source_path}",
        )
        self._validate_jpeg(tmp, source_path)
        self._run_tool(
            ["exiftool", "-TagsFromFile", str(source_path), *EXIF_COPY_TAGS,
             "-overwrite_original", str(tmp)],
            f"exiftool tag copy {source_path} -> {tmp}",
        )
        source_ts = self.metadata_loader(str(source_path)).capture_datetime
        output_ts = self.metadata_loader(str(tmp)).capture_datetime
        if source_ts != output_ts:
            raise ResolutionError(
                f"Timestamp mismatch after conversion: {source_path}={source_ts} "
                f"vs {tmp}={output_ts}"
            )
        tmp.replace(output)

    def _validate_jpeg(self, path: Path, source_path: Path) -> tuple[int, int]:
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()  # integrity scan without a full decode
        except Exception as exc:
            raise ResolutionError(f"{path} failed PIL verification: {exc}") from exc

        native = self._native_size(source_path)
        if native is not None:
            # Orientation-agnostic: a rotated decode is still a full decode.
            if {width, height} != set(native):
                raise ResolutionError(
                    f"{path} is {width}x{height} but source {source_path.name} "
                    f"is natively {native[0]}x{native[1]} — embedded-preview "
                    "fallback or partial decode; treating as unresolved."
                )
        elif max(width, height) < self.min_longest_side:
            raise ResolutionError(
                f"{path} is {width}x{height}, native size of {source_path.name} "
                f"is unreadable, and longest side < {self.min_longest_side}px. "
                "Likely an embedded-preview fallback; treating as unresolved."
            )
        return width, height

    def _native_size(self, source_path: Path) -> Optional[tuple[int, int]]:
        """Source RAW's native pixel size via `sips -g`, cached; None if unreadable."""
        if source_path not in self._native_sizes:
            result = self.runner(
                ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(source_path)],
                capture_output=True,
                text=True,
            )
            match = re.search(
                r"pixelWidth:\s*(\d+).*?pixelHeight:\s*(\d+)",
                getattr(result, "stdout", "") or "",
                re.DOTALL,
            )
            if result.returncode != 0 or match is None:
                print(
                    f"WARNING: could not read native size of {source_path.name}; "
                    f"falling back to the {self.min_longest_side}px floor.",
                    file=sys.stderr,
                )
                self._native_sizes[source_path] = None
            else:
                self._native_sizes[source_path] = (int(match.group(1)), int(match.group(2)))
        return self._native_sizes[source_path]