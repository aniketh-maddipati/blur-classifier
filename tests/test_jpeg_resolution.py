from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image

from exif_analyzer import ShotMetadata
from jpeg_resolution import (
    AmbiguousDecisionError,
    JpegResolver,
    LabelConflictError,
    ResolutionError,
    enumerate_and_validate_decisions,
)
from split_dataset import read_manifest, rebuild_dataset_from_rows

FULL_RES = (6000, 4000)
PREVIEW_RES = (1616, 1080)
BASE_TS = datetime(2026, 7, 20, 9, 0, 0)


def _write_jpeg(path: Path, size: tuple[int, int] = FULL_RES) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (128, 128, 128)).save(path, format="JPEG")
    return path


def _write_arw_stub(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"NOT-A-REAL-ARW")
    return path


class FakeRunner:
    """Stands in for subprocess.run; 'converts' by writing a real JPEG and
    answers `sips -g` native-size queries. fail_tool forces a nonzero exit
    for that tool ('sips' or 'exiftool'); fail_geometry breaks only -g."""

    def __init__(
        self,
        output_size: tuple[int, int] = FULL_RES,
        native_size: tuple[int, int] | None = FULL_RES,
        fail_tool: str = "",
        fail_geometry: bool = False,
    ):
        self.output_size = output_size
        self.native_size = native_size
        self.fail_tool = fail_tool
        self.fail_geometry = fail_geometry
        self.calls: list[list[str]] = []

    def __call__(self, cmd, capture_output=True, text=True):
        self.calls.append(list(cmd))

        class _Result:
            returncode = 0
            stderr = ""
            stdout = ""

        is_geometry = cmd[0] == "sips" and "-g" in cmd
        if is_geometry:
            if self.fail_geometry or self.native_size is None:
                _Result.returncode = 1
            else:
                _Result.stdout = (
                    f"{cmd[-1]}\n  pixelWidth: {self.native_size[0]}\n"
                    f"  pixelHeight: {self.native_size[1]}\n"
                )
        elif cmd[0] == self.fail_tool:
            _Result.returncode = 1
            _Result.stderr = f"{cmd[0]} exploded"
        elif cmd[0] == "sips":
            _write_jpeg(Path(cmd[cmd.index("--out") + 1]), self.output_size)
        return _Result()

    def sips_calls(self) -> int:
        return sum(1 for cmd in self.calls if cmd[0] == "sips" and "-g" not in cmd)


def _loader_by_stem(mapping: dict[str, datetime | None]):
    """EXIF timestamps keyed by basename stem, so ARW and converted JPEG agree."""

    def load(image_path: str) -> ShotMetadata:
        stem = Path(image_path).name.split(".")[0]  # DSC02.jpg.tmp -> DSC02
        return ShotMetadata(capture_datetime=mapping[stem])

    return load


def _resolver(tmp_path: Path, *, loader=None, runner=None, **kwargs) -> JpegResolver:
    return JpegResolver(
        tmp_path / "archive",
        tmp_path / "extracted",
        metadata_loader=loader or _loader_by_stem({}),
        runner=runner or FakeRunner(),
        binary_checker=kwargs.pop("binary_checker", lambda binary: f"/usr/bin/{binary}"),
        **kwargs,
    )


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    session = tmp_path / "archive" / "raw-import-2026-07-20" / "100MSDCF"
    session.mkdir(parents=True)
    return session


# -- decision enumeration -----------------------------------------------------


def test_decision_counts_are_printed(capsys: pytest.CaptureFixture[str]):
    rows = [
        {"human_decision": "confirm"},
        {"human_decision": "confirm"},
        {"human_decision": "skip"},
        {"human_decision": "reject"},
        {"human_decision": "reclassify"},
    ]
    counts = enumerate_and_validate_decisions(rows)
    output = capsys.readouterr().out
    assert counts["confirm"] == 2
    assert "confirm: 2" in output
    assert "skip: 1" in output


def test_unknown_decision_value_hard_errors():
    with pytest.raises(AmbiguousDecisionError, match="maybe-later"):
        enumerate_and_validate_decisions([{"human_decision": "maybe-later"}])


# -- resolution order ---------------------------------------------------------


def test_default_converts_even_when_paired_jpeg_exists(tmp_path: Path, session_dir: Path):
    """Uniform provenance: camera JPEGs are ignored by default so jpeg_source
    and pixel_size cannot become class-correlated shortcut features."""
    arw = _write_arw_stub(session_dir / "DSC01.ARW")
    _write_jpeg(session_dir / "DSC01.JPG")
    resolver = _resolver(tmp_path, loader=_loader_by_stem({"DSC01": BASE_TS}))

    resolved = resolver.resolve(arw)

    assert resolved.jpeg_source == "sips_from_raw"
    assert resolved.jpeg_path.parent == tmp_path / "extracted"


def test_paired_jpeg_used_when_opted_in(tmp_path: Path, session_dir: Path):
    arw = _write_arw_stub(session_dir / "DSC01B.ARW")
    _write_jpeg(session_dir / "DSC01B.JPG")
    runner = FakeRunner()
    resolver = _resolver(tmp_path, runner=runner, prefer_paired=True)

    resolved = resolver.resolve(arw)

    assert resolved.jpeg_source == "paired_jpeg"
    assert resolved.jpeg_path == session_dir / "DSC01B.JPG"
    assert runner.sips_calls() == 0


def test_failed_conversion_leaves_no_cached_output(tmp_path: Path, session_dir: Path):
    """Atomicity regression (the DSC06934 incident): a conversion that dies
    mid-pipeline must not leave a file the cache would later trust."""
    arw = _write_arw_stub(session_dir / "DSC01C.ARW")
    loader = _loader_by_stem({"DSC01C": BASE_TS})
    broken = _resolver(tmp_path, loader=loader, runner=FakeRunner(fail_tool="exiftool"))

    with pytest.raises(ResolutionError, match="exiftool"):
        broken.resolve(arw)
    assert not (tmp_path / "extracted" / "DSC01C.jpg").exists()

    fixed = _resolver(tmp_path, loader=loader)
    resolved = fixed.resolve(arw)
    assert fixed.n_converted == 1 and fixed.n_cached == 0
    assert resolved.jpeg_path.exists()


def test_changed_conversion_params_invalidate_cache(tmp_path: Path, session_dir: Path):
    arw = _write_arw_stub(session_dir / "DSC01D.ARW")
    loader = _loader_by_stem({"DSC01D": BASE_TS})
    resolver = _resolver(tmp_path, loader=loader)
    output = resolver.resolve(arw).jpeg_path
    os.utime(output, (output.stat().st_atime, arw.stat().st_mtime + 60))

    # Same params -> cache hit
    again = _resolver(tmp_path, loader=loader)
    again.resolve(arw)
    assert again.n_cached == 1

    # Recorded params differ -> full miss, sidecar rewritten
    (tmp_path / "extracted" / ".sips_params").write_text("format=jpeg,quality=80\n")
    stale = _resolver(tmp_path, loader=loader)
    stale.resolve(arw)
    assert stale.n_converted == 1 and stale.n_cached == 0
    assert "quality=95" in (tmp_path / "extracted" / ".sips_params").read_text()


def test_conversion_records_source_session(tmp_path: Path, session_dir: Path):
    arw = _write_arw_stub(session_dir / "DSC02.ARW")
    resolver = _resolver(tmp_path, loader=_loader_by_stem({"DSC02": BASE_TS}))

    resolved = resolver.resolve(arw)

    assert resolved.jpeg_source == "sips_from_raw"
    assert resolved.source_session == "raw-import-2026-07-20"
    assert resolved.jpeg_path.suffix == ".jpg"


def test_source_path_used_before_fallback_index(tmp_path: Path, session_dir: Path):
    arw = _write_arw_stub(session_dir / "DSC03.ARW")
    # A same-stem decoy elsewhere in the archive must never be considered
    # while source_path itself exists.
    _write_arw_stub(tmp_path / "archive" / "raw-import-2026-07-20-card2" / "DSC03.ARW")
    resolver = _resolver(tmp_path, loader=_loader_by_stem({"DSC03": BASE_TS}))

    resolver.resolve(arw)

    assert resolver._index is None  # fallback index was never built


def test_cache_reuse_skips_second_conversion(tmp_path: Path, session_dir: Path):
    arw = _write_arw_stub(session_dir / "DSC04.ARW")
    runner = FakeRunner()
    resolver = _resolver(tmp_path, loader=_loader_by_stem({"DSC04": BASE_TS}), runner=runner)

    output = resolver.resolve(arw).jpeg_path
    os.utime(output, (output.stat().st_atime, arw.stat().st_mtime + 60))
    resolver.resolve(arw)

    assert runner.sips_calls() == 1
    assert resolver.n_converted == 1
    assert resolver.n_cached == 1


# -- conversion validation ----------------------------------------------------


def test_undersized_conversion_rejected(tmp_path: Path, session_dir: Path):
    arw = _write_arw_stub(session_dir / "DSC05.ARW")
    resolver = _resolver(tmp_path, runner=FakeRunner(output_size=PREVIEW_RES))

    with pytest.raises(ResolutionError, match="natively 6000x4000"):
        resolver.resolve(arw)


def test_crop_mode_conversion_passes_when_sizes_match(tmp_path: Path, session_dir: Path):
    """Regression for the a7III APS-C crop mode: 3936x2624 output is valid
    when the source RAW is natively 3936x2624 (caught live at Checkpoint 1)."""
    crop = (3936, 2624)
    arw = _write_arw_stub(session_dir / "DSC05C.ARW")
    resolver = _resolver(
        tmp_path,
        loader=_loader_by_stem({"DSC05C": BASE_TS}),
        runner=FakeRunner(output_size=crop, native_size=crop),
    )

    resolved = resolver.resolve(arw)

    assert resolved.jpeg_source == "sips_from_raw"
    assert resolved.pixel_size == "3936x2624"


def test_unreadable_native_size_falls_back_to_floor(
    tmp_path: Path, session_dir: Path, capsys: pytest.CaptureFixture[str]
):
    arw_ok = _write_arw_stub(session_dir / "DSC05D.ARW")
    resolver = _resolver(
        tmp_path,
        loader=_loader_by_stem({"DSC05D": BASE_TS, "DSC05E": BASE_TS}),
        runner=FakeRunner(fail_geometry=True),
    )
    resolver.resolve(arw_ok)  # 6000x4000 output passes the floor
    assert "falling back" in capsys.readouterr().err

    arw_small = _write_arw_stub(session_dir / "DSC05E.ARW")
    small = _resolver(
        tmp_path,
        loader=_loader_by_stem({"DSC05E": BASE_TS}),
        runner=FakeRunner(output_size=PREVIEW_RES, fail_geometry=True),
    )
    with pytest.raises(ResolutionError, match="floor|preview"):
        small.resolve(arw_small)


def test_stale_undersized_cache_hit_is_rejected(tmp_path: Path, session_dir: Path):
    arw = _write_arw_stub(session_dir / "DSC06.ARW")
    stale = _write_jpeg(tmp_path / "extracted" / "DSC06.jpg", PREVIEW_RES)
    os.utime(stale, (stale.stat().st_atime, arw.stat().st_mtime + 60))

    with pytest.raises(ResolutionError, match="natively"):
        _resolver(tmp_path).resolve(arw)


def test_corrupt_cache_hit_fails_pil_verification(tmp_path: Path, session_dir: Path):
    arw = _write_arw_stub(session_dir / "DSC07.ARW")
    corrupt = tmp_path / "extracted" / "DSC07.jpg"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"\xff\xd8garbage-not-a-jpeg")
    os.utime(corrupt, (corrupt.stat().st_atime, arw.stat().st_mtime + 60))

    with pytest.raises(ResolutionError, match="PIL verification"):
        _resolver(tmp_path).resolve(arw)


def test_sips_failure_surfaces_stderr(tmp_path: Path, session_dir: Path):
    arw = _write_arw_stub(session_dir / "DSC08.ARW")
    resolver = _resolver(tmp_path, runner=FakeRunner(fail_tool="sips"))

    with pytest.raises(ResolutionError, match="sips exploded"):
        resolver.resolve(arw)


def test_exiftool_failure_surfaces_stderr(tmp_path: Path, session_dir: Path):
    arw = _write_arw_stub(session_dir / "DSC09.ARW")
    resolver = _resolver(
        tmp_path,
        loader=_loader_by_stem({"DSC09": BASE_TS}),
        runner=FakeRunner(fail_tool="exiftool"),
    )

    with pytest.raises(ResolutionError, match="exiftool exploded"):
        resolver.resolve(arw)


def test_timestamp_mismatch_after_conversion_hard_errors(tmp_path: Path, session_dir: Path):
    arw = _write_arw_stub(session_dir / "DSC10.ARW")

    def loader(image_path: str) -> ShotMetadata:
        offset = timedelta() if image_path.lower().endswith(".arw") else timedelta(seconds=1)
        return ShotMetadata(capture_datetime=BASE_TS + offset)

    with pytest.raises(ResolutionError, match="Timestamp mismatch"):
        _resolver(tmp_path, loader=loader).resolve(arw)


def test_exif_copy_command_shape(tmp_path: Path, session_dir: Path):
    arw = _write_arw_stub(session_dir / "DSC11.ARW")
    runner = FakeRunner()
    _resolver(tmp_path, loader=_loader_by_stem({"DSC11": BASE_TS}), runner=runner).resolve(arw)

    exiftool_calls = [cmd for cmd in runner.calls if cmd[0] == "exiftool"]
    assert len(exiftool_calls) == 1
    assert "-TagsFromFile" in exiftool_calls[0]
    assert str(arw) in exiftool_calls[0]
    assert "-DateTimeOriginal" in exiftool_calls[0]


# -- preflight and unsupported inputs -----------------------------------------


def test_preflight_reports_unmounted_archive(tmp_path: Path):
    resolver = JpegResolver(
        tmp_path / "not-mounted",
        tmp_path / "extracted",
        metadata_loader=_loader_by_stem({}),
    )
    with pytest.raises(ResolutionError, match="check that T7 is mounted"):
        resolver.preflight()


def test_preflight_reports_missing_binary(tmp_path: Path):
    (tmp_path / "archive").mkdir()
    resolver = _resolver(tmp_path, binary_checker=lambda binary: None)
    with pytest.raises(ResolutionError, match="sips"):
        resolver.preflight()


def test_unsupported_extension_hard_errors(tmp_path: Path, session_dir: Path):
    png = session_dir / "DSC12.png"
    png.write_bytes(b"png-bytes")
    with pytest.raises(ResolutionError, match="unsupported extension"):
        _resolver(tmp_path).resolve(png)


# -- fallback basename search -------------------------------------------------


def test_missing_source_falls_back_when_unique(
    tmp_path: Path, session_dir: Path, capsys: pytest.CaptureFixture[str]
):
    _write_arw_stub(session_dir / "DSC13.ARW")
    resolver = _resolver(tmp_path, loader=_loader_by_stem({"DSC13": BASE_TS}))

    resolved = resolver.resolve(tmp_path / "gone" / "DSC13.ARW")

    assert resolved.jpeg_source == "sips_from_raw"
    captured = capsys.readouterr()
    assert "resolved via archive index" in captured.err
    assert "basename index" in captured.err


def test_missing_source_with_no_archive_hit_is_unresolved(tmp_path: Path, session_dir: Path):
    with pytest.raises(ResolutionError, match="UNRESOLVED GONE.ARW"):
        _resolver(tmp_path).resolve(tmp_path / "gone" / "GONE.ARW")


def test_ambiguous_fallback_with_differing_timestamps_hard_errors(
    tmp_path: Path, session_dir: Path
):
    _write_arw_stub(session_dir / "DSC14.ARW")
    _write_arw_stub(tmp_path / "archive" / "raw-import-2026-07-20-card2" / "DSC14.ARW")

    def loader(image_path: str) -> ShotMetadata:
        offset = timedelta(hours=1) if "card2" in image_path else timedelta()
        return ShotMetadata(capture_datetime=BASE_TS + offset)

    with pytest.raises(ResolutionError, match="Ambiguous basename"):
        _resolver(tmp_path, loader=loader).resolve(tmp_path / "gone" / "DSC14.ARW")


def test_ambiguous_fallback_with_equal_timestamps_is_disambiguated(
    tmp_path: Path, session_dir: Path
):
    _write_arw_stub(session_dir / "DSC15.ARW")
    _write_arw_stub(tmp_path / "archive" / "raw-import-2026-07-20-card2" / "DSC15.ARW")
    resolver = _resolver(tmp_path, loader=_loader_by_stem({"DSC15": BASE_TS}))

    assert resolver.resolve(tmp_path / "gone" / "DSC15.ARW").jpeg_source == "sips_from_raw"


def test_ambiguous_fallback_with_unreadable_exif_hard_errors(
    tmp_path: Path, session_dir: Path
):
    _write_arw_stub(session_dir / "DSC16.ARW")
    _write_arw_stub(tmp_path / "archive" / "raw-import-2026-07-20-card2" / "DSC16.ARW")

    def loader(image_path: str) -> ShotMetadata:
        raise OSError("EXIF read failed")

    with pytest.raises(ResolutionError, match="EXIF read failed"):
        _resolver(tmp_path, loader=loader).resolve(tmp_path / "gone" / "DSC16.ARW")


# -- integration through rebuild_dataset_from_rows ----------------------------


def _review_row(source_path: Path, target_class: str, *, review_offset_hours: int) -> dict[str, str]:
    """Review-time columns are populated with wildly different times than the
    EXIF timestamps, so any grouping code that touched them would misgroup."""
    review_time = (BASE_TS + timedelta(hours=review_offset_hours)).isoformat()
    return {
        "filename": source_path.name,
        "source_path": str(source_path),
        "target_class": target_class,
        "human_decision": "confirm",
        "indoor_or_outdoor": "",
        "focal_length": "50",
        "zero_shot_guess": "",
        "agrees_with_human": "",
        "timestamp": review_time,
        "first_display_timestamp": review_time,
        "session_id": f"review-session-{review_offset_hours}",
    }


def _rebuild(tmp_path: Path, rows, loader, resolver):
    return rebuild_dataset_from_rows(
        rows,
        train_dir=tmp_path / "dataset" / "train",
        holdout_dir=tmp_path / "dataset" / "holdout",
        manifest_path=tmp_path / "dataset" / "split_manifest.csv",
        holdout_per_class=1,
        metadata_loader=loader,
        resolver=resolver,
    )


def test_rebuild_produces_all_jpeg_dataset_with_annotated_manifest(
    tmp_path: Path, session_dir: Path
):
    stems = ["DSC20", "DSC21", "DSC22", "DSC23"]
    arws = [_write_arw_stub(session_dir / f"{stem}.ARW") for stem in stems]
    _write_jpeg(session_dir / "DSC20.JPG")  # paired JPEG present but ignored by default
    exif = {stem: BASE_TS + timedelta(minutes=5 * index) for index, stem in enumerate(stems)}
    loader = _loader_by_stem(exif)
    resolver = _resolver(tmp_path, loader=loader)
    classes = ["sharp", "sharp", "intentional_blur", "unintentional_blur"]
    rows = [
        _review_row(arw, cls, review_offset_hours=index * 100)
        for index, (arw, cls) in enumerate(zip(arws, classes))
    ]

    result = _rebuild(tmp_path, rows, loader, resolver)

    dataset_files = [
        path
        for split in ("train", "holdout")
        for path in (tmp_path / "dataset" / split).rglob("*")
        if path.is_file()
    ]
    assert dataset_files, "expected copied dataset files"
    assert all(path.suffix == ".jpg" for path in dataset_files)

    manifest_rows = read_manifest(result.manifest_path)
    assert {row["basename"] for row in manifest_rows} == {f"{stem}.jpg" for stem in stems}
    assert all(row["jpeg_source"] == "sips_from_raw" for row in manifest_rows)
    assert all(row["source_session"] == "raw-import-2026-07-20" for row in manifest_rows)
    assert all(row["pixel_size"] == "6000x4000" for row in manifest_rows)

    basenames_by_split: dict[str, set[str]] = {"train": set(), "holdout": set()}
    for row in manifest_rows:
        basenames_by_split[row["split"]].add(row["basename"])
    assert basenames_by_split["train"].isdisjoint(basenames_by_split["holdout"])


def test_grouping_uses_exif_not_review_time_columns(tmp_path: Path, session_dir: Path):
    """Regression: three shots form one EXIF burst (<30s apart) although their
    review-time columns are 100+ hours apart. Review-time grouping would make
    three singleton groups; EXIF grouping must make exactly one group."""
    stems = ["DSC30", "DSC31", "DSC32"]
    arws = [_write_arw_stub(session_dir / f"{stem}.ARW") for stem in stems]
    exif = {stem: BASE_TS + timedelta(seconds=10 * index) for index, stem in enumerate(stems)}
    loader = _loader_by_stem(exif)
    rows = [
        _review_row(arw, "sharp", review_offset_hours=(index + 1) * 100)
        for index, arw in enumerate(arws)
    ]

    result = _rebuild(tmp_path, rows, loader, _resolver(tmp_path, loader=loader))

    assert len(result.assignments) == 1
    assert len(result.assignments[0].group.items) == 3


def test_missing_exif_survives_conversion_as_singleton(
    tmp_path: Path, session_dir: Path, capsys: pytest.CaptureFixture[str]
):
    """Edge: RAW with no capture timestamp still converts (None == None passes
    the equality check) and lands in a warned singleton group."""
    arw = _write_arw_stub(session_dir / "DSC33.ARW")
    good = _write_arw_stub(session_dir / "DSC34.ARW")
    loader = _loader_by_stem({"DSC33": None, "DSC34": BASE_TS})
    rows = [
        _review_row(arw, "sharp", review_offset_hours=0),
        _review_row(good, "sharp", review_offset_hours=1),
    ]

    result = _rebuild(tmp_path, rows, loader, _resolver(tmp_path, loader=loader))

    assert "Missing or unparseable EXIF capture timestamp" in capsys.readouterr().out
    singletons = [a for a in result.assignments if a.group.start_timestamp is None]
    assert len(singletons) == 1
    assert singletons[0].group.items[0].basename == "DSC33.jpg"


def test_rebuild_aggregates_all_unresolved_basenames(tmp_path: Path, session_dir: Path):
    arw = _write_arw_stub(session_dir / "DSC40.ARW")
    loader = _loader_by_stem({"DSC40": BASE_TS})
    rows = [
        _review_row(arw, "sharp", review_offset_hours=0),
        _review_row(tmp_path / "gone" / "GONE1.ARW", "sharp", review_offset_hours=1),
        _review_row(tmp_path / "gone" / "GONE2.ARW", "sharp", review_offset_hours=2),
    ]

    with pytest.raises(ResolutionError) as excinfo:
        _rebuild(tmp_path, rows, loader, _resolver(tmp_path, loader=loader))

    message = str(excinfo.value)
    assert "2 unresolved" in message
    assert "GONE1" in message and "GONE2" in message


def test_rebuild_rejects_label_conflicts(tmp_path: Path, session_dir: Path):
    arw = _write_arw_stub(session_dir / "DSC50.ARW")
    loader = _loader_by_stem({"DSC50": BASE_TS})
    rows = [
        _review_row(arw, "sharp", review_offset_hours=0),
        _review_row(arw, "unintentional_blur", review_offset_hours=1),
    ]

    with pytest.raises(LabelConflictError, match="DSC50"):
        _rebuild(tmp_path, rows, loader, _resolver(tmp_path, loader=loader))


def test_duplicate_identical_review_rows_are_deduplicated(
    tmp_path: Path, session_dir: Path, capsys: pytest.CaptureFixture[str]
):
    arw = _write_arw_stub(session_dir / "DSC51.ARW")
    loader = _loader_by_stem({"DSC51": BASE_TS})
    rows = [_review_row(arw, "sharp", review_offset_hours=hour) for hour in range(3)]

    result = _rebuild(tmp_path, rows, loader, _resolver(tmp_path, loader=loader))

    assert len(read_manifest(result.manifest_path)) == 1
    assert "dropped 2 duplicate review rows" in capsys.readouterr().out


def test_rebuild_with_nothing_selected_hard_errors(tmp_path: Path, session_dir: Path):
    arw = _write_arw_stub(session_dir / "DSC52.ARW")
    row = _review_row(arw, "sharp", review_offset_hours=0)
    row["human_decision"] = "skip"

    with pytest.raises(ResolutionError, match="No confirmed rows"):
        _rebuild(tmp_path, [row], _loader_by_stem({}), _resolver(tmp_path))


def test_legacy_human_confirmed_rows_are_selected(tmp_path: Path, session_dir: Path):
    arw = _write_arw_stub(session_dir / "DSC53.ARW")
    loader = _loader_by_stem({"DSC53": BASE_TS})
    row = _review_row(arw, "sharp", review_offset_hours=0)
    row["human_decision"] = ""
    row["human_confirmed"] = "y"

    result = _rebuild(tmp_path, [row], loader, _resolver(tmp_path, loader=loader))

    assert len(read_manifest(result.manifest_path)) == 1


def test_raw_copy_guard_blocks_arw_even_without_resolver(tmp_path: Path, session_dir: Path):
    arw = _write_arw_stub(session_dir / "DSC60.ARW")
    rows = [_review_row(arw, "sharp", review_offset_hours=0)]

    with pytest.raises(ResolutionError, match="Refusing to copy RAW"):
        _rebuild(tmp_path, rows, _loader_by_stem({"DSC60": BASE_TS}), resolver=None)


def test_canary_converts_first_raw_even_with_pair_by_default(
    tmp_path: Path, session_dir: Path, capsys: pytest.CaptureFixture[str]
):
    arw = _write_arw_stub(session_dir / "DSC69.ARW")
    _write_jpeg(session_dir / "DSC69.JPG")
    resolver = _resolver(tmp_path, loader=_loader_by_stem({"DSC69": BASE_TS}))

    resolver.run_canary([arw])

    assert "Canary conversion: DSC69.ARW" in capsys.readouterr().out


def test_canary_runs_before_batch_and_reports(
    tmp_path: Path, session_dir: Path, capsys: pytest.CaptureFixture[str]
):
    arw = _write_arw_stub(session_dir / "DSC70.ARW")
    loader = _loader_by_stem({"DSC70": BASE_TS})
    rows = [_review_row(arw, "sharp", review_offset_hours=0)]

    _rebuild(tmp_path, rows, loader, _resolver(tmp_path, loader=loader))

    output = capsys.readouterr().out
    assert "Canary conversion: DSC70.ARW" in output
    assert "Canary OK" in output
    assert "Resolution: 0 paired / 1 converted" in output