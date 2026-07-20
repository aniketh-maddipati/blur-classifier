"""
Fast keyboard-driven review loop for candidate images.

This does not assign labels automatically. It samples likely candidates,
records human decisions first, and only then runs an optional blind
zero-shot comparison against the untouched base model for later analysis.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from config import CLASSES, MODEL_NAME

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".raw",
    ".cr2",
    ".cr3",
    ".nef",
    ".arw",
    ".dng",
    ".raf",
    ".rw2",
    ".orf",
}
CSV_FIELDS = [
    "filename",
    "source_path",
    "claimed_class",
    "human_confirmed",
    "indoor_or_outdoor",
    "focal_length",
    "zero_shot_guess",
    "agrees_with_human",
]
BUCKET_ORDER = ("35mm", "85mm", "28-75mm_zoom")


@dataclass(frozen=True)
class Candidate:
    path: Path
    metadata: Any
    bucket: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample candidates, then run a fast crash-safe review loop."
    )
    parser.add_argument("candidates_folder", help="Folder containing candidate images")
    parser.add_argument("target_count", type=int, help="How many images to review")
    parser.add_argument(
        "--csv",
        dest="csv_path",
        help="Where to append review results (default: results/review_<folder>.csv)",
    )
    parser.add_argument(
        "--claimed-class",
        choices=CLASSES,
        help="Override the claimed class if it cannot be inferred from the folder name",
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_NAME,
        help="Base model used only for the blind comparison log",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic sampling seed",
    )
    return parser.parse_args()


def infer_claimed_class(folder: Path, explicit_value: str | None) -> str:
    if explicit_value:
        return explicit_value

    name = folder.name
    if name.endswith("_candidates"):
        maybe_class = name[: -len("_candidates")]
        if maybe_class in CLASSES:
            return maybe_class

    if name in CLASSES:
        return name

    raise ValueError(
        "Could not infer claimed_class from the folder name. "
        "Pass --claimed-class explicitly."
    )


def default_csv_path(folder: Path) -> Path:
    return Path("results") / f"review_{folder.name}.csv"


def iter_image_files(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def classify_focal_bucket(focal_length: float | None) -> str | None:
    if focal_length is None:
        return None
    if 33.0 <= focal_length <= 37.0:
        return "35mm"
    if 80.0 <= focal_length <= 90.0:
        return "85mm"
    if 28.0 <= focal_length <= 75.0:
        return "28-75mm_zoom"
    return None


def load_candidates(folder: Path) -> tuple[list[Candidate], int]:
    from exif_analyzer import extract_metadata

    candidates: list[Candidate] = []
    skipped = 0

    for path in iter_image_files(folder):
        try:
            metadata = extract_metadata(str(path))
        except Exception as exc:
            skipped += 1
            print(f"[warn] Skipping {path.name}: could not read EXIF ({exc})")
            continue

        bucket = classify_focal_bucket(metadata.focal_length)
        if bucket is None:
            skipped += 1
            continue

        candidates.append(Candidate(path=path, metadata=metadata, bucket=bucket))

    return candidates, skipped


def allocate_samples(bucket_counts: dict[str, int], target_count: int) -> tuple[dict[str, int], list[str]]:
    allocations = {bucket: 0 for bucket in BUCKET_ORDER}
    notes: list[str] = []
    remaining = target_count

    active_buckets = list(BUCKET_ORDER)
    while remaining > 0 and active_buckets:
        even_share = max(1, remaining // len(active_buckets))
        next_round: list[str] = []

        for bucket in active_buckets:
            available = bucket_counts[bucket] - allocations[bucket]
            if available <= 0:
                notes.append(
                    f"{bucket} had no image(s) left when its even-share target was {even_share}; "
                    "reallocated the shortfall to other buckets."
                )
                continue

            take = min(even_share, available)
            allocations[bucket] += take
            remaining -= take

            if take < even_share:
                notes.append(
                    f"{bucket} only had {available} image(s) left when its even-share target was {even_share}; "
                    "reallocated the shortfall to other buckets."
                )

            if bucket_counts[bucket] - allocations[bucket] > 0:
                next_round.append(bucket)

            if remaining == 0:
                break

        if active_buckets == next_round and remaining > 0:
            for bucket in active_buckets:
                available = bucket_counts[bucket] - allocations[bucket]
                if available <= 0:
                    continue
                allocations[bucket] += 1
                remaining -= 1
                if remaining == 0:
                    break
            next_round = [
                bucket for bucket in active_buckets if bucket_counts[bucket] - allocations[bucket] > 0
            ]

        active_buckets = next_round

    total_available = sum(bucket_counts.values())
    if target_count > total_available:
        notes.append(
            f"Requested {target_count} image(s), but only {total_available} matched the focal-length buckets."
        )

    return allocations, notes


def sample_candidates(candidates: list[Candidate], target_count: int, seed: int) -> tuple[list[Candidate], dict[str, int], list[str]]:
    by_bucket: dict[str, list[Candidate]] = {bucket: [] for bucket in BUCKET_ORDER}
    for candidate in candidates:
        by_bucket[candidate.bucket].append(candidate)

    rng = random.Random(seed)
    for items in by_bucket.values():
        rng.shuffle(items)

    bucket_counts = {bucket: len(items) for bucket, items in by_bucket.items()}
    allocation_target = min(target_count, sum(bucket_counts.values()))
    allocations, notes = allocate_samples(bucket_counts, allocation_target)

    sampled: list[Candidate] = []
    max_len = max(allocations.values(), default=0)
    for index in range(max_len):
        for bucket in BUCKET_ORDER:
            if index < allocations[bucket]:
                sampled.append(by_bucket[bucket][index])

    return sampled, bucket_counts, notes


def read_reviewed_filenames(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["filename"] for row in reader if row.get("filename")}


def ensure_csv_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()


def append_review_row(csv_path: Path, row: dict[str, str]) -> None:
    ensure_csv_header(csv_path)
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def open_in_viewer(path: Path) -> None:
    subprocess.run(["open", str(path)], check=False)


def read_single_key(prompt: str, valid_keys: set[str]) -> str:
    print(prompt, end="", flush=True)
    if os.name != "posix":
        while True:
            value = input().strip().lower()
            if value in valid_keys:
                return value
            print(f"Please enter one of: {', '.join(sorted(valid_keys))}")

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            key = sys.stdin.read(1).lower()
            if key in valid_keys:
                print(key)
                return key
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def blind_zero_shot_guess(image_path: Path, model_path: str) -> str:
    try:
        from classify import classify_image, parse_classification, resize_image

        image_bytes = resize_image(str(image_path))
        raw = classify_image(image_bytes, model_path)
        return parse_classification(raw) or ""
    except Exception as exc:
        print(f"[warn] Zero-shot guess failed for {image_path.name}: {exc}")
        return ""


def agrees_with_human(claimed_class: str, human_confirmed: str, zero_shot_guess: str) -> str:
    if not zero_shot_guess:
        return ""
    if human_confirmed == "y":
        return "true" if zero_shot_guess == claimed_class else "false"
    return "true" if zero_shot_guess != claimed_class else "false"


def review_candidates(
    sampled: list[Candidate],
    claimed_class: str,
    csv_path: Path,
    model_path: str,
) -> None:
    reviewed = read_reviewed_filenames(csv_path)
    pending = [candidate for candidate in sampled if candidate.path.name not in reviewed]

    print(f"\nSampled {len(sampled)} image(s); {len(reviewed)} already reviewed, {len(pending)} pending.")
    if not pending:
        print("Nothing left to review for this CSV.")
        return

    for index, candidate in enumerate(pending, start=1):
        open_in_viewer(candidate.path)
        focal_display = (
            f"{candidate.metadata.focal_length:.1f}mm"
            if candidate.metadata.focal_length is not None
            else "unknown"
        )
        print(
            f"\n[{index}/{len(pending)}] {candidate.path.name}  "
            f"bucket={candidate.bucket}  focal={focal_display}"
        )

        label_key = read_single_key(
            "[y] confirm label is correct   [n] reject, wrong label   [s] skip   [q] quit: ",
            {"y", "n", "s", "q"},
        )
        if label_key == "q":
            print("Quitting. All previously reviewed rows are already saved.")
            return
        if label_key == "s":
            print("Skipped for later.")
            continue

        indoor_key = read_single_key(
            "[i] mark indoor   [o] mark outdoor   [s] skip this image   [q] quit: ",
            {"i", "o", "s", "q"},
        )
        if indoor_key == "q":
            print("Quitting. All previously reviewed rows are already saved.")
            return
        if indoor_key == "s":
            print("Skipped for later.")
            continue

        zero_shot_guess = blind_zero_shot_guess(candidate.path, model_path)
        row = {
            "filename": candidate.path.name,
            "source_path": str(candidate.path.resolve()),
            "claimed_class": claimed_class,
            "human_confirmed": label_key,
            "indoor_or_outdoor": indoor_key,
            "focal_length": (
                f"{candidate.metadata.focal_length:.4f}"
                if candidate.metadata.focal_length is not None
                else ""
            ),
            "zero_shot_guess": zero_shot_guess,
            "agrees_with_human": agrees_with_human(claimed_class, label_key, zero_shot_guess),
        }
        append_review_row(csv_path, row)
        print("Saved review row.")


def main() -> None:
    args = parse_args()
    folder = Path(args.candidates_folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Candidates folder not found: {folder}")

    claimed_class = infer_claimed_class(folder, args.claimed_class)
    csv_path = Path(args.csv_path) if args.csv_path else default_csv_path(folder)

    candidates, skipped_count = load_candidates(folder)
    sampled, bucket_counts, notes = sample_candidates(candidates, args.target_count, args.seed)

    print(f"Claimed class: {claimed_class}")
    print(f"Candidates folder: {folder}")
    print(f"Review CSV: {csv_path}")
    print("\nBucket inventory:")
    for bucket in BUCKET_ORDER:
        print(f"  {bucket:<13} {bucket_counts[bucket]}")
    if skipped_count:
        print(f"  skipped/unbucketed {skipped_count}")

    print("\nSampling plan:")
    sampled_counts = defaultdict(int)
    for candidate in sampled:
        sampled_counts[candidate.bucket] += 1
    for bucket in BUCKET_ORDER:
        print(f"  {bucket:<13} {sampled_counts[bucket]}")
    for note in notes:
        print(f"  [note] {note}")

    review_candidates(sampled, claimed_class, csv_path, args.model_path)


if __name__ == "__main__":
    main()
