"""
triage_dataset.py
------------------
Sorts ~1000 RAW files into rough candidate buckets for the blur classifier,
reusing the EXIF logic already in exif_analyzer.py — so labeling becomes
confirming a smaller pre-sorted candidate pool, not eyeballing 1000 photos.

Important: this is a PRE-FILTER, not a labeler. EXIF alone can't tell
intentional from unintentional blur — that's exactly the judgment the
classifier exists to make. This script only narrows down which photos are
even worth a human look for each class.

Usage:
    python triage_dataset.py /path/to/sd_card/DCIM /path/to/output_folder

Requires exif_analyzer.py in the same folder (or on the Python path).
"""

import sys
from pathlib import Path

from exif_analyzer import extract_metadata, CROP_FACTOR

# Rough shutter-speed bands, relative to each photo's own reciprocal-rule
# safe speed (1 / effective focal length). Starting points — tune after
# looking at your first batch of results.
SHARP_CANDIDATE_MARGIN = 2.0      # at least 2x faster than the safe speed
INTENTIONAL_BLUR_THRESHOLD = 1.0  # seconds — tripod-range slow


def bucket_for(shutter, focal_length) -> str:
    """Buckets a shot by shutter speed relative to its own safe threshold."""
    if shutter is None or focal_length is None or focal_length <= 0:
        return "unknown"

    safe_shutter = 1.0 / (focal_length * CROP_FACTOR)

    if shutter <= safe_shutter / SHARP_CANDIDATE_MARGIN:
        return "sharp_candidates"
    if shutter >= INTENTIONAL_BLUR_THRESHOLD:
        return "intentional_blur_candidates"
    if shutter > safe_shutter:
        return "unintentional_blur_candidates"
    return "borderline"  # doesn't cleanly fall in any bucket -- low priority


def main():
    if len(sys.argv) != 3:
        print("Usage: python triage_dataset.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = Path(sys.argv[1])
    output_folder = Path(sys.argv[2])

    raw_files = sorted(input_folder.glob("*.ARW")) + sorted(input_folder.glob("*.arw"))
    print(f"Found {len(raw_files)} RAW files.")

    counts: dict[str, int] = {}
    unreadable = 0

    for path in raw_files:
        try:
            metadata = extract_metadata(str(path))
        except Exception as exc:
            unreadable += 1
            print(f"  Skipping {path.name}: could not read EXIF ({exc})")
            continue

        bucket = bucket_for(metadata.shutter_speed, metadata.focal_length)
        counts[bucket] = counts.get(bucket, 0) + 1

        bucket_dir = output_folder / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)

        # Symlink rather than copy: instant, and doesn't duplicate ~50MB
        # RAW files onto disk a second time.
        link_path = bucket_dir / path.name
        if not link_path.exists():
            link_path.symlink_to(path.resolve())

    print("\nCandidate counts:")
    for bucket, count in sorted(counts.items()):
        print(f"  {bucket}: {count}")
    if unreadable:
        print(f"  (unreadable/skipped: {unreadable})")

    print(
        "\nNext: open each *_candidates folder and manually confirm/cull "
        "down to ~40 per class (30 train + 10 holdout). 'borderline' and "
        "'unknown' are worth a skim but not a priority — most of your "
        "labels should come from the three confident buckets."
    )


if __name__ == "__main__":
    main()
    