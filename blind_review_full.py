#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from blur_labels import normalize

EVAL_CSV = Path("results/eval_runC_1344.csv")
FIELDNAMES = ("file", "original_label", "blind_relabel", "note")
KEYS_BY_QUESTION = {
    "intent": {"1": "intentional_blur", "2": "unintentional_blur", "3": "sharp"},
    "keep": {"1": "keep", "2": "toss"},
}
PROMPTS_BY_QUESTION = {
    "intent": "[{index}/{total}]  1=intentional_blur  2=unintentional_blur  3=sharp  q=quit",
    "keep": "[{index}/{total}]  keep or toss?  1=keep  2=toss  q=quit",
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".arw", ".cr2", ".nef", ".dng"}


@dataclass(frozen=True)
class ReviewItem:
    file_id: str
    basename: str
    original_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out", default="results/blind_full_review.csv")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Require a fresh output path instead of resuming/appending to an existing review CSV.",
    )
    parser.add_argument(
        "--question",
        choices=sorted(KEYS_BY_QUESTION),
        default="intent",
    )
    return parser.parse_args()


def load_eval_items(eval_csv: Path = EVAL_CSV) -> list[ReviewItem]:
    rows = list(csv.DictReader(eval_csv.open(newline="")))
    required = {"basename", "actual"}
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(
            f"HARD ERROR: {eval_csv} missing columns {sorted(missing)}. "
            f"Actual columns: {list(rows[0])}"
        )
    if len(rows) != 42:
        raise AssertionError(f"expected 42 holdout rows, got {len(rows)}")

    items = [
        ReviewItem(
            file_id=Path(row["basename"]).stem,
            basename=row["basename"].strip(),
            original_label=normalize(row["actual"]),
        )
        for row in rows
    ]
    random.Random(42).shuffle(items)
    return items


def find_image(image_dir: Path, basename: str) -> Path:
    for path in image_dir.rglob(basename):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            return path
    raise SystemExit(f"HARD ERROR: {basename} not found under {image_dir}")


def load_done_rows(out_path: Path, *, question: str) -> dict[str, dict[str, str]]:
    if not out_path.exists():
        return {}
    rows = list(csv.DictReader(out_path.open(newline="")))
    for row in rows:
        if set(row) != set(FIELDNAMES):
            raise AssertionError(
                f"{out_path} must have columns {list(FIELDNAMES)}, found {list(row)}"
            )
        blind_relabel = row["blind_relabel"].strip()
        valid_keys = set(KEYS_BY_QUESTION["intent"].values()) | set(KEYS_BY_QUESTION["keep"].values())
        if blind_relabel not in valid_keys:
            raise AssertionError(
                f"{out_path} has unsupported blind_relabel {blind_relabel!r}"
            )
        if question == "intent":
            normalize(row["original_label"])
    return {row["file"].strip(): row for row in rows}


def ensure_output_path(out_path: Path, *, fresh: bool) -> None:
    if fresh and out_path.exists():
        raise SystemExit(
            f"Refusing to append to existing review file {out_path}; remove it or choose a new --out path."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)


def open_anonymous_copy(source_path: Path) -> str:
    suffix = source_path.suffix.lower() or ".jpg"
    with tempfile.NamedTemporaryFile(
        prefix="blind_review_",
        suffix=suffix,
        delete=False,
    ) as handle:
        tmp_path = handle.name
    shutil.copy(source_path, tmp_path)
    subprocess.run(["open", "-a", "Preview", tmp_path], check=True)
    return tmp_path


def prompt_for_label(question: str, *, index: int, total: int) -> str:
    print()
    print(PROMPTS_BY_QUESTION[question].format(index=index, total=total))
    valid_choices = KEYS_BY_QUESTION[question]
    while True:
        response = input("call> ").strip().lower()
        if response == "q":
            raise KeyboardInterrupt
        if response in valid_choices:
            return valid_choices[response]
        expected = "/".join(valid_choices)
        print(f"{expected} or q only.")


def append_review_rows(
    items: list[ReviewItem],
    *,
    image_dir: Path,
    out_path: Path,
    question: str,
) -> None:
    done = load_done_rows(out_path, question=question)
    if done:
        print(f"Resuming: {len(done)}/42 already reviewed.")

    remaining = [item for item in items if item.file_id not in done]
    out_exists = out_path.exists()
    with out_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if not out_exists:
            writer.writeheader()

        for offset, item in enumerate(remaining, start=1):
            source_path = find_image(image_dir, item.basename)
            open_anonymous_copy(source_path)
            try:
                blind_relabel = prompt_for_label(
                    question,
                    index=len(done) + offset,
                    total=len(items),
                )
            except KeyboardInterrupt:
                print("Saved. Re-run the same command to resume.")
                return

            note = input("note (optional)> ").strip()
            writer.writerow(
                {
                    "file": item.file_id,
                    "original_label": item.original_label,
                    "blind_relabel": blind_relabel,
                    "note": note,
                }
            )
            handle.flush()
            print("  committed.")

    print(f"\nAll {len(items)} done. Next: verify join, then python3 recompute_corrected.py")


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    ensure_output_path(out_path, fresh=args.fresh)
    items = load_eval_items()
    append_review_rows(
        items,
        image_dir=Path(args.image_dir),
        out_path=out_path,
        question=args.question,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
