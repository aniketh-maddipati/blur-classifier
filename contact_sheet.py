#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from recompute_corrected import DEFAULT_REVIEW_PATH
from review_analysis import (
    DEFAULT_ADJUDICATIONS,
    build_adjudicated_source,
    build_review_source,
    load_eval_records,
    load_review_records,
    parse_adjudications,
)

BACKGROUND = (248, 246, 240)
TEXT = (24, 24, 24)
BORDER = (190, 184, 172)


@dataclass(frozen=True)
class SheetEntry:
    file_id: str
    image_path: Path
    corrected_label: str
    prediction: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--review-csv", default=str(DEFAULT_REVIEW_PATH))
    parser.add_argument("--adjudicate", action="append", default=[])
    parser.add_argument("--cases", default=None, help="Comma-separated image ids to render in a single large row.")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def build_entries(
    *,
    eval_csv: Path,
    review_csv: Path,
    adjudications: dict[str, str],
    cases: list[str] | None,
) -> list[SheetEntry]:
    review_records = load_review_records(review_csv, source_name="blind-corrected")
    blind_source = build_review_source(review_records)
    adjudicated_source = build_adjudicated_source(blind_source, adjudications)
    eval_records = load_eval_records(eval_csv, run_name=eval_csv.stem)
    by_file_id = {record.file_id: record for record in eval_records}

    if cases:
        entries: list[SheetEntry] = []
        for file_id in cases:
            if file_id not in by_file_id:
                raise AssertionError(f"{eval_csv} does not contain image id {file_id!r}")
            record = by_file_id[file_id]
            entries.append(
                SheetEntry(
                    file_id=file_id,
                    image_path=record.image_path,
                    corrected_label=adjudicated_source.labels_by_file[file_id],
                    prediction=record.majority_pred,
                )
            )
        return entries

    misses = [
        record
        for record in eval_records
        if adjudicated_source.labels_by_file[record.file_id] != record.majority_pred
    ]
    return [
        SheetEntry(
            file_id=record.file_id,
            image_path=record.image_path,
            corrected_label=adjudicated_source.labels_by_file[record.file_id],
            prediction=record.majority_pred,
        )
        for record in misses
    ]


def render_contact_sheet(entries: list[SheetEntry], *, single_row: bool) -> Image.Image:
    if not entries:
        raise AssertionError("No contact-sheet entries selected")

    font = ImageFont.load_default()
    thumb_size = (720, 540) if single_row else (320, 240)
    columns = len(entries) if single_row else min(4, len(entries))
    rows = math.ceil(len(entries) / columns)
    padding = 24
    caption_height = 48
    canvas_width = columns * thumb_size[0] + padding * (columns + 1)
    canvas_height = rows * (thumb_size[1] + caption_height) + padding * (rows + 1)
    canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    for index, entry in enumerate(entries):
        column = index % columns
        row = index // columns
        x0 = padding + column * (thumb_size[0] + padding)
        y0 = padding + row * (thumb_size[1] + caption_height + padding)
        with Image.open(entry.image_path) as source_image:
            image = source_image.convert("RGB")
        image.thumbnail(thumb_size)
        paste_x = x0 + (thumb_size[0] - image.width) // 2
        paste_y = y0 + (thumb_size[1] - image.height) // 2
        canvas.paste(image, (paste_x, paste_y))
        draw.rectangle(
            [x0, y0, x0 + thumb_size[0], y0 + thumb_size[1]],
            outline=BORDER,
            width=2,
        )
        caption = f"{entry.file_id}: {entry.corrected_label} -> {entry.prediction}"
        draw.text((x0, y0 + thumb_size[1] + 10), caption, fill=TEXT, font=font)
    return canvas


def default_output_path(eval_csv: Path, *, single_row: bool) -> Path:
    suffix = "cases" if single_row else "misses"
    return Path("results") / f"{eval_csv.stem}_{suffix}.png"


def main() -> None:
    args = parse_args()
    cases = [item.strip() for item in args.cases.split(",")] if args.cases else None
    adjudications = dict(DEFAULT_ADJUDICATIONS)
    adjudications.update(parse_adjudications(args.adjudicate))

    eval_csv = Path(args.eval_csv)
    out_path = Path(args.out) if args.out else default_output_path(eval_csv, single_row=bool(cases))
    if out_path.parent != Path("results"):
        raise AssertionError("Contact sheets must be written under results/")

    entries = build_entries(
        eval_csv=eval_csv,
        review_csv=Path(args.review_csv),
        adjudications=adjudications,
        cases=cases,
    )
    sheet = render_contact_sheet(entries, single_row=bool(cases))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="PNG")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
