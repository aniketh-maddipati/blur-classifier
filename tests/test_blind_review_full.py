import csv
from pathlib import Path

import pytest

from blind_review_full import ensure_output_path, find_image, load_done_rows


def test_ensure_output_path_refuses_existing_file_in_fresh_mode(tmp_path: Path):
    out_path = tmp_path / "review.csv"
    out_path.write_text("already here")

    with pytest.raises(SystemExit, match="Refusing to append"):
        ensure_output_path(out_path, fresh=True)


def test_load_done_rows_accepts_keep_toss_schema(tmp_path: Path):
    out_path = tmp_path / "review.csv"
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file", "original_label", "blind_relabel", "note"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file": "DSC0001",
                "original_label": "sharp",
                "blind_relabel": "keep",
                "note": "",
            }
        )

    rows = load_done_rows(out_path, question="keep")
    assert rows["DSC0001"]["blind_relabel"] == "keep"


def test_find_image_searches_recursively(tmp_path: Path):
    image_path = tmp_path / "holdout" / "sharp" / "DSC0001.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"test")

    assert find_image(tmp_path, "DSC0001.jpg") == image_path
