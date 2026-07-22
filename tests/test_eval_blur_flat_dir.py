import csv
from pathlib import Path

from eval_blur import _iter_flat_dir_images, _write_flat_predictions_csv


def test_iter_flat_dir_images_ignores_nested_files(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"a")
    (tmp_path / "b.png").write_bytes(b"b")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.jpg").write_bytes(b"c")

    records = _iter_flat_dir_images(tmp_path)

    assert [record.basename for record in records] == ["a.jpg", "b.png"]


def test_write_flat_predictions_csv_writes_expected_columns(tmp_path: Path):
    csv_path = tmp_path / "predictions.csv"
    images = _iter_flat_dir_images(tmp_path)
    if not images:
        (tmp_path / "a.jpg").write_bytes(b"a")
        images = _iter_flat_dir_images(tmp_path)

    _write_flat_predictions_csv(
        csv_path,
        images,
        predictions_by_run=[["sharp"], ["unintentional blur"]],
    )

    rows = list(csv.DictReader(csv_path.open(newline="")))
    assert rows[0]["basename"] == "a.jpg"
    assert rows[0]["pred_run1"] == "sharp"
    assert rows[0]["pred_run2"] == "unintentional blur"
    assert rows[0]["flipped"] == "True"
    assert rows[0]["majority_pred"] == "sharp"
