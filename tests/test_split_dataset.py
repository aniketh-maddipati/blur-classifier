from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from exif_analyzer import ShotMetadata
from split_dataset import (
    SplitAssignment,
    assert_no_leakage,
    group_images,
    read_manifest,
    rebuild_dataset_from_rows,
)


def _make_row(tmp_path: Path, basename: str, target_class: str) -> dict[str, str]:
    source_path = tmp_path / basename
    source_path.write_bytes(b"jpeg")
    return {
        "filename": basename,
        "source_path": str(source_path),
        "target_class": target_class,
        "human_decision": "confirm",
        "indoor_or_outdoor": "",
        "focal_length": "50",
        "zero_shot_guess": "",
        "agrees_with_human": "",
    }


def _metadata_loader(mapping: dict[str, datetime | None]):
    def load(image_path: str) -> ShotMetadata:
        basename = Path(image_path).name
        return ShotMetadata(capture_datetime=mapping[basename])

    return load


def test_group_images_uses_chain_rule_for_bursts(tmp_path: Path):
    base = datetime(2026, 7, 21, 9, 0, 0)
    rows = [
        _make_row(tmp_path, "A.jpg", "sharp"),
        _make_row(tmp_path, "B.jpg", "sharp"),
        _make_row(tmp_path, "C.jpg", "sharp"),
    ]
    result = rebuild_dataset_from_rows(
        rows,
        train_dir=tmp_path / "train",
        holdout_dir=tmp_path / "holdout",
        manifest_path=tmp_path / "split_manifest.csv",
        gap_seconds=30,
        holdout_per_class=1,
        metadata_loader=_metadata_loader(
            {
                "A.jpg": base,
                "B.jpg": base + timedelta(seconds=10),
                "C.jpg": base + timedelta(seconds=20),
            }
        ),
    )

    assert len(result.assignments) == 1
    assert len(result.assignments[0].group.items) == 3


def test_group_images_splits_at_exact_gap_boundary():
    base = datetime(2026, 7, 21, 9, 0, 0)
    groups = group_images(
        [
            type("RI", (), {"basename": "A.jpg", "capture_datetime": base, "target_class": "sharp", "source_path": Path("A.jpg")})(),
            type("RI", (), {"basename": "B.jpg", "capture_datetime": base + timedelta(seconds=30), "target_class": "sharp", "source_path": Path("B.jpg")})(),
        ],
        gap_seconds=30,
    )

    assert len(groups) == 2


def test_mixed_class_group_stays_intact(tmp_path: Path):
    base = datetime(2026, 7, 21, 9, 0, 0)
    rows = [
        _make_row(tmp_path, "A.jpg", "sharp"),
        _make_row(tmp_path, "B.jpg", "intentional_blur"),
        _make_row(tmp_path, "C.jpg", "unintentional_blur"),
    ]
    result = rebuild_dataset_from_rows(
        rows,
        train_dir=tmp_path / "train",
        holdout_dir=tmp_path / "holdout",
        manifest_path=tmp_path / "split_manifest.csv",
        holdout_per_class=1,
        metadata_loader=_metadata_loader(
            {
                "A.jpg": base,
                "B.jpg": base + timedelta(seconds=5),
                "C.jpg": base + timedelta(seconds=10),
            }
        ),
    )

    manifest_rows = read_manifest(tmp_path / "split_manifest.csv")
    assert {row["split"] for row in manifest_rows} == {"holdout"}
    assert {row["group_id"] for row in manifest_rows} == {result.assignments[0].group.group_id}


def test_missing_exif_becomes_singleton_and_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    base = datetime(2026, 7, 21, 9, 0, 0)
    rows = [
        _make_row(tmp_path, "A.jpg", "sharp"),
        _make_row(tmp_path, "B.jpg", "sharp"),
    ]
    result = rebuild_dataset_from_rows(
        rows,
        train_dir=tmp_path / "train",
        holdout_dir=tmp_path / "holdout",
        manifest_path=tmp_path / "split_manifest.csv",
        holdout_per_class=1,
        metadata_loader=_metadata_loader({"A.jpg": base, "B.jpg": None}),
    )

    captured = capsys.readouterr().out
    assert "WARNING: Missing or unparseable EXIF capture timestamp" in captured
    assert "B.jpg" in captured
    assert any(len(assignment.group.items) == 1 and assignment.group.items[0].basename == "B.jpg" for assignment in result.assignments)


def test_split_is_deterministic_for_same_seed(tmp_path: Path):
    base = datetime(2026, 7, 21, 9, 0, 0)
    rows = [
        _make_row(tmp_path, "A.jpg", "sharp"),
        _make_row(tmp_path, "B.jpg", "sharp"),
        _make_row(tmp_path, "C.jpg", "intentional_blur"),
        _make_row(tmp_path, "D.jpg", "intentional_blur"),
    ]
    metadata = _metadata_loader(
        {
            "A.jpg": base,
            "B.jpg": base + timedelta(seconds=40),
            "C.jpg": base + timedelta(seconds=80),
            "D.jpg": base + timedelta(seconds=120),
        }
    )

    first = rebuild_dataset_from_rows(
        rows,
        train_dir=tmp_path / "train1",
        holdout_dir=tmp_path / "holdout1",
        manifest_path=tmp_path / "split_manifest_1.csv",
        seed=7,
        holdout_per_class=1,
        metadata_loader=metadata,
    )
    second = rebuild_dataset_from_rows(
        rows,
        train_dir=tmp_path / "train2",
        holdout_dir=tmp_path / "holdout2",
        manifest_path=tmp_path / "split_manifest_2.csv",
        seed=7,
        holdout_per_class=1,
        metadata_loader=metadata,
    )

    first_rows = read_manifest(first.manifest_path)
    second_rows = read_manifest(second.manifest_path)
    assert first_rows == second_rows


def test_no_leakage_assertion_rejects_cross_split_group():
    group = type(
        "Group",
        (),
        {
            "group_id": "group_0000",
            "items": (
                type("RI", (), {"basename": "A.jpg", "target_class": "sharp"})(),
            ),
        },
    )()
    assignments = [
        SplitAssignment(group=group, split="train"),
        SplitAssignment(group=group, split="holdout"),
    ]

    with pytest.raises(AssertionError, match="group"):
        assert_no_leakage(assignments)


def test_manifest_copy_has_no_basename_leakage(tmp_path: Path):
    base = datetime(2026, 7, 21, 9, 0, 0)
    rows = [
        _make_row(tmp_path, "A.jpg", "sharp"),
        _make_row(tmp_path, "B.jpg", "sharp"),
        _make_row(tmp_path, "C.jpg", "sharp"),
    ]
    rebuild_dataset_from_rows(
        rows,
        train_dir=tmp_path / "train",
        holdout_dir=tmp_path / "holdout",
        manifest_path=tmp_path / "split_manifest.csv",
        holdout_per_class=1,
        metadata_loader=_metadata_loader(
            {
                "A.jpg": base,
                "B.jpg": base + timedelta(seconds=40),
                "C.jpg": base + timedelta(seconds=80),
            }
        ),
    )

    manifest_rows = read_manifest(tmp_path / "split_manifest.csv")
    basenames_by_split: dict[str, set[str]] = {"train": set(), "holdout": set()}
    for row in manifest_rows:
        basenames_by_split[row["split"]].add(row["basename"])

    assert basenames_by_split["train"].isdisjoint(basenames_by_split["holdout"])
    assert sum(1 for _ in (tmp_path / "train" / "sharp").iterdir()) + sum(
        1 for _ in (tmp_path / "holdout" / "sharp").iterdir()
    ) == 3


def test_unreachable_holdout_target_prints_warning(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    base = datetime(2026, 7, 21, 9, 0, 0)
    rows = [_make_row(tmp_path, "A.jpg", "sharp"), _make_row(tmp_path, "B.jpg", "sharp")]
    rebuild_dataset_from_rows(
        rows,
        train_dir=tmp_path / "train",
        holdout_dir=tmp_path / "holdout",
        manifest_path=tmp_path / "split_manifest.csv",
        holdout_per_class=5,
        metadata_loader=_metadata_loader({"A.jpg": base, "B.jpg": base + timedelta(seconds=40)}),
    )

    captured = capsys.readouterr().out
    assert "did not reach the holdout target of 5" in captured
