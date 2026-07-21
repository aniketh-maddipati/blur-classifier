from __future__ import annotations

import io
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from subprocess import CompletedProcess

import pytest
from PIL import Image

from exif_analyzer import ShotMetadata
from split_dataset import (
    ConversionFailure,
    assert_no_leakage,
    collect_reviewed_images,
    convert_raw_to_jpeg,
    group_images,
    read_manifest,
    rebuild_dataset_from_rows,
    resolve_jpeg_sources,
)


def _write_jpeg(path: Path, size: tuple[int, int] = (6000, 4000)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (100, 120, 140)).save(path, format="JPEG")


def _write_raw(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"raw")


def _make_row(
    source_path: Path,
    target_class: str,
    *,
    decision: str = "confirm",
    review_timestamp: str = "2026-07-21T11:00:00+00:00",
    session_id: str = "session-1",
) -> dict[str, str]:
    return {
        "filename": source_path.name,
        "source_path": str(source_path),
        "target_class": target_class,
        "human_decision": decision,
        "indoor_or_outdoor": "",
        "focal_length": "50",
        "zero_shot_guess": "",
        "agrees_with_human": "",
        "timestamp": review_timestamp,
        "first_display_timestamp": review_timestamp,
        "session_id": session_id,
    }


def _metadata_loader(mapping: dict[str, datetime | None]):
    def load(image_path: str) -> ShotMetadata:
        return ShotMetadata(capture_datetime=mapping.get(Path(image_path).name))

    return load


def _runner_factory(
    sizes_by_source: dict[str, tuple[int, int]] | None = None,
    calls: list[list[str]] | None = None,
):
    sizes_by_source = sizes_by_source or {}
    calls = calls if calls is not None else []

    def run(cmd: list[str], check: bool, capture_output: bool, text: bool):
        calls.append(cmd)
        if cmd[0] == "sips":
            source = Path(cmd[-3])
            output = Path(cmd[-1])
            size = sizes_by_source.get(source.name, (6000, 4000))
            _write_jpeg(output, size=size)
            return CompletedProcess(cmd, 0, "", "")
        if cmd[0] == "exiftool":
            return CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"Unexpected command: {cmd}")

    return run


def test_group_images_uses_capture_metadata_not_review_metadata(tmp_path: Path):
    base = datetime(2026, 7, 21, 9, 0, 0)
    a = tmp_path / "A.jpg"
    b = tmp_path / "B.jpg"
    _write_jpeg(a, size=(600, 600))
    _write_jpeg(b, size=(600, 600))
    rows = [
        _make_row(a, "sharp", review_timestamp="2026-07-21T12:00:00+00:00", session_id="review-a"),
        _make_row(b, "sharp", review_timestamp="2026-07-21T18:00:00+00:00", session_id="review-b"),
    ]
    images, _missing = collect_reviewed_images(
        rows,
        archive_root=tmp_path,
        metadata_loader=_metadata_loader({"A.jpg": base, "B.jpg": base + timedelta(seconds=10)}),
    )

    groups = group_images(images, gap_seconds=30)
    assert len(groups) == 1


def test_paired_jpeg_preferred_over_conversion(tmp_path: Path):
    archive_root = tmp_path / "archive"
    raw_path = archive_root / "raw-import-2026-07-20" / "100MSDCF" / "DSC0001.ARW"
    jpg_path = raw_path.with_suffix(".JPG")
    _write_raw(raw_path)
    _write_jpeg(jpg_path)
    calls: list[list[str]] = []

    rebuild_dataset_from_rows(
        [_make_row(raw_path, "sharp")],
        train_dir=tmp_path / "train",
        holdout_dir=tmp_path / "holdout",
        manifest_path=tmp_path / "split_manifest.csv",
        holdout_per_class=1,
        metadata_loader=_metadata_loader({"DSC0001.ARW": datetime(2026, 7, 21, 9, 0, 0)}),
        archive_root=archive_root,
        extracted_dir=tmp_path / "_extracted",
        runner=_runner_factory(calls=calls),
    )

    assert not any(cmd[0] == "sips" for cmd in calls)
    manifest_rows = read_manifest(tmp_path / "split_manifest.csv")
    assert manifest_rows[0]["basename"] == "DSC0001.JPG"
    assert manifest_rows[0]["jpeg_source"] == "paired_jpeg"
    copied = list((tmp_path / "holdout" / "sharp").iterdir())
    assert [path.name for path in copied] == ["DSC0001.JPG"]


def test_conversion_cache_reuse_and_stale_reconvert(tmp_path: Path):
    raw_path = tmp_path / "DSC0002.ARW"
    extracted_dir = tmp_path / "_extracted"
    output_path = extracted_dir / "DSC0002.jpg"
    _write_raw(raw_path)
    _write_jpeg(output_path)
    raw_dt = datetime(2026, 7, 21, 9, 0, 0)
    jpg_dt = raw_dt

    calls: list[list[str]] = []
    runner = _runner_factory(calls=calls)
    output_path.touch()
    output_path.unlink()
    _write_jpeg(output_path)
    raw_path.touch()
    output_path.touch()
    output_path.touch()
    # Make the cache newer than the RAW.
    import os
    os.utime(raw_path, (1, 1))
    os.utime(output_path, (2, 2))

    cached = convert_raw_to_jpeg(
        raw_path,
        extracted_dir=extracted_dir,
        metadata_loader=_metadata_loader({raw_path.name: raw_dt, output_path.name: jpg_dt}),
        runner=runner,
    )
    assert cached == output_path
    assert not any(cmd[0] == "sips" for cmd in calls)

    calls.clear()
    os.utime(raw_path, (3, 3))
    reconverted = convert_raw_to_jpeg(
        raw_path,
        extracted_dir=extracted_dir,
        metadata_loader=_metadata_loader({raw_path.name: raw_dt, output_path.name: jpg_dt}),
        runner=runner,
    )
    assert reconverted == output_path
    assert any(cmd[0] == "sips" for cmd in calls)


def test_undersized_conversion_rejected(tmp_path: Path):
    raw_path = tmp_path / "DSC0003.ARW"
    _write_raw(raw_path)
    with pytest.raises(RuntimeError, match="undersized"):
        resolve_jpeg_sources(
            [
                type(
                    "Image",
                    (),
                    {
                        "basename": "DSC0003.jpg",
                        "jpeg_source": "sips_from_raw",
                        "source_path": raw_path,
                    },
                )()
            ],
            extracted_dir=tmp_path / "_extracted",
            archive_root=tmp_path,
            metadata_loader=_metadata_loader(
                {"DSC0003.ARW": datetime(2026, 7, 21, 9, 0, 0), "DSC0003.jpg": datetime(2026, 7, 21, 9, 0, 0)}
            ),
            runner=_runner_factory(sizes_by_source={"DSC0003.ARW": (1616, 1080)}),
        )


def test_source_path_used_before_basename_fallback_and_fallback_warns(tmp_path: Path):
    archive_root = tmp_path / "archive"
    existing = archive_root / "raw-import-2026-07-20" / "100MSDCF" / "DSC0004.JPG"
    fallback = archive_root / "raw-import-2026-07-21" / "100MSDCF" / "DSC9999.JPG"
    _write_jpeg(existing)
    _write_jpeg(fallback)
    rows = [
        _make_row(existing, "sharp"),
        _make_row(tmp_path / "missing" / "DSC9999.JPG", "sharp"),
    ]

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        images, _missing = collect_reviewed_images(
            rows,
            archive_root=archive_root,
            metadata_loader=_metadata_loader(
                {"DSC0004.JPG": datetime(2026, 7, 21, 9, 0, 0), "DSC9999.JPG": datetime(2026, 7, 21, 9, 0, 1)}
            ),
        )
    assert images[0].source_path == existing
    assert images[1].source_path == fallback
    assert "fell back to archive search for DSC9999.JPG" in stdout.getvalue()


def test_ambiguous_basename_hard_errors(tmp_path: Path):
    archive_root = tmp_path / "archive"
    a = archive_root / "raw-import-2026-07-20" / "100MSDCF" / "DSC0005.ARW"
    b = archive_root / "raw-import-2026-07-21" / "100MSDCF" / "DSC0005.ARW"
    _write_raw(a)
    _write_raw(b)
    row = _make_row(tmp_path / "missing" / "DSC0005.ARW", "sharp")

    with pytest.raises(RuntimeError, match="Ambiguous basename"):
        collect_reviewed_images(
            [row],
            archive_root=archive_root,
            metadata_loader=_metadata_loader({"DSC0005.ARW": datetime(2026, 7, 21, 9, 0, 0)}),
        )


def test_conflicting_labels_hard_error(tmp_path: Path):
    path = tmp_path / "DSC0006.JPG"
    _write_jpeg(path)
    rows = [_make_row(path, "sharp"), _make_row(path, "intentional_blur", decision="reclassify")]

    with pytest.raises(RuntimeError, match="Conflicting labels"):
        collect_reviewed_images(
            rows,
            archive_root=tmp_path,
            metadata_loader=_metadata_loader({"DSC0006.JPG": datetime(2026, 7, 21, 9, 0, 0)}),
        )


def test_all_jpeg_and_no_leakage_assertions(tmp_path: Path):
    archive_root = tmp_path / "archive"
    a = archive_root / "raw-import-2026-07-20" / "100MSDCF" / "A.ARW"
    b = archive_root / "raw-import-2026-07-20" / "100MSDCF" / "B.ARW"
    _write_raw(a)
    _write_raw(b)

    rebuild_dataset_from_rows(
        [_make_row(a, "sharp"), _make_row(b, "sharp", review_timestamp="2026-07-21T12:00:00+00:00")],
        train_dir=tmp_path / "train",
        holdout_dir=tmp_path / "holdout",
        manifest_path=tmp_path / "split_manifest.csv",
        holdout_per_class=1,
        metadata_loader=_metadata_loader(
            {
                "A.ARW": datetime(2026, 7, 21, 9, 0, 0),
                "B.ARW": datetime(2026, 7, 21, 10, 0, 0),
                "A.jpg": datetime(2026, 7, 21, 9, 0, 0),
                "B.jpg": datetime(2026, 7, 21, 10, 0, 0),
            }
        ),
        archive_root=archive_root,
        extracted_dir=tmp_path / "_extracted",
        runner=_runner_factory(),
    )

    all_files = list((tmp_path / "train").rglob("*")) + list((tmp_path / "holdout").rglob("*"))
    assert all(not path.is_file() or path.suffix.lower() == ".jpg" for path in all_files)
    manifest_rows = read_manifest(tmp_path / "split_manifest.csv")
    basenames_by_split = {"train": set(), "holdout": set()}
    for row in manifest_rows:
        basenames_by_split[row["split"]].add(row["basename"])
    assert basenames_by_split["train"].isdisjoint(basenames_by_split["holdout"])


def test_exif_timestamp_equality_between_raw_and_converted_jpeg(tmp_path: Path):
    raw_path = tmp_path / "DSC0007.ARW"
    _write_raw(raw_path)
    calls: list[list[str]] = []
    timestamp = datetime(2026, 7, 21, 9, 0, 0, 123456)

    output = convert_raw_to_jpeg(
        raw_path,
        extracted_dir=tmp_path / "_extracted",
        metadata_loader=_metadata_loader({raw_path.name: timestamp, "DSC0007.jpg": timestamp}),
        runner=_runner_factory(calls=calls),
    )

    assert output.name == "DSC0007.jpg"
    assert any(cmd[0] == "exiftool" for cmd in calls)


def test_no_leakage_assertion_rejects_cross_split_group():
    group = type(
        "Group",
        (),
        {
            "group_id": "group_0000",
            "items": (type("RI", (), {"basename": "A.jpg", "target_class": "sharp"})(),),
        },
    )()
    assignments = [type("A", (), {"group": group, "split": "train"})(), type("A", (), {"group": group, "split": "holdout"})()]

    with pytest.raises(AssertionError, match="group"):
        assert_no_leakage(assignments)
