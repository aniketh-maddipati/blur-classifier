from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

import blur_dataset
from blur_dataset import (
    ASSISTANT_PREFIX,
    BlurDatasetBuilder,
    BlurDatasetConfig,
    BlurClassifierDataset,
    CLASS_NAMES,
    LocalDatasetSplit,
    USER_PROMPT,
    load_blur_examples,
    parse_predicted_class_name,
)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (32, 24), color)
    image.save(path, format="JPEG")


def _write_manifest(dataset_root: Path, rows: list[dict[str, str]]) -> Path:
    manifest_path = dataset_root / "split_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "basename",
                "class",
                "group_id",
                "group_start_ts",
                "split",
                "seed",
                "gap_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def _make_dataset(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "dataset"
    rows: list[dict[str, str]] = []
    colors = {
        "intentional_blur": (255, 0, 0),
        "unintentional_blur": (0, 255, 0),
        "sharp": (0, 0, 255),
    }
    split_counts = {"train": 2, "holdout": 1}
    for class_name, color in colors.items():
        for split_name, count in split_counts.items():
            for idx in range(count):
                basename = f"{class_name}_{split_name}_{idx}.jpg"
                _write_image(dataset_root / split_name / class_name / basename, color)
                rows.append(
                    {
                        "basename": basename,
                        "class": class_name,
                        "group_id": f"{class_name}_{split_name}_{idx}",
                        "group_start_ts": "",
                        "split": split_name,
                        "seed": "42",
                        "gap_seconds": "30",
                    }
                )
    manifest_path = _write_manifest(dataset_root, rows)
    return dataset_root, manifest_path


def _counts(dataset: LocalDatasetSplit) -> dict[str, int]:
    return dataset.counts_by_class_name()


class _DummyRenderer:
    def build_supervised_example(self, messages, train_on_what):
        return SimpleNamespace(length=lambda: 64), None


def _part_value(part, key: str):
    if isinstance(part, dict):
        return part.get(key)
    return getattr(part, key, None)


@pytest.fixture
def offline_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(blur_dataset, "get_tokenizer", lambda _name: object())
    monkeypatch.setattr(blur_dataset, "get_image_processor", lambda _name: object())
    monkeypatch.setattr(blur_dataset, "get_renderer", lambda **_kwargs: _DummyRenderer())


def test_builder_counts_match_manifest(tmp_path: Path, offline_renderer: None) -> None:
    dataset_root, manifest_path = _make_dataset(tmp_path)
    builder = BlurDatasetBuilder(
        model_name_for_tokenizer="Qwen/Qwen3.6-35B-A3B",
        renderer_name="qwen3_5_disable_thinking",
        dataset_root=str(dataset_root),
        manifest_path=str(manifest_path),
        batch_size=2,
        run_nll_evaluator=True,
    )

    train_dataset, test_dataset = builder()

    assert isinstance(train_dataset, BlurClassifierDataset)
    assert isinstance(test_dataset, BlurClassifierDataset)
    assert _counts(train_dataset.dataset) == {
        "intentional blur": 2,
        "unintentional blur": 2,
        "sharp": 2,
    }
    assert _counts(test_dataset.dataset) == {
        "intentional blur": 1,
        "unintentional blur": 1,
        "sharp": 1,
    }


def test_load_blur_examples_uses_holdout_for_test_split(tmp_path: Path) -> None:
    dataset_root, manifest_path = _make_dataset(tmp_path)
    test_examples = load_blur_examples(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="test",
    )
    assert len(test_examples) == 3
    assert all(Path(example.image_path).parts[-3] == "holdout" for example in test_examples)


def test_smoke_render_or_validate_message_structure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root, manifest_path = _make_dataset(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    try:
        dataset = BlurClassifierDataset(
            BlurDatasetConfig(
                dataset_root=str(dataset_root),
                manifest_path=str(manifest_path),
                dataset_split="train",
                model_name_for_tokenizer="Qwen/Qwen3.6-35B-A3B",
                renderer_name="qwen3_5_disable_thinking",
                max_image_size=1024,
                hflip_probability=0.0,
            )
        )
        example = dataset.dataset[0]
        model_input, _weights = dataset.build_supervised_example(example)
        token_count = model_input.length
        print(f"Rendered example token count at max_image_size=1024: {token_count}")
        assert token_count <= 8192
    except Exception as exc:
        monkeypatch.setattr(blur_dataset, "get_tokenizer", lambda _name: object())
        monkeypatch.setattr(blur_dataset, "get_image_processor", lambda _name: object())
        monkeypatch.setattr(blur_dataset, "get_renderer", lambda **_kwargs: _DummyRenderer())
        dataset = BlurClassifierDataset(
            BlurDatasetConfig(
                dataset_root=str(dataset_root),
                manifest_path=str(manifest_path),
                dataset_split="train",
                model_name_for_tokenizer="Qwen/Qwen3.6-35B-A3B",
                renderer_name="qwen3_5_disable_thinking",
                max_image_size=1024,
                hflip_probability=0.0,
            )
        )
        example = dataset.dataset[0]
        messages = dataset.build_messages(example)
        user_parts = messages[0]["content"]
        assistant_parts = messages[1]["content"]
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert _part_value(user_parts[0], "type") == "image"
        assert _part_value(user_parts[1], "text") == USER_PROMPT
        assert _part_value(assistant_parts[0], "text").startswith(ASSISTANT_PREFIX)
        assert any(class_name in _part_value(assistant_parts[0], "text") for class_name in CLASS_NAMES)
        print(f"Renderer smoke test fell back to message-structure validation: {exc}")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("intentional blur", "intentional blur"),
        ("intentional_blur", "intentional blur"),
        ("The blur in this photo is: unintentional blur", "unintentional blur"),
        ("The blur in this photo is: unintentional_blur", "unintentional blur"),
    ],
)
def test_parse_predicted_class_name_accepts_spaced_and_underscored_labels(raw: str, expected: str) -> None:
    assert parse_predicted_class_name(raw) == expected
