from __future__ import annotations

import asyncio
import io
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, TypedDict, cast

import chz
import numpy as np
import tinker
import torch
from PIL import Image
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.image_processing_utils import get_image_processor, resize_image
from tinker_cookbook.recipes.vlm_classifier.data import ClassifierDataset
from tinker_cookbook.renderers import ContentPart, ImagePart, Message, TextPart, TrainOnWhat, get_renderer, get_text_content
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.misc_utils import timed

from blur_labels import CLASSES, find_in_text, normalize

logger = logging.getLogger(__name__)

CLASS_SLUGS = CLASSES
CLASS_NAMES = tuple(class_name.replace("_", " ") for class_name in CLASS_SLUGS)
CLASS_SLUG_TO_NAME = dict(zip(CLASS_SLUGS, CLASS_NAMES, strict=True))
CLASS_NAME_TO_SLUG = {name: slug for slug, name in CLASS_SLUG_TO_NAME.items()}
SPLIT_TO_DIRNAME = {"train": "train", "test": "holdout"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".arw", ".cr2", ".nef", ".dng"}
RAW_EXTENSIONS = {".arw", ".cr2", ".nef", ".dng"}
USER_PROMPT = "What is the blur category in this photo?"
ASSISTANT_PREFIX = "The blur in this photo is:"

try:
    import rawpy
except ImportError:  # pragma: no cover - exercised only when rawpy is absent
    rawpy = None


@dataclass(frozen=True)
class BlurExample:
    image_path: str
    label: int
    basename: str
    class_slug: str
    split: str


class LocalClassLabel:
    def __init__(self, names: Iterable[str]):
        self._names = tuple(names)
        self._name_to_index = {name: idx for idx, name in enumerate(self._names)}

    def int2str(self, index: int) -> str:
        return self._names[index]

    def str2int(self, name: str) -> int:
        return self._name_to_index[name]


class LocalDatasetSplit:
    def __init__(self, examples: list[BlurExample]):
        self._examples = list(examples)
        self.features = {"label": LocalClassLabel(CLASS_NAMES)}

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int | str) -> BlurExample | list[Any]:
        if isinstance(index, str):
            return [getattr(example, index) for example in self._examples]
        return self._examples[index]

    def select(self, indices: list[int]) -> "LocalDatasetSplit":
        return LocalDatasetSplit([self._examples[index] for index in indices])

    def shuffle(self, seed: int = 0) -> list[BlurExample]:
        shuffled = list(self._examples)
        random.Random(seed).shuffle(shuffled)
        return shuffled

    def counts_by_class_name(self) -> dict[str, int]:
        counts = Counter(CLASS_NAMES[example.label] for example in self._examples)
        return {name: counts.get(name, 0) for name in CLASS_NAMES}


def _normalize_split(split: str) -> str:
    normalized = split.strip().lower()
    if normalized not in SPLIT_TO_DIRNAME:
        raise ValueError(f"Unsupported split {split!r}; expected one of {sorted(SPLIT_TO_DIRNAME)}")
    return normalized


def _normalize_class_slug(class_slug: str) -> str:
    return normalize(class_slug)


def _example_from_path(image_path: Path, split: str, class_slug: str) -> BlurExample:
    human_name = CLASS_SLUG_TO_NAME[class_slug]
    return BlurExample(
        image_path=str(image_path),
        label=CLASS_NAMES.index(human_name),
        basename=image_path.name,
        class_slug=class_slug,
        split=split,
    )


def _open_image_from_path(image_path: str) -> Image.Image:
    path = Path(image_path)
    if path.suffix.lower() in RAW_EXTENSIONS:
        if rawpy is None:
            raise RuntimeError(
                f"RAW image support requires rawpy, but it is not installed: {path.name}"
            )
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
        return Image.fromarray(rgb).convert("RGB")

    with path.open("rb") as handle:
        pil_image = Image.open(io.BytesIO(handle.read()))
        return pil_image.convert("RGB")


def parse_predicted_class_name(text: str) -> str:
    canonical = find_in_text(text)
    if canonical is not None:
        return CLASS_SLUG_TO_NAME[canonical]
    return text.strip().lower().split(":")[-1].strip()


def _iter_split_directory(dataset_root: Path, split: str) -> list[BlurExample]:
    split_dir = dataset_root / SPLIT_TO_DIRNAME[split]
    examples: list[BlurExample] = []
    for class_slug in CLASS_SLUGS:
        class_dir = split_dir / class_slug
        if not class_dir.exists():
            continue
        if not class_dir.is_dir():
            raise AssertionError(f"Expected directory at {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            examples.append(_example_from_path(path, split=split, class_slug=class_slug))
    return examples


def _read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    import csv

    with manifest_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_examples_from_manifest(dataset_root: Path, manifest_path: Path, split: str) -> list[BlurExample]:
    split_name = SPLIT_TO_DIRNAME[split]
    rows = _read_manifest_rows(manifest_path)
    examples: list[BlurExample] = []
    for row in rows:
        if row.get("split") != split_name:
            continue
        class_slug = _normalize_class_slug(row["class"])
        image_path = dataset_root / split_name / class_slug / row["basename"]
        if not image_path.exists():
            raise AssertionError(
                f"Manifest entry {row['basename']!r} for split={split_name!r}, class={class_slug!r} "
                f"does not exist at {image_path}"
            )
        examples.append(_example_from_path(image_path, split=split, class_slug=class_slug))
    return examples


def _assert_manifest_matches_disk(dataset_root: Path, manifest_path: Path) -> None:
    manifest_examples = [
        (example.split, example.class_slug, example.basename)
        for split in SPLIT_TO_DIRNAME
        for example in _load_examples_from_manifest(dataset_root, manifest_path, split)
    ]
    disk_examples = [
        (example.split, example.class_slug, example.basename)
        for split in SPLIT_TO_DIRNAME
        for example in _iter_split_directory(dataset_root, split)
    ]
    manifest_set = set(manifest_examples)
    disk_set = set(disk_examples)
    if manifest_set == disk_set:
        return

    missing_on_disk = sorted(manifest_set - disk_set)
    missing_in_manifest = sorted(disk_set - manifest_set)
    raise AssertionError(
        "dataset/ folders disagree with split_manifest.csv.\n"
        f"Missing on disk: {missing_on_disk[:10]}\n"
        f"Missing in manifest: {missing_in_manifest[:10]}"
    )


def load_blur_examples(
    dataset_root: str | Path = "dataset",
    manifest_path: str | Path | None = None,
    split: str = "train",
    *,
    prefer_manifest: bool = True,
) -> list[BlurExample]:
    split = _normalize_split(split)
    dataset_root = Path(dataset_root)
    manifest = Path(manifest_path) if manifest_path is not None else dataset_root / "split_manifest.csv"

    if prefer_manifest:
        if not manifest.exists():
            raise FileNotFoundError(
                f"Expected manifest at {manifest}. Rebuild the dataset with split_dataset.py "
                "or pass prefer_manifest=False to fall back to directory walking."
            )
        _assert_manifest_matches_disk(dataset_root, manifest)
        return _load_examples_from_manifest(dataset_root, manifest, split)

    return _iter_split_directory(dataset_root, split)


@chz.chz
class BlurDatasetConfig:
    dataset_root: str = "dataset"
    manifest_path: str | None = None
    dataset_split: Literal["train", "test"]

    model_name_for_tokenizer: str
    renderer_name: str

    num_repeats: float = 1
    batch_size: int = 8
    max_length: int = 8192
    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE
    examples_per_class: int | None = None
    subset_seed: int = 0
    # Preserve subtle blur cues: aggressive downsampling low-passes away the exact
    # signal that separates sharp from slightly soft. Keep this configurable so
    # we can ablate 480 vs 1024 vs 1344.
    max_image_size: int = 1024
    hflip_probability: float = 0.5
    prefer_manifest: bool = True


class BlurClassifierDataset(ClassifierDataset):
    def __init__(self, config: BlurDatasetConfig):
        self.config = config

        tokenizer = get_tokenizer(self.config.model_name_for_tokenizer)
        image_processor = get_image_processor(self.config.model_name_for_tokenizer)
        self.renderer = get_renderer(
            name=self.config.renderer_name,
            tokenizer=tokenizer,
            image_processor=image_processor,
        )

        self.dataset = LocalDatasetSplit(
            load_blur_examples(
                dataset_root=self.config.dataset_root,
                manifest_path=self.config.manifest_path,
                split=self.config.dataset_split,
                prefer_manifest=self.config.prefer_manifest,
            )
        )
        if self.config.examples_per_class is not None and self.config.dataset_split == "train":
            self.dataset = self._sample_per_class(self.dataset)

        self.class_labels = cast(LocalClassLabel, self.dataset.features["label"])
        self.shuffled_indices = self.get_shuffled_indices()

    def _sample_per_class(self, dataset: LocalDatasetSplit) -> LocalDatasetSplit:
        rng = random.Random(self.config.subset_seed)
        class_indices: dict[int, list[int]] = defaultdict(list)
        labels = cast(list[int], dataset["label"])
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)

        selected_indices: list[int] = []
        for label in sorted(class_indices):
            indices = class_indices[label]
            rng.shuffle(indices)
            selected_indices.extend(indices[: self.config.examples_per_class])

        logger.info(
            "Sampled %s examples (%s per class, %s classes)",
            len(selected_indices),
            self.config.examples_per_class,
            len(class_indices),
        )
        return dataset.select(selected_indices)

    def get_class_name(self, label: str) -> str:
        return label

    def open_image(self, example: BlurExample) -> Image.Image:
        return _open_image_from_path(example.image_path)

    def build_messages(self, example: BlurExample) -> list[Message]:
        class_name = self.class_labels.int2str(example.label)
        user_parts: list[ContentPart] = [
            ImagePart(type="image", image=self._load_image_for_training(example)),
            TextPart(type="text", text=USER_PROMPT),
        ]
        assistant_parts: list[ContentPart] = [
            TextPart(type="text", text=f"{ASSISTANT_PREFIX} {class_name}\n"),
        ]
        return [
            Message(role="user", content=user_parts),
            Message(role="assistant", content=assistant_parts),
        ]

    def _load_image_for_training(self, example: BlurExample) -> Image.Image:
        pil_image = resize_image(self.open_image(example), max_size=self.config.max_image_size)
        if random.random() < self.config.hflip_probability:
            pil_image = pil_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return pil_image

    def build_supervised_example(self, example: BlurExample) -> tuple[tinker.ModelInput, torch.Tensor]:
        messages = self.build_messages(example)
        return self.renderer.build_supervised_example(
            messages=messages,
            train_on_what=self.config.train_on_what,
        )

    def get_batch(self, index: int) -> list[tinker.Datum]:
        return [
            datum_from_model_input_weights(
                *self.build_supervised_example(cast(BlurExample, self.dataset[self.shuffled_indices[idx]])),
                max_length=self.config.max_length,
            )
            for idx in range(
                self.config.batch_size * index,
                min(self.config.batch_size * (index + 1), len(self.shuffled_indices)),
            )
        ]


@chz.chz
class BlurDatasetBuilder(SupervisedDatasetBuilder):
    model_name_for_tokenizer: str
    renderer_name: str

    dataset_root: str = "dataset"
    manifest_path: str | None = None

    num_repeats: float = 1
    batch_size: int = 8
    max_length: int = 8192
    train_on_what: TrainOnWhat | None = None
    examples_per_class: int | None = None
    subset_seed: int = 0
    max_image_size: int = 1024
    run_nll_evaluator: bool = False
    prefer_manifest: bool = True

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        default_train_on_what = self.train_on_what or TrainOnWhat.LAST_ASSISTANT_MESSAGE
        train_dataset = BlurClassifierDataset(
            BlurDatasetConfig(
                dataset_root=self.dataset_root,
                manifest_path=self.manifest_path,
                dataset_split="train",
                renderer_name=self.renderer_name,
                model_name_for_tokenizer=self.model_name_for_tokenizer,
                num_repeats=self.num_repeats,
                batch_size=self.batch_size,
                max_length=self.max_length,
                train_on_what=default_train_on_what,
                examples_per_class=self.examples_per_class,
                subset_seed=self.subset_seed,
                max_image_size=self.max_image_size,
                hflip_probability=0.5,
                prefer_manifest=self.prefer_manifest,
            )
        )

        if not self.run_nll_evaluator:
            return train_dataset, None

        test_dataset = BlurClassifierDataset(
            BlurDatasetConfig(
                dataset_root=self.dataset_root,
                manifest_path=self.manifest_path,
                dataset_split="test",
                renderer_name=self.renderer_name,
                model_name_for_tokenizer=self.model_name_for_tokenizer,
                batch_size=self.batch_size,
                max_length=self.max_length,
                train_on_what=default_train_on_what,
                max_image_size=self.max_image_size,
                hflip_probability=0.0,
                prefer_manifest=self.prefer_manifest,
            )
        )
        return train_dataset, test_dataset


@chz.chz
class BlurEvaluatorConfig:
    dataset_root: str = "dataset"
    manifest_path: str | None = None
    dataset_split: Literal["train", "test"] = "test"

    model_name_for_tokenizer: str
    renderer_name: str

    temperature: float = 0.0
    max_tokens: int = 128
    top_p: float = 1.0
    top_k: int = -1
    n_eval: int | None = None
    max_parallel_tasks: int = 128
    max_image_size: int = 1024
    prefer_manifest: bool = True


class BlurClassifierOutput(TypedDict):
    predicted_class_name: str


class BlurClassifierEvaluator(SamplingClientEvaluator):
    def __init__(self, config: BlurEvaluatorConfig):
        self.config = config

        tokenizer = get_tokenizer(self.config.model_name_for_tokenizer)
        image_processor = get_image_processor(self.config.model_name_for_tokenizer)
        self.renderer = renderers.get_renderer(
            name=self.config.renderer_name,
            tokenizer=tokenizer,
            image_processor=image_processor,
        )

        self.dataset = LocalDatasetSplit(
            load_blur_examples(
                dataset_root=self.config.dataset_root,
                manifest_path=self.config.manifest_path,
                split=self.config.dataset_split,
                prefer_manifest=self.config.prefer_manifest,
            )
        )
        self.shuffled_dataset = self.dataset.shuffle(seed=0)
        self.class_labels = cast(LocalClassLabel, self.dataset.features["label"])

    def build_generation_prompt(self, example: BlurExample) -> tinker.ModelInput:
        pil_image = resize_image(
            image=_open_image_from_path(example.image_path),
            max_size=self.config.max_image_size,
        )
        messages = [
            Message(
                role="user",
                content=[
                    ImagePart(type="image", image=pil_image),
                    TextPart(type="text", text=USER_PROMPT),
                ],
            )
        ]
        return self.renderer.build_generation_prompt(
            messages=messages,
            role="assistant",
            prefill=ASSISTANT_PREFIX,
        )

    async def generate_output(
        self,
        model_input: tinker.ModelInput,
        sampling_client: tinker.SamplingClient,
        sampling_params: types.SamplingParams,
    ) -> BlurClassifierOutput:
        response: types.SampleResponse = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        tokens: list[int] = response.sequences[0].tokens
        rendered_message = self.renderer.parse_response(tokens)[0]
        predicted_class_name = parse_predicted_class_name(get_text_content(rendered_message))
        return BlurClassifierOutput(predicted_class_name=predicted_class_name)

    def get_metrics_for_output(
        self,
        example: BlurExample,
        classifier_output: BlurClassifierOutput,
    ) -> dict[str, float]:
        expected_class_name = self.class_labels.int2str(example.label)
        return {"accuracy": float(classifier_output["predicted_class_name"] == expected_class_name)}

    async def evaluate_details(
        self,
        sampling_client: tinker.SamplingClient,
    ) -> tuple[dict[str, float], list[tuple[BlurExample, BlurClassifierOutput]]]:
        sampling_params = types.SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            stop=self.renderer.get_stop_sequences(),
        )

        num_examples = min(len(self.shuffled_dataset), self.config.n_eval or len(self.shuffled_dataset))
        semaphore = asyncio.Semaphore(self.config.max_parallel_tasks)

        async def bounded_generate_output(example: BlurExample) -> BlurClassifierOutput:
            async with semaphore:
                return await self.generate_output(
                    self.build_generation_prompt(example),
                    sampling_client,
                    sampling_params,
                )

        tasks = [asyncio.create_task(bounded_generate_output(self.shuffled_dataset[idx])) for idx in range(num_examples)]
        with timed("sample outputs", {}):
            outputs = await asyncio.gather(*tasks)

        pairs = [(self.shuffled_dataset[idx], outputs[idx]) for idx in range(num_examples)]
        metrics_per_example = [self.get_metrics_for_output(example, output) for example, output in pairs]
        aggregated_metrics = {
            key: np.mean([example_metrics[key] for example_metrics in metrics_per_example]).item()
            for key in metrics_per_example[0]
        }
        return aggregated_metrics, pairs

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        metrics, _pairs = await self.evaluate_details(sampling_client)
        return metrics


@chz.chz
class BlurEvaluatorBuilder:
    model_name_for_tokenizer: str
    renderer_name: str

    dataset_root: str = "dataset"
    manifest_path: str | None = None

    temperature: float = 0.0
    max_tokens: int = 128
    top_p: float = 1.0
    top_k: int = -1
    n_eval: int | None = None
    max_parallel_tasks: int = 128
    max_image_size: int = 1024
    prefer_manifest: bool = True

    def __call__(self) -> BlurClassifierEvaluator:
        return BlurClassifierEvaluator(
            BlurEvaluatorConfig(
                dataset_root=self.dataset_root,
                manifest_path=self.manifest_path,
                dataset_split="test",
                renderer_name=self.renderer_name,
                model_name_for_tokenizer=self.model_name_for_tokenizer,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
                n_eval=self.n_eval,
                max_parallel_tasks=self.max_parallel_tasks,
                max_image_size=self.max_image_size,
                prefer_manifest=self.prefer_manifest,
            )
        )


def confusion_matrix_from_pairs(
    pairs: list[tuple[BlurExample, BlurClassifierOutput]],
) -> tuple[list[list[int]], dict[str, dict[str, float]]]:
    index_for_class = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}
    confusion = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]

    for example, output in pairs:
        actual = CLASS_NAMES[example.label]
        predicted = output["predicted_class_name"]
        if predicted not in index_for_class:
            continue
        confusion[index_for_class[actual]][index_for_class[predicted]] += 1

    metrics: dict[str, dict[str, float]] = {}
    for class_name, class_index in index_for_class.items():
        tp = confusion[class_index][class_index]
        fp = sum(confusion[row][class_index] for row in range(len(CLASS_NAMES)) if row != class_index)
        fn = sum(confusion[class_index][col] for col in range(len(CLASS_NAMES)) if col != class_index)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics[class_name] = {"precision": precision, "recall": recall}

    return confusion, metrics
