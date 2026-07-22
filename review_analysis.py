from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from blur_labels import CLASSES, normalize

DEFAULT_ADJUDICATIONS = {
    "DSC06207": "unintentional_blur",
    "DSC05659": "unintentional_blur",
}
REVIEW_FIELDNAMES = ("file", "original_label", "blind_relabel", "note")
EVAL_RUN_PATHS = {
    "C_final": Path("results/eval_runC_1344.csv"),
    "D_final": Path("results/eval_runD_1344.csv"),
    "D_step20": Path("results/eval_runD_step20_1344.csv"),
}


@dataclass(frozen=True)
class ReviewRecord:
    source_name: str
    file_id: str
    original_label: str
    reviewed_label: str
    note: str


@dataclass(frozen=True)
class EvalRecord:
    run_name: str
    file_id: str
    basename: str
    image_path: Path
    actual_label: str
    majority_pred: str
    majority_correct: bool


@dataclass(frozen=True)
class LabelSource:
    name: str
    labels_by_file: dict[str, str]
    notes_by_file: dict[str, str]


@dataclass(frozen=True)
class AccuracySummary:
    correct: int
    total: int


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_adjudications(specs: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for spec in specs:
        for assignment in spec.split(","):
            assignment = assignment.strip()
            if not assignment:
                continue
            if "=" not in assignment:
                raise ValueError(
                    f"Invalid adjudication {assignment!r}; expected NAME=label"
                )
            file_id, raw_label = assignment.split("=", 1)
            normalized_file = file_id.strip()
            if not normalized_file:
                raise ValueError(f"Invalid adjudication {assignment!r}; missing file id")
            parsed[normalized_file] = normalize(raw_label)
    return parsed


def load_review_records(
    path: Path,
    *,
    source_name: str,
    expected_count: int = 42,
    row_loader: Callable[[Path], list[dict[str, str]]] = load_csv_rows,
) -> list[ReviewRecord]:
    rows = row_loader(path)
    if len(rows) != expected_count:
        raise AssertionError(f"Expected {expected_count} rows in {path}, found {len(rows)}")

    seen: set[str] = set()
    records: list[ReviewRecord] = []
    for row in rows:
        if set(row) != set(REVIEW_FIELDNAMES):
            raise AssertionError(
                f"{path} must have columns {list(REVIEW_FIELDNAMES)}, found {list(row)}"
            )
        file_id = row["file"].strip()
        if not file_id:
            raise AssertionError(f"{path} contains an empty file id")
        if file_id in seen:
            raise AssertionError(f"{path} contains duplicate file id {file_id!r}")
        seen.add(file_id)
        records.append(
            ReviewRecord(
                source_name=source_name,
                file_id=file_id,
                original_label=normalize(row["original_label"]),
                reviewed_label=normalize(row["blind_relabel"]),
                note=row["note"].strip(),
            )
        )
    return sorted(records, key=lambda record: record.file_id)


def load_eval_records(
    path: Path,
    *,
    run_name: str,
    expected_count: int = 42,
    row_loader: Callable[[Path], list[dict[str, str]]] = load_csv_rows,
) -> list[EvalRecord]:
    rows = row_loader(path)
    if len(rows) != expected_count:
        raise AssertionError(f"Expected {expected_count} rows in {path}, found {len(rows)}")

    seen: set[str] = set()
    records: list[EvalRecord] = []
    for row in rows:
        required = {"image_path", "basename", "actual", "majority_pred", "majority_correct"}
        missing = required - set(row)
        if missing:
            raise AssertionError(f"{path} missing columns {sorted(missing)}")
        basename = row["basename"].strip()
        file_id = Path(basename).stem
        if file_id in seen:
            raise AssertionError(f"{path} contains duplicate basename {basename!r}")
        seen.add(file_id)
        records.append(
            EvalRecord(
                run_name=run_name,
                file_id=file_id,
                basename=basename,
                image_path=Path(row["image_path"]),
                actual_label=normalize(row["actual"]),
                majority_pred=normalize(row["majority_pred"]),
                majority_correct=_parse_bool(row["majority_correct"]),
            )
        )
    return sorted(records, key=lambda record: record.file_id)


def build_review_source(records: list[ReviewRecord]) -> LabelSource:
    if not records:
        raise AssertionError("Expected at least one review record")
    return LabelSource(
        name=records[0].source_name,
        labels_by_file={record.file_id: record.reviewed_label for record in records},
        notes_by_file={record.file_id: record.note for record in records},
    )


def build_original_source(records: list[ReviewRecord]) -> LabelSource:
    return LabelSource(
        name="original",
        labels_by_file={record.file_id: record.original_label for record in records},
        notes_by_file={record.file_id: record.note for record in records},
    )


def build_adjudicated_source(
    blind_source: LabelSource,
    overrides: dict[str, str],
) -> LabelSource:
    labels = dict(blind_source.labels_by_file)
    for file_id, label in overrides.items():
        if file_id not in labels:
            raise AssertionError(f"Adjudication references unknown file id {file_id!r}")
        labels[file_id] = label
    return LabelSource(
        name="adjudicated",
        labels_by_file=labels,
        notes_by_file=dict(blind_source.notes_by_file),
    )


def join_eval_records(
    eval_records: list[EvalRecord],
    review_source: LabelSource,
) -> list[EvalRecord]:
    missing = sorted(set(review_source.labels_by_file) - {record.file_id for record in eval_records})
    if missing:
        raise AssertionError(
            f"{eval_records[0].run_name if eval_records else 'eval'} missing review rows for {missing[:10]}"
        )
    matched = [record for record in eval_records if record.file_id in review_source.labels_by_file]
    if len(matched) != len(review_source.labels_by_file):
        raise AssertionError(
            f"Expected {len(review_source.labels_by_file)}/{len(review_source.labels_by_file)} matches "
            f"for {eval_records[0].run_name if eval_records else 'eval'}, found {len(matched)}"
        )
    return matched


def compute_agreement(left: LabelSource, right: LabelSource) -> AccuracySummary:
    _assert_same_keys(left, right)
    correct = sum(
        1
        for file_id, left_label in left.labels_by_file.items()
        if left_label == right.labels_by_file[file_id]
    )
    return AccuracySummary(correct=correct, total=len(left.labels_by_file))


def compute_accuracy(eval_records: list[EvalRecord], source: LabelSource) -> AccuracySummary:
    matched_records = join_eval_records(eval_records, source)
    correct = sum(
        1
        for record in matched_records
        if record.majority_pred == source.labels_by_file[record.file_id]
    )
    return AccuracySummary(correct=correct, total=len(matched_records))


def compute_confusion(
    actual_source: LabelSource,
    predicted_source: LabelSource,
) -> Counter[tuple[str, str]]:
    _assert_same_keys(actual_source, predicted_source)
    return Counter(
        (actual_label, predicted_source.labels_by_file[file_id])
        for file_id, actual_label in actual_source.labels_by_file.items()
    )


def compute_precision_recall(
    eval_records: list[EvalRecord],
    source: LabelSource,
) -> dict[str, dict[str, float]]:
    matched_records = join_eval_records(eval_records, source)
    confusion = [[0 for _ in CLASSES] for _ in CLASSES]
    index_for_class = {class_name: idx for idx, class_name in enumerate(CLASSES)}
    for record in matched_records:
        actual = source.labels_by_file[record.file_id]
        predicted = record.majority_pred
        confusion[index_for_class[actual]][index_for_class[predicted]] += 1

    metrics: dict[str, dict[str, float]] = {}
    for class_name, class_index in index_for_class.items():
        tp = confusion[class_index][class_index]
        fp = sum(confusion[row][class_index] for row in range(len(CLASSES)) if row != class_index)
        fn = sum(confusion[class_index][col] for col in range(len(CLASSES)) if col != class_index)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics[class_name] = {"precision": precision, "recall": recall}
    return metrics


def build_pairwise_agreement_matrix(
    sources: list[LabelSource],
) -> list[list[AccuracySummary]]:
    return [[compute_agreement(left, right) for right in sources] for left in sources]


def build_stability_rows(sources: list[LabelSource]) -> list[tuple[str, dict[str, str]]]:
    keys = set(sources[0].labels_by_file)
    for source in sources[1:]:
        if set(source.labels_by_file) != keys:
            raise AssertionError("Label sources do not cover the same files")

    rows: list[tuple[str, dict[str, str]]] = []
    for file_id in sorted(keys):
        labels = {source.name: source.labels_by_file[file_id] for source in sources}
        if len(set(labels.values())) > 1:
            rows.append((file_id, labels))
    return rows


def label_for_display(label: str) -> str:
    return label


def _assert_same_keys(left: LabelSource, right: LabelSource) -> None:
    if set(left.labels_by_file) != set(right.labels_by_file):
        raise AssertionError(
            f"Label sources {left.name!r} and {right.name!r} do not cover the same files"
        )


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Expected boolean-like value, found {value!r}")
