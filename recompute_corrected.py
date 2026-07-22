#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from eval_blur import wilson_interval
from review_analysis import (
    DEFAULT_ADJUDICATIONS,
    EVAL_RUN_PATHS,
    AccuracySummary,
    LabelSource,
    build_adjudicated_source,
    build_original_source,
    build_pairwise_agreement_matrix,
    build_review_source,
    build_stability_rows,
    compute_accuracy,
    compute_agreement,
    compute_confusion,
    compute_precision_recall,
    join_eval_records,
    load_eval_records,
    load_review_records,
    parse_adjudications,
)


DEFAULT_REVIEW_PATH = Path("results/blind_full_review.csv")
EXPECTED_DEFAULT_REGRESSION = {
    "agreement": (31, 42),
    "C_final": {"as-originally-labeled": 28, "blind-corrected": 30},
    "D_final": {"as-originally-labeled": 29, "blind-corrected": 27},
    "D_step20": {"as-originally-labeled": 33, "blind-corrected": 35},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", default=str(DEFAULT_REVIEW_PATH))
    parser.add_argument(
        "--adjudicate",
        action="append",
        default=[],
        help="Comma-separated NAME=label overrides applied on top of blind relabels.",
    )
    parser.add_argument(
        "--reviewer",
        action="append",
        default=[],
        help="Additional review CSVs in name=path.csv form.",
    )
    return parser.parse_args()


def parse_reviewer_specs(specs: list[str]) -> dict[str, Path]:
    reviewers: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid reviewer spec {spec!r}; expected name=path.csv")
        name, raw_path = spec.split("=", 1)
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError(f"Invalid reviewer spec {spec!r}; missing reviewer name")
        if normalized_name in reviewers:
            raise ValueError(f"Duplicate reviewer name {normalized_name!r}")
        reviewers[normalized_name] = Path(raw_path.strip())
    return reviewers


def load_label_sources(
    review_csv: Path,
    *,
    adjudications: dict[str, str],
    reviewers: dict[str, Path],
) -> tuple[LabelSource, LabelSource, LabelSource, list[LabelSource]]:
    blind_records = load_review_records(review_csv, source_name="blind-corrected")
    original_source = build_original_source(blind_records)
    original_source = LabelSource(
        name="as-originally-labeled",
        labels_by_file=original_source.labels_by_file,
        notes_by_file=original_source.notes_by_file,
    )
    blind_source = build_review_source(blind_records)
    adjudicated_source = build_adjudicated_source(blind_source, adjudications)

    additional_sources: list[LabelSource] = []
    for reviewer_name, reviewer_path in reviewers.items():
        reviewer_records = load_review_records(reviewer_path, source_name=reviewer_name)
        reviewer_source = build_review_source(reviewer_records)
        if reviewer_source.labels_by_file.keys() != blind_source.labels_by_file.keys():
            raise AssertionError(
                f"Reviewer {reviewer_name!r} does not cover the same image ids as the blind review"
            )
        additional_sources.append(reviewer_source)
    return original_source, blind_source, adjudicated_source, additional_sources


def load_eval_runs() -> dict[str, list]:
    eval_runs: dict[str, list] = {}
    for run_name, path in EVAL_RUN_PATHS.items():
        eval_runs[run_name] = load_eval_records(path, run_name=run_name)
    return eval_runs


def print_agreement_section(original_source: LabelSource, blind_source: LabelSource) -> AccuracySummary:
    agreement = compute_agreement(original_source, blind_source)
    pct = 100.0 * agreement.correct / agreement.total
    print(
        f"Blind re-review: {agreement.total} images, "
        f"{agreement.correct}/{agreement.total} agree with original ({pct:.1f}%)"
    )
    print("\nOriginal -> Blind relabel confusion:")
    confusion = compute_confusion(original_source, blind_source)
    for actual_label, predicted_label in sorted(confusion):
        count = confusion[(actual_label, predicted_label)]
        marker = "" if actual_label == predicted_label else "  <-- disagreement"
        print(f"  {actual_label:20s} -> {predicted_label:20s} : {count}{marker}")

    disagreements = [
        file_id for file_id, original_label in original_source.labels_by_file.items()
        if original_label != blind_source.labels_by_file[file_id]
    ]
    print("\nDisagreeing images:")
    for file_id in disagreements:
        print(
            f"  {file_id}: {original_source.labels_by_file[file_id]} -> "
            f"{blind_source.labels_by_file[file_id]}  "
            f"note={blind_source.notes_by_file[file_id] or '-'}"
        )
    return agreement


def print_accuracy_table(
    eval_runs: dict[str, list],
    label_sources: list[LabelSource],
) -> dict[str, dict[str, AccuracySummary]]:
    print("\nAccuracy by run:")
    print(
        "run".ljust(12)
        + "  "
        + "  ".join(source.name.rjust(32) for source in label_sources)
    )
    summaries: dict[str, dict[str, AccuracySummary]] = {}
    for run_name, eval_records in eval_runs.items():
        run_summaries: dict[str, AccuracySummary] = {}
        cells: list[str] = []
        for source in label_sources:
            summary = compute_accuracy(eval_records, source)
            low, high = wilson_interval(summary.correct, summary.total)
            cells.append(
                f"{summary.correct:2d}/{summary.total:<2d} "
                f"({100.0 * summary.correct / summary.total:5.1f}%, [{low:.3f}, {high:.3f}])"
            )
            run_summaries[source.name] = summary
        summaries[run_name] = run_summaries
        print(run_name.ljust(12) + "  " + "  ".join(cell.rjust(32) for cell in cells))
    return summaries


def print_precision_recall_tables(
    eval_runs: dict[str, list],
    label_sources: list[LabelSource],
) -> None:
    for run_name, eval_records in eval_runs.items():
        print(f"\nPer-class precision/recall for {run_name}:")
        for source in label_sources:
            print(f"  label set: {source.name}")
            metrics = compute_precision_recall(eval_records, source)
            for class_name, class_metrics in metrics.items():
                print(
                    f"    {class_name:20s} "
                    f"precision={class_metrics['precision']:.4f} "
                    f"recall={class_metrics['recall']:.4f}"
                )


def print_pairwise_review_agreement(
    original_source: LabelSource,
    blind_source: LabelSource,
    adjudicated_source: LabelSource,
    additional_sources: list[LabelSource],
) -> None:
    reviewer_sources = [blind_source, *additional_sources]
    if len(reviewer_sources) < 2:
        return

    sources = [original_source, blind_source, adjudicated_source, *additional_sources]
    matrix = build_pairwise_agreement_matrix(sources)
    header = ["source", *[source.name for source in sources]]
    widths = [max(len(cell), 12) for cell in header]
    print("\nPairwise agreement matrix:")
    print("  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(header)))
    for row_source, row in zip(sources, matrix, strict=True):
        cells = [row_source.name]
        for summary in row:
            pct = 100.0 * summary.correct / summary.total
            cells.append(f"{summary.correct}/{summary.total} ({pct:.1f}%)")
        print("  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells)))

    print("\nPer-image stability table:")
    for file_id, labels in build_stability_rows(sources):
        rendered = ", ".join(f"{name}={label}" for name, label in labels.items())
        print(f"  {file_id}: {rendered}")


def assert_default_regression(
    agreement: AccuracySummary,
    accuracy_summaries: dict[str, dict[str, AccuracySummary]],
    *,
    using_default_inputs: bool,
) -> None:
    if not using_default_inputs:
        return

    if (agreement.correct, agreement.total) != EXPECTED_DEFAULT_REGRESSION["agreement"]:
        raise AssertionError(
            "Blind review regression changed: "
            f"expected {EXPECTED_DEFAULT_REGRESSION['agreement']}, "
            f"got {(agreement.correct, agreement.total)}"
        )

    for run_name, expectations in EXPECTED_DEFAULT_REGRESSION.items():
        if run_name == "agreement":
            continue
        for source_name, expected_correct in expectations.items():
            observed = accuracy_summaries[run_name][source_name].correct
            if observed != expected_correct:
                raise AssertionError(
                    f"{run_name} {source_name} regression changed: "
                    f"expected {expected_correct}, got {observed}"
                )


def main() -> None:
    args = parse_args()
    adjudications = dict(DEFAULT_ADJUDICATIONS)
    adjudications.update(parse_adjudications(args.adjudicate))
    reviewers = parse_reviewer_specs(args.reviewer)

    original_source, blind_source, adjudicated_source, additional_sources = load_label_sources(
        Path(args.review_csv),
        adjudications=adjudications,
        reviewers=reviewers,
    )
    eval_runs = load_eval_runs()
    for eval_records in eval_runs.values():
        join_eval_records(eval_records, blind_source)
        print(f"{eval_records[0].run_name}: matched {len(eval_records)}/{len(blind_source.labels_by_file)} review rows")

    agreement = print_agreement_section(original_source, blind_source)
    label_sources = [original_source, blind_source, adjudicated_source]
    accuracy_summaries = print_accuracy_table(eval_runs, label_sources)
    print_precision_recall_tables(eval_runs, label_sources)
    print_pairwise_review_agreement(
        original_source,
        blind_source,
        adjudicated_source,
        additional_sources,
    )
    assert_default_regression(
        agreement,
        accuracy_summaries,
        using_default_inputs=(
            Path(args.review_csv) == DEFAULT_REVIEW_PATH
            and not args.adjudicate
            and not args.reviewer
        ),
    )


if __name__ == "__main__":
    main()
