"""
Evaluate a blur-classifier checkpoint on the local holdout split.

Changes vs the original version:
  - `max_image_size` is REQUIRED (no default). Evals must run at the same
    resolution the checkpoint was trained at; a silent 1024 default caused
    train/eval resolution mismatch for 1344-trained runs.
  - `repeats` runs the full eval N times against the same checkpoint so
    serving-side nondeterminism (MoE batch effects at temperature=0) can be
    measured instead of trusted away. Reports per-run accuracy, mean/min/max,
    a Wilson 95% interval, and a majority-vote confusion matrix.
  - Per-image predictions for every repeat are written to a CSV so unstable
    ("coin-flip") images can be identified individually.

Usage:
  python eval_blur.py \
    model_path=tinker://.../sampler_weights/final \
    max_image_size=1344 \
    repeats=5 \
    per_image_csv=results/eval_per_image.csv
"""

from __future__ import annotations

import asyncio
import csv
import math
from collections import Counter
from pathlib import Path

import chz
import tinker

from blur_dataset import (
    BlurEvaluatorBuilder,
    CLASS_NAMES,
    BlurClassifierOutput,
    BlurExample,
    confusion_matrix_from_pairs,
)
from tinker_client import disable_pyqwest_transport
from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.utils.git_rev import recipe_user_metadata


@chz.chz
class EvalConfig:
    model_path: str
    # REQUIRED: must match the max_image_size the checkpoint was trained with.
    # (Runs trained at 1344 were previously evaluated at a silent 1024 default;
    # blur classification is acutely sensitive to resize resolution.)
    max_image_size: int

    dataset_root: str = "dataset"
    manifest_path: str | None = None

    renderer_name: str | None = "qwen3_5_disable_thinking"
    model_name: str | None = "Qwen/Qwen3.6-35B-A3B"
    base_url: str | None = None

    temperature: float = 0.0
    max_tokens: int = 128
    top_p: float = 1.0
    top_k: int = -1
    n_eval: int | None = None
    max_parallel_tasks: int = 128
    prefer_manifest: bool = True

    # Number of full eval passes over the holdout. >=3 recommended until
    # serving-side nondeterminism is characterized; results are aggregated.
    repeats: int = 1
    # Per-image predictions for every repeat land here (one row per image).
    per_image_csv: str | None = "results/eval_per_image.csv"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    p_hat = successes / total
    denom = 1.0 + (z**2) / total
    center = (p_hat + (z**2) / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / total + (z**2) / (4 * total**2))
    return (max(0.0, center - margin), min(1.0, center + margin))


def _format_confusion(confusion: list[list[int]]) -> str:
    header = ["actual \\ predicted", *CLASS_NAMES]
    widths = [max(len(item), 18) for item in header]
    rows = [
        "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(header)),
    ]
    for class_name, counts in zip(CLASS_NAMES, confusion, strict=True):
        row = [class_name, *[str(count) for count in counts]]
        rows.append("  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))
    return "\n".join(rows)


def _report_single_run(
    run_index: int,
    metrics: dict[str, float],
    pairs: list[tuple[BlurExample, BlurClassifierOutput]],
) -> None:
    confusion, class_metrics = confusion_matrix_from_pairs(pairs)
    unparseable = sum(
        1 for _example, output in pairs if output["predicted_class_name"] not in CLASS_NAMES
    )
    n_total = len(pairs)
    n_correct = round(metrics["accuracy"] * n_total)
    low, high = wilson_interval(n_correct, n_total)
    print(f"\n=== Run {run_index}: accuracy {metrics['accuracy']:.4f} "
          f"({n_correct}/{n_total}, Wilson 95% [{low:.3f}, {high:.3f}]) ===")
    if unparseable:
        print(f"Unparseable predictions: {unparseable}")
    print("Per-class precision / recall:")
    for class_name in CLASS_NAMES:
        class_result = class_metrics[class_name]
        print(
            f"  {class_name}: precision={class_result['precision']:.4f} "
            f"recall={class_result['recall']:.4f}"
        )
    print("Confusion matrix:")
    print(_format_confusion(confusion))


def _majority_vote(predictions: list[str]) -> str:
    counts = Counter(predictions)
    top = counts.most_common()
    # Deterministic tie-break: alphabetical among the tied top predictions.
    best_count = top[0][1]
    tied = sorted(name for name, count in top if count == best_count)
    return tied[0]


def _write_per_image_csv(
    csv_path: Path,
    all_runs: list[list[tuple[BlurExample, BlurClassifierOutput]]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    n_repeats = len(all_runs)
    # Runs iterate the holdout in a fixed order (shuffle seed=0), so index i is
    # the same image across runs; assert rather than trust.
    reference = all_runs[0]
    header = (
        ["image_path", "basename", "actual"]
        + [f"pred_run{r + 1}" for r in range(n_repeats)]
        + ["n_correct", "flipped", "majority_pred", "majority_correct"]
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for idx, (example, _output) in enumerate(reference):
            preds: list[str] = []
            for run in all_runs:
                run_example, run_output = run[idx]
                if run_example.image_path != example.image_path:
                    raise AssertionError(
                        f"Eval ordering mismatch at index {idx}: "
                        f"{run_example.image_path} != {example.image_path}"
                    )
                preds.append(run_output["predicted_class_name"])
            actual = CLASS_NAMES[example.label]
            n_correct = sum(1 for pred in preds if pred == actual)
            flipped = len(set(preds)) > 1
            majority = _majority_vote(preds)
            writer.writerow(
                [example.image_path, example.basename, actual]
                + preds
                + [n_correct, flipped, majority, majority == actual]
            )


def _report_aggregate(
    all_metrics: list[dict[str, float]],
    all_runs: list[list[tuple[BlurExample, BlurClassifierOutput]]],
) -> None:
    n_total = len(all_runs[0])
    accuracies = [metrics["accuracy"] for metrics in all_metrics]
    mean_acc = sum(accuracies) / len(accuracies)
    print(f"\n===== Aggregate over {len(all_runs)} runs =====")
    print(f"Accuracy per run: {', '.join(f'{acc:.4f}' for acc in accuracies)}")
    print(f"Mean {mean_acc:.4f}  min {min(accuracies):.4f}  max {max(accuracies):.4f}  "
          f"spread {max(accuracies) - min(accuracies):.4f}")

    # Flip analysis: which images are unstable across repeats?
    flip_count = 0
    for idx in range(n_total):
        preds = {run[idx][1]["predicted_class_name"] for run in all_runs}
        if len(preds) > 1:
            flip_count += 1
    print(f"Images with unstable predictions across runs: {flip_count}/{n_total}")

    # Majority-vote pairs -> confusion matrix with Wilson interval.
    majority_pairs: list[tuple[BlurExample, BlurClassifierOutput]] = []
    for idx in range(n_total):
        example = all_runs[0][idx][0]
        preds = [run[idx][1]["predicted_class_name"] for run in all_runs]
        majority_pairs.append(
            (example, BlurClassifierOutput(predicted_class_name=_majority_vote(preds)))
        )
    confusion, class_metrics = confusion_matrix_from_pairs(majority_pairs)
    n_correct = sum(
        1
        for example, output in majority_pairs
        if output["predicted_class_name"] == CLASS_NAMES[example.label]
    )
    low, high = wilson_interval(n_correct, n_total)
    print(f"\nMajority-vote accuracy: {n_correct / n_total:.4f} "
          f"({n_correct}/{n_total}, Wilson 95% [{low:.3f}, {high:.3f}])")
    print("Majority-vote per-class recall (Wilson 95%):")
    for class_index, class_name in enumerate(CLASS_NAMES):
        row = confusion[class_index]
        class_total = sum(row)
        class_correct = row[class_index]
        class_low, class_high = wilson_interval(class_correct, class_total)
        print(f"  {class_name}: {class_correct}/{class_total} "
              f"[{class_low:.3f}, {class_high:.3f}]")
    print("Majority-vote confusion matrix:")
    print(_format_confusion(confusion))


def run_eval(eval_config: EvalConfig) -> None:
    if eval_config.repeats < 1:
        raise ValueError("repeats must be >= 1")

    with disable_pyqwest_transport():
        service_client = tinker.ServiceClient(
            base_url=eval_config.base_url,
            user_metadata=recipe_user_metadata("eval_vlm_classifier_blur_local"),
        )
        sampling_client = service_client.create_sampling_client(model_path=eval_config.model_path)

        resolved_model_name = eval_config.model_name
        resolved_renderer_name = eval_config.renderer_name

        rest_client = service_client.create_rest_client()
        try:
            training_run = rest_client.get_training_run_by_tinker_path(eval_config.model_path).result()
        except Exception:
            training_run = None

        if training_run is not None:
            if resolved_model_name is not None and resolved_model_name != training_run.base_model:
                raise ValueError(
                    f"Model name {resolved_model_name} does not match checkpoint base model {training_run.base_model}"
                )
            resolved_model_name = resolved_model_name or training_run.base_model

        resolved_renderer_name = (
            resolved_renderer_name
            or checkpoint_utils.get_renderer_name_from_checkpoint(service_client, eval_config.model_path)
        )
        resolved_model_name = resolved_model_name or eval_config.model_path
        if resolved_renderer_name is None:
            resolved_renderer_name = model_info.get_recommended_renderer_name(resolved_model_name)

        evaluator_builder = BlurEvaluatorBuilder(
            model_name_for_tokenizer=resolved_model_name,
            renderer_name=resolved_renderer_name,
            dataset_root=eval_config.dataset_root,
            manifest_path=eval_config.manifest_path,
            temperature=eval_config.temperature,
            max_tokens=eval_config.max_tokens,
            top_p=eval_config.top_p,
            top_k=eval_config.top_k,
            n_eval=eval_config.n_eval,
            max_parallel_tasks=eval_config.max_parallel_tasks,
            max_image_size=eval_config.max_image_size,
            prefer_manifest=eval_config.prefer_manifest,
        )
        evaluator = evaluator_builder()

        print(f"Evaluating {eval_config.model_path}")
        print(f"max_image_size={eval_config.max_image_size}  "
              f"temperature={eval_config.temperature}  repeats={eval_config.repeats}")

        async def main() -> None:
            all_metrics: list[dict[str, float]] = []
            all_runs: list[list[tuple[BlurExample, BlurClassifierOutput]]] = []
            for run_index in range(1, eval_config.repeats + 1):
                metrics, pairs = await evaluator.evaluate_details(sampling_client)
                all_metrics.append(metrics)
                all_runs.append(pairs)
                _report_single_run(run_index, metrics, pairs)

            if eval_config.repeats > 1:
                _report_aggregate(all_metrics, all_runs)

            if eval_config.per_image_csv:
                csv_path = Path(eval_config.per_image_csv)
                _write_per_image_csv(csv_path, all_runs)
                print(f"\nPer-image predictions written to {csv_path}")

        asyncio.run(main())


if __name__ == "__main__":
    run_eval(chz.entrypoint(EvalConfig))