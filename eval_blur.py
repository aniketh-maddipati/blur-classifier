"""
Evaluate a blur-classifier checkpoint on the local holdout split.
"""

from __future__ import annotations

import asyncio

import chz
import tinker

from blur_dataset import BlurEvaluatorBuilder, CLASS_NAMES, confusion_matrix_from_pairs
from tinker_client import disable_pyqwest_transport
from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.utils.git_rev import recipe_user_metadata


@chz.chz
class EvalConfig:
    model_path: str
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
    max_image_size: int = 1024
    prefer_manifest: bool = True


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


def run_eval(eval_config: EvalConfig) -> None:
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

        async def main() -> None:
            metrics, pairs = await evaluator.evaluate_details(sampling_client)
            confusion, class_metrics = confusion_matrix_from_pairs(pairs)
            unparseable = sum(
                1 for _example, output in pairs if output["predicted_class_name"] not in CLASS_NAMES
            )
            print(f"Overall accuracy: {metrics['accuracy']:.4f}")
            if unparseable:
                print(f"Unparseable predictions: {unparseable}")
            print("\nPer-class precision / recall:")
            for class_name in CLASS_NAMES:
                class_result = class_metrics[class_name]
                print(
                    f"  {class_name}: precision={class_result['precision']:.4f} "
                    f"recall={class_result['recall']:.4f}"
                )
            print("\nConfusion matrix:")
            print(_format_confusion(confusion))

        asyncio.run(main())


if __name__ == "__main__":
    run_eval(chz.entrypoint(EvalConfig))
