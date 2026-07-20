"""
Fine-tune the base model on the training set.

The training-example builder (build_training_examples) is fully implemented —
it walks TRAIN_DIR, loads and resizes each image, and pairs it with its class label.

The actual fine-tuning loop is left as a clearly marked TODO.  Before filling it in,
consult the vlm_classifier recipe in the Tinker cookbook:
    https://github.com/thinking-machines-lab/tinker-cookbook
The SFT loop syntax (forward_backward, optim_step, save_weights, etc.) should
come from that recipe, not be invented here.

Usage:
    python train.py
"""

from pathlib import Path

from config import CLASSES, MODEL_NAME, TRAIN_DIR
from classify import resize_image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".raw", ".cr2", ".nef", ".arw"}

# Three learning rates to sweep — adjust after reviewing baseline results.
LEARNING_RATES: list[float] = [1e-5, 5e-6, 1e-6]


# ---------------------------------------------------------------------------
# Training-example builder (concrete — safe to generate)
# ---------------------------------------------------------------------------

def build_training_examples(train_dir: str) -> list[dict]:
    """
    Walk train_dir/<class>/ for each class in CLASSES.
    Returns a list of {"image_bytes": bytes, "label": str} dicts.
    """
    examples: list[dict] = []

    for cls in CLASSES:
        class_dir = Path(train_dir) / cls
        if not class_dir.exists():
            print(f"  [warn] {class_dir} not found — skipping")
            continue

        images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        print(f"  {cls}: {len(images)} image(s)")

        for img_path in images:
            image_bytes = resize_image(str(img_path))
            examples.append({"image_bytes": image_bytes, "label": cls})

    return examples


# ---------------------------------------------------------------------------
# Fine-tuning loop (TODO — fill in from tinker-cookbook vlm_classifier recipe)
# ---------------------------------------------------------------------------

def run_finetune(examples: list[dict], learning_rate: float, run_name: str) -> None:
    """
    Run one SFT pass over examples at learning_rate and save the checkpoint
    to results/<run_name>.

    TODO: Implement this function using the vlm_classifier recipe from
          https://github.com/thinking-machines-lab/tinker-cookbook

          The loop will look roughly like:

            model = load_model(MODEL_NAME)                # TODO: verify API
            optimizer = make_optimizer(lr=learning_rate)  # TODO: verify API

            for batch in batches(examples, batch_size=...):
                loss = model.forward_backward(batch)      # TODO: verify API
                optimizer.step()                          # TODO: verify API

            model.save_weights(f"results/{run_name}")     # TODO: verify API

          Do NOT invent these call signatures — copy them from the cookbook.
    """

    # TODO: confirm import path for Tinker training utilities
    # from tinker.train import load_model, make_optimizer, batches  # TODO

    print(f"  [TODO] fine-tune run '{run_name}' at lr={learning_rate} — not yet implemented")
    print("         See: https://github.com/thinking-machines-lab/tinker-cookbook (vlm_classifier)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading training examples from {TRAIN_DIR} …")
    examples = build_training_examples(TRAIN_DIR)
    print(f"Total: {len(examples)} labeled example(s).\n")

    if not examples:
        print("No training images found — add labeled images to", TRAIN_DIR)
        return

    for i, lr in enumerate(LEARNING_RATES):
        run_name = f"{MODEL_NAME}_lr{lr}_run{i}"
        print(f"=== Run {i + 1}/{len(LEARNING_RATES)}: lr={lr}  run_name={run_name} ===")
        run_finetune(examples, learning_rate=lr, run_name=run_name)
        print()


if __name__ == "__main__":
    main()
