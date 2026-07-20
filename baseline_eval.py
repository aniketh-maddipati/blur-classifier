"""
Evaluate the unmodified base model on the holdout set.

Writes one row per image to results/baseline_results.csv and prints
overall accuracy plus a per-class breakdown.

Usage:
    python baseline_eval.py
"""

import csv
import os
from pathlib import Path

from config import CLASSES, HOLDOUT_DIR, MODEL_NAME
from classify import classify_image, parse_classification, resize_image

RESULTS_FILE = "results/baseline_results.csv"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".raw", ".cr2", ".nef", ".arw"}


def main() -> None:
    rows = []

    for cls in CLASSES:
        class_dir = Path(HOLDOUT_DIR) / cls
        if not class_dir.exists():
            print(f"  [warn] {class_dir} not found — skipping")
            continue

        images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        print(f"\n{cls}: {len(images)} image(s)")

        for img_path in images:
            image_bytes = resize_image(str(img_path))
            raw = classify_image(image_bytes, MODEL_NAME)
            predicted = parse_classification(raw)
            match = predicted == cls
            status = "OK  " if match else "MISS"
            print(f"  [{status}] {img_path.name}  pred={predicted!r}  raw={raw!r}")
            rows.append({
                "image": img_path.name,
                "true_label": cls,
                "raw_output": raw,
                "predicted": predicted or "",
                "match": match,
            })

    os.makedirs("results", exist_ok=True)
    fieldnames = ["image", "true_label", "raw_output", "predicted", "match"]
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows → {RESULTS_FILE}")

    total = len(rows)
    if total == 0:
        print("No images evaluated — add images to", HOLDOUT_DIR)
        return

    correct = sum(1 for r in rows if r["match"])
    print(f"\nOverall accuracy : {correct}/{total} = {correct / total:.1%}")
    print("\nPer-class breakdown:")
    for cls in CLASSES:
        cls_rows = [r for r in rows if r["true_label"] == cls]
        n = len(cls_rows)
        if n == 0:
            print(f"  {cls:<22} —  (no images)")
        else:
            c = sum(1 for r in cls_rows if r["match"])
            print(f"  {cls:<22} {c}/{n} = {c / n:.1%}")


if __name__ == "__main__":
    main()
