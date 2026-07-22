#!/usr/bin/env python3
"""Review chronic-miss images: show original label + all checkpoint predictions,
prompt for your current judgment, print/save a clean summary."""
import csv, sys

FILES = ['DSC06207', 'DSC06236', 'DSC06237', 'DSC06309', 'DSC06310']

RUNS = {
    'C_final':    'results/eval_runC_1344.csv',
    'D_final':    'results/eval_runD_1344.csv',
    'D_step20':   'results/eval_runD_step20_1344.csv',
}

def load(path):
    with open(path) as f:
        return {r['basename'].replace('.jpg',''): r for r in csv.DictReader(f)}

data = {name: load(path) for name, path in RUNS.items()}

results = []
for f in FILES:
    print(f"\n{'='*60}")
    print(f"  {f}.jpg")
    print(f"{'='*60}")

    original = None
    for run_name, run_data in data.items():
        r = run_data.get(f)
        if r is None:
            print(f"  [{run_name}] not in this holdout")
            continue
        if original is None:
            original = r['actual']
        preds = [r['pred_run1'], r['pred_run2'], r['pred_run3']]
        print(f"  [{run_name:10s}] majority={r['majority_pred']:20s} "
              f"(votes: {preds})  correct={r['majority_correct']}")

    print(f"\n  ORIGINAL LABEL (yours): {original}")
    print(f"  Image path: dataset/holdout/intentional_blur/{f}.jpg")
    print(f"  (view it, then judge)")

    verdict = input("\n  Your current judgment "
                     "[i=intentional / u=unintentional / s=sharp / k=keep original / ?=unsure]: ").strip().lower()
    verdict_map = {'i': 'intentional_blur', 'u': 'unintentional_blur',
                    's': 'sharp', 'k': original, '?': 'UNSURE'}
    verdict_label = verdict_map.get(verdict, verdict)

    results.append({
        'file': f,
        'original_label': original,
        'your_verdict': verdict_label,
        'changed': verdict_label != original,
    })

# clean summary
print(f"\n\n{'#'*60}")
print("# SUMMARY")
print(f"{'#'*60}")
print(f"{'file':14s} {'original':20s} {'your_verdict':20s} {'changed':8s}")
for r in results:
    print(f"{r['file']:14s} {r['original_label']:20s} {r['your_verdict']:20s} {str(r['changed']):8s}")

with open('results/chronic_miss_review.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['file','original_label','your_verdict','changed'])
    w.writeheader()
    w.writerows(results)
print(f"\nSaved to results/chronic_miss_review.csv")
