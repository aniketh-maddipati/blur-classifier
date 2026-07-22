#!/usr/bin/env python3
import csv, random

FILES = ['DSC06207','DSC06236','DSC06237','DSC06309','DSC06310']
# add a few control images you're confident about, mix in
CONTROLS = ['DSC05659']  # e.g. a clean sharp/unintentional example from the sharp misses

def load_actual(path='results/eval_runD_1344.csv'):
    with open(path) as f:
        return {r['basename'].replace('.jpg',''): r['actual'] for r in csv.DictReader(f)}

actual = load_actual()
pool = FILES + CONTROLS
random.shuffle(pool)

results = []
for f in pool:
    print(f"\nview: dataset/holdout/intentional_blur/{f}.jpg  (or search dataset/holdout/*/{f}.jpg)")
    v = input("your call [i/u/s]: ").strip().lower()
    label = {'i':'intentional_blur','u':'unintentional_blur','s':'sharp'}.get(v, v)
    results.append((f, label))

print("\n--- blind verdicts (unshuffled order shown for your own tracking) ---")
for f, label in results:
    tag = " <-- was in chronic-miss set" if f in FILES else " (control)"
    print(f"{f:12s} {label:20s}{tag}")
