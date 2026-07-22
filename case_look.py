#!/usr/bin/env python3
import os, subprocess, sys

CASES = ["DSC06207", "DSC05659"]
IMAGE_DIR = "dataset/holdout"

def find(base):
    for root, _, files in os.walk(IMAGE_DIR):
        for f in files:
            if f.startswith(base):
                return os.path.join(root, f)
    sys.exit(f"not found: {base}")

for c in CASES:
    subprocess.run(["open", "-a", "Preview", find(c)])

QUESTIONS = [
 ("DSC06207_verdict",
  "DSC06207 (blind reviews disagree: earlier=unintentional, today=intentional).\n"
  "Final call + one-line reason (or 'ambiguous-exclude'):"),
 ("DSC05659_verdict",
  "DSC05659 (3-way unstable: orig=sharp, model=unintentional, earlier blind=intentional, today=sharp).\n"
  "Final call + one-line reason (or 'ambiguous-exclude'):"),
 ("denominator",
  "Keep both in the eval (n=42) or exclude as ambiguous (n=40/41)?\n"
  "[keep / exclude-both / exclude-05659-only]:"),
 ("reference_labels",
  "Headline accuracy framing: [range] (report 78.6-83.3% as-labeled-to-corrected, my rec)\n"
  "or [wait-friend] (majority-of-three once his CSV arrives)?:"),
 ("step20_crowned",
  "AFTER seeing the new precision numbers from recompute: does step20 hold up,\n"
  "or is 21/21 unintentional recall just liberal over-calling? [holds / flip-to-D / rerun-needed]:"),
 ("venue",
  "Where does this get posted/submitted (determines length + how big the product bookends are)?:"),
]

with open("results/decisions_questionnaire.txt", "w") as f:
    print("Both case images should now be open in Preview.\n")
    for key, q in QUESTIONS:
        print("\n" + q)
        ans = input("> ").strip()
        f.write(f"{key}: {ans}\n")
print("\nSaved to results/decisions_questionnaire.txt — feeds DECISIONS.md + writeup placeholders.")
