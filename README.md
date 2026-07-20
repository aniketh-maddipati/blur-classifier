# blur-classifier

This project fine-tunes **Qwen3-VL-30B-A3B-Instruct** on [Tinker](https://github.com/thinking-machines-lab/tinker-cookbook) to classify photos into three blur categories: `intentional_blur` (deliberate artistic blur such as motion trails or shallow depth-of-field), `unintentional_blur` (camera shake, missed focus, or subject movement), and `sharp` (no significant blur). The goal is to build a labeled training set from real iPhone RAW/JPEG exports, establish a zero-shot baseline, fine-tune with a learning-rate sweep, and compare checkpoint accuracy against the baseline.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then open .env and paste your Tinker API key
```

Add labeled photos to the dataset folders — one subfolder per class:

```
dataset/train/intentional_blur/   dataset/holdout/intentional_blur/
dataset/train/unintentional_blur/ dataset/holdout/unintentional_blur/
dataset/train/sharp/              dataset/holdout/sharp/
```

JPEG, PNG, TIFF, and RAW formats (CR2, NEF, ARW) are all supported.

## Run order

```bash
python classify.py        # fast sanity check — pure assertions, no network call
python baseline_eval.py   # zero-shot accuracy on holdout set → results/baseline_results.csv
python train.py           # fine-tune with LR sweep → results/<run_name>/ checkpoints
python evaluate.py results/<run_name>   # per-checkpoint accuracy → results/finetuned_<run_name>_results.csv
```

> **Note:** Before running `train.py` for real, fill in the `run_finetune()` TODO in `train.py` using the `vlm_classifier` recipe from the [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook). The SFT loop syntax (`forward_backward`, `optim_step`, `save_weights`) must come from the current cookbook, not be guessed. Similarly, verify the Tinker SDK call signatures in `classify.py` before running `baseline_eval.py` or `evaluate.py`.
