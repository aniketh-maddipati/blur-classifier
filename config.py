"""
Central configuration — import from here, never hard-code values in other modules.
"""

MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"

CLASSES = ["intentional_blur", "unintentional_blur", "sharp"]

CLASSIFICATION_PROMPT = (
    "Look at this photo and classify its blur into exactly one of these categories:\n\n"
    "  intentional_blur  — blur added deliberately for artistic effect "
    "(e.g. motion blur conveying speed, shallow depth-of-field isolating a subject)\n"
    "  unintentional_blur — blur caused by camera shake, missed focus, "
    "subject movement, or any other unintended factor\n"
    "  sharp             — the image is in focus with no significant blur\n\n"
    "Reply with just the category name and nothing else. "
    "Your entire response must be one of: intentional_blur, unintentional_blur, sharp"
)

TRAIN_DIR = "dataset/train"
HOLDOUT_DIR = "dataset/holdout"

MAX_IMAGE_DIMENSION = 1600
