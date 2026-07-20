"""
Shared classification helpers.

Imported by baseline_eval.py and evaluate.py — do not duplicate this logic.
"""

import io

from PIL import Image

from config import CLASSES, CLASSIFICATION_PROMPT, MAX_IMAGE_DIMENSION


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def resize_dimensions(w: int, h: int, max_dim: int = MAX_IMAGE_DIMENSION) -> tuple[int, int]:
    """
    Return (new_w, new_h) after scaling so the longest edge equals max_dim.
    Images already within the limit are returned unchanged.
    Pure function — safe to call in tests without touching the filesystem.
    """
    if max(w, h) <= max_dim:
        return w, h
    scale = max_dim / max(w, h)
    return int(w * scale), int(h * scale)


def resize_image(image_path: str) -> bytes:
    """
    Load an image file, resize so its longest edge is MAX_IMAGE_DIMENSION,
    and return the result as raw JPEG bytes.
    """
    img = Image.open(image_path)
    w, h = img.size
    new_w, new_h = resize_dimensions(w, h)
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Classification output parsing
# ---------------------------------------------------------------------------

def parse_classification(raw: str) -> str | None:
    """
    Extract a CLASSES label from the model's raw text response.
    Strips whitespace and does a case-insensitive substring search.
    Returns the matched class string, or None if no valid class is found.
    """
    text = raw.strip().lower()
    # Sort longest-first so "unintentional_blur" is checked before the
    # shorter "intentional_blur" which is a substring of it.
    for cls in sorted(CLASSES, key=len, reverse=True):
        if cls in text:
            return cls
    return None


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

def classify_image(image_bytes: bytes, model_path: str) -> str:
    """
    Send image_bytes and CLASSIFICATION_PROMPT to the model at model_path
    via the Tinker SDK and return the raw text response.

    TODO: Verify the entire block below against the current Tinker SDK docs
          before running for real.  The sketch follows patterns visible in
          tinker-cookbook examples as of mid-2025 but field names and call
          signatures may have changed.
          Reference: https://github.com/thinking-machines-lab/tinker-cookbook
    """

    # TODO: confirm correct import path for the Tinker SDK
    from tinker import TinkerClient, ModelInput, ImageChunk, EncodedTextChunk  # type: ignore

    # TODO: confirm whether TinkerClient() reads TINKER_API_KEY from the
    #       environment automatically or requires an explicit api_key= kwarg.
    client = TinkerClient()  # TODO: verify init signature

    model_input = ModelInput(  # TODO: verify ModelInput constructor
        chunks=[
            ImageChunk(data=image_bytes, format="jpeg"),  # TODO: verify field names
            EncodedTextChunk(text=CLASSIFICATION_PROMPT),  # TODO: verify field names
        ]
    )

    response = client.sample(model_path, model_input)  # TODO: verify sample() signature
    return response.text  # TODO: verify response attribute name


# ---------------------------------------------------------------------------
# Sanity-check assertions — run with: python classify.py
# No network calls, no test framework required.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- resize_dimensions ---

    # Landscape: longest edge (3200) should scale to 1600
    assert resize_dimensions(3200, 1600) == (1600, 800), "landscape resize failed"

    # Portrait: longest edge (3200) should scale to 1600
    assert resize_dimensions(800, 3200) == (400, 1600), "portrait resize failed"

    # Square at exactly the limit — no resize needed
    assert resize_dimensions(1600, 1600) == (1600, 1600), "exact-limit square should not resize"

    # Already smaller than the limit — no resize needed
    assert resize_dimensions(400, 300) == (400, 300), "small image should not resize"

    # Custom max_dim override
    assert resize_dimensions(1000, 500, max_dim=500) == (500, 250), "custom max_dim failed"

    # --- parse_classification ---

    # Exact match
    assert parse_classification("intentional_blur") == "intentional_blur", "exact intentional_blur"
    assert parse_classification("sharp") == "sharp", "exact sharp"
    assert parse_classification("unintentional_blur") == "unintentional_blur", "exact unintentional_blur"

    # Embedded in a sentence
    assert parse_classification("This image shows unintentional_blur.") == "unintentional_blur", \
        "embedded unintentional_blur"

    # Case-insensitive (class names are lowercase so this also covers mixed output)
    assert parse_classification("  SHARP  ") == "sharp", "uppercase sharp"

    # No valid class → None
    assert parse_classification("I cannot determine the blur type.") is None, \
        "unrecognised output should be None"
    assert parse_classification("") is None, "empty string should be None"

    print("All sanity checks passed.")
