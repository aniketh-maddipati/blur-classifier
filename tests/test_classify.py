import io
import os

import pytest
from PIL import Image

from classify import classify_image, parse_classification, resize_dimensions
from config import MODEL_NAME


@pytest.mark.parametrize(
    ("width", "height", "max_dim", "expected"),
    [
        (400, 300, 1600, (400, 300)),
        (1600, 1600, 1600, (1600, 1600)),
        (3200, 1600, 1600, (1600, 800)),
        (1200, 2400, 1600, (800, 1600)),
        (3000, 2000, 1600, (1600, 1066)),
    ],
)
def test_resize_dimensions(width, height, max_dim, expected):
    assert resize_dimensions(width, height, max_dim) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("sharp", "sharp"),
        ("  sharp!!! ", "sharp"),
        ("I would classify this as sharp.", "sharp"),
        ("UNINTENTIONAL_BLUR", "unintentional_blur"),
        ("intentional_blur,", "intentional_blur"),
        ("totally unrelated output", None),
        ("", None),
    ],
)
def test_parse_classification(raw, expected):
    assert parse_classification(raw) == expected


@pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_API_TESTS"),
    reason="Live API test, costs money, opt-in only",
)
def test_classify_image_live_api_round_trip():
    # This intentionally hits the real API and must stay opt-in only.
    image = Image.new("RGB", (16, 16), (120, 130, 140))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")

    raw = classify_image(buffer.getvalue(), MODEL_NAME)

    assert raw.strip()
    assert parse_classification(raw) in {"intentional_blur", "unintentional_blur", "sharp"}
