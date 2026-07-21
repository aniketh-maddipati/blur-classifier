"""
Shared classification helpers.

Imported by baseline_eval.py and evaluate.py — do not duplicate this logic.
"""

import io
import os
import asyncio
import importlib.metadata
import queue
import threading
from functools import lru_cache

import httpx
import tinker
from PIL import Image
from tinker_cookbook import model_info, renderers
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.renderers import ImagePart, Message, TextPart, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer

from config import CLASSES, CLASSIFICATION_PROMPT, MAX_IMAGE_DIMENSION


SERVICE_CLIENT_TIMEOUT_SECONDS = 30
SAMPLING_CLIENT_TIMEOUT_SECONDS = 60
SAMPLE_TIMEOUT_SECONDS = 90
TINKER_BASE_URL = os.getenv("TINKER_BASE_URL", "https://tinker.thinkingmachines.dev/services/tinker-prod")


def _fetch_client_config_http(api_key: str) -> dict[str, object]:
    sdk_version = importlib.metadata.version("tinker")
    url = TINKER_BASE_URL.rstrip("/") + "/api/v1/client/config"
    with httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0), follow_redirects=True) as client:
        response = client.post(
            url,
            headers={"X-API-Key": api_key},
            json={"sdk_version": sdk_version},
        )
    response.raise_for_status()
    return response.json()


def _fetch_jwt_http(api_key: str) -> str:
    url = TINKER_BASE_URL.rstrip("/") + "/api/v1/auth/token"
    with httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0), follow_redirects=True) as client:
        response = client.post(
            url,
            headers={"X-API-Key": api_key},
            json={},
        )
    response.raise_for_status()
    payload = response.json()
    return payload["jwt"]


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


@lru_cache(maxsize=None)
def _get_renderer(model_path: str):
    if model_path == "Qwen/Qwen3.6-35B-A3B":
        renderer_name = "qwen3_5_disable_thinking"
    else:
        renderer_name = model_info.get_recommended_renderer_name(model_path)

    renderer = renderers.get_renderer(
        name=renderer_name,
        tokenizer=get_tokenizer(model_path),
        image_processor=get_image_processor(model_path),
    )
    return renderer


def _create_service_client(api_key: str) -> tinker.ServiceClient:
    result_queue: queue.Queue[tinker.ServiceClient | BaseException] = queue.Queue(maxsize=1)

    def _target() -> None:
        try:
            config = _fetch_client_config_http(api_key)
            # The raw HTTP path works in this environment while the SDK path can hang
            # when it follows the server-advertised pyqwest transport.
            config["use_pyqwest_transport"] = False
            jwt_seed = None
            if config.get("pjwt_auth_enabled"):
                jwt_seed = _fetch_jwt_http(api_key)
            result_queue.put(
                tinker.ServiceClient(
                    api_key=api_key,
                    project_id=os.getenv("TINKER_PROJECT_ID"),
                    _client_config=config,
                    _jwt_auth_seed=jwt_seed,
                )
            )
        except BaseException as exc:  # pragma: no cover - exercised by live path
            result_queue.put(exc)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    try:
        result = result_queue.get(timeout=SERVICE_CLIENT_TIMEOUT_SECONDS)
    except queue.Empty as exc:
        raise TimeoutError(
            "Timed out while initializing Tinker ServiceClient. "
            "The SDK appears to be stuck during client-config fetch, JWT exchange, or session creation."
        ) from exc

    if isinstance(result, BaseException):
        raise result
    return result


async def _create_sampling_client_async(api_key: str, model_path: str) -> tinker.SamplingClient:
    service_client = await asyncio.to_thread(_create_service_client, api_key)
    sampling_client = await asyncio.wait_for(
        service_client.create_sampling_client_async(base_model=model_path),
        timeout=SAMPLING_CLIENT_TIMEOUT_SECONDS,
    )
    return sampling_client


async def _sample_text_async(
    sampling_client: tinker.SamplingClient,
    prompt: tinker.ModelInput,
    stop_sequences: list[str] | list[int],
) -> tinker.SampleResponse:
    response = await asyncio.wait_for(
        sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=500,
                temperature=0.0,
                stop=stop_sequences,
            ),
        ),
        timeout=SAMPLE_TIMEOUT_SECONDS,
    )
    return response


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

def classify_image(image_bytes: bytes, model_path: str) -> str:
    """
    Send image_bytes and CLASSIFICATION_PROMPT to the model at model_path
    via the native Tinker sampling client and return the raw text response.
    """
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("TINKER_API_KEY is not set")

    with Image.open(io.BytesIO(image_bytes)) as image:
        pil_image = image.convert("RGB")

    renderer = _get_renderer(model_path)

    messages: list[Message] = [
        Message(
            role="user",
            content=[
                ImagePart(type="image", image=pil_image),
                TextPart(type="text", text=CLASSIFICATION_PROMPT),
            ],
        )
    ]
    prompt = renderer.build_generation_prompt(messages)

    try:
        sampling_client = asyncio.run(_create_sampling_client_async(api_key, model_path))
        response = asyncio.run(
            _sample_text_async(sampling_client, prompt, renderer.get_stop_sequences())
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            "Timed out while initializing Tinker or waiting for a model response. "
            "If this happens on the first run, Hugging Face asset downloads may still be pending."
        ) from exc

    parsed_message, _termination = renderer.parse_response(response.sequences[0].tokens)
    text = get_text_content(parsed_message)
    return text


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
