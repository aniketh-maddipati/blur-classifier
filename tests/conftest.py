from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def create_synthetic_jpeg(
    path: Path,
    *,
    color: tuple[int, int, int],
    focal_length: float,
) -> Path:
    image = Image.new("RGB", (24, 24), color)
    image.save(path, format="JPEG")
    return path


@pytest.fixture
def synthetic_candidates_dir(tmp_path: Path) -> Path:
    root = tmp_path / "candidates"
    buckets = {
        "sharp_candidates": ((220, 70, 70), 35.0),
        "intentional_blur_candidates": ((70, 220, 70), 85.0),
        "unintentional_blur_candidates": ((70, 70, 220), 50.0),
    }

    for bucket, (color, focal_length) in buckets.items():
        bucket_dir = root / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        for index in range(2):
            create_synthetic_jpeg(
                bucket_dir / f"{bucket}_{index}.jpg",
                color=color,
                focal_length=focal_length,
            )

    return root


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def live_review_server(tmp_path: Path, synthetic_candidates_dir: Path):
    csv_path = tmp_path / "review.csv"
    log_path = tmp_path / "review-server.log"
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    stub_dir = REPO_ROOT / "tests" / "stubs"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(stub_dir), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(stub_dir)]
    )
    env["PYTHONUNBUFFERED"] = "1"

    log_handle = log_path.open("w")
    process = subprocess.Popen(
        [
            sys.executable,
            str(REPO_ROOT / "tests" / "stubs" / "run_cull_review_app.py"),
            "review",
            str(synthetic_candidates_dir),
            "3",
            "--csv",
            str(csv_path),
            "--port",
            str(port),
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.time() + 10
    last_error = None
    while time.time() < deadline:
        if process.poll() is not None:
            break
        try:
            response = requests.get(f"{base_url}/", timeout=0.5)
            if response.status_code == 200:
                yield {
                    "base_url": base_url,
                    "csv_path": csv_path,
                    "process": process,
                }
                break
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(0.2)
    else:
        process.terminate()
        process.wait(timeout=5)
        log_handle.close()
        raise pytest.fail(f"Review server did not come up within 10 seconds: {last_error}")

    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    log_handle.close()
    if process.returncode not in (0, -15):
        output = log_path.read_text()
        raise AssertionError(f"Review server exited unexpectedly with code {process.returncode}:\n{output}")
