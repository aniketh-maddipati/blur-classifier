from __future__ import annotations

import csv
import re
from pathlib import Path

import requests

import finalize_dataset


def _current_image_id(html: str) -> str:
    match = re.search(r'/image/([^"]+)"', html)
    assert match is not None
    return match.group(1)


def _current_filename(html: str) -> str:
    match = re.search(r"<h1>\s*([^<]+?)\s*</h1>", html)
    assert match is not None
    return match.group(1)


def test_review_page_loads(live_review_server):
    response = requests.get(f"{live_review_server['base_url']}/", timeout=2)

    assert response.status_code == 200
    assert "Cull Review" in response.text
    assert "reviewed" in response.text


def test_image_route_returns_jpeg_bytes(live_review_server):
    page = requests.get(f"{live_review_server['base_url']}/", timeout=2)
    image_id = _current_image_id(page.text)

    image_response = requests.get(f"{live_review_server['base_url']}/image/{image_id}", timeout=2)

    assert image_response.status_code == 200
    assert image_response.headers["Content-Type"].startswith("image/jpeg")
    assert image_response.content


def test_confirm_action_writes_csv_with_unavailable_zero_shot_guess(live_review_server):
    base_url = live_review_server["base_url"]
    csv_path = live_review_server["csv_path"]

    scene_response = requests.post(
        f"{base_url}/action",
        data={"action": "outdoor"},
        timeout=2,
        allow_redirects=True,
    )
    assert scene_response.status_code == 200

    confirm_response = requests.post(
        f"{base_url}/action",
        data={"action": "confirm"},
        timeout=2,
        allow_redirects=True,
    )
    assert confirm_response.status_code == 200

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["human_decision"] == "confirm"
    assert row["indoor_or_outdoor"] == "outdoor"
    assert row["zero_shot_guess"] == "unavailable"
    assert row["target_class"]
    assert row["filename"]


def test_previous_image_id_still_serves_after_queue_advances(live_review_server):
    base_url = live_review_server["base_url"]

    page = requests.get(f"{base_url}/", timeout=2)
    image_id = _current_image_id(page.text)

    confirm_response = requests.post(
        f"{base_url}/action",
        data={"action": "skip"},
        timeout=2,
        allow_redirects=True,
    )
    assert confirm_response.status_code == 200

    stale_image_response = requests.get(f"{base_url}/image/{image_id}", timeout=2)

    assert stale_image_response.status_code == 200
    assert stale_image_response.headers["Content-Type"].startswith("image/jpeg")
    assert stale_image_response.content


def test_reclassify_logs_corrected_class_and_finalize_uses_it(live_review_server, tmp_path, monkeypatch):
    base_url = live_review_server["base_url"]
    csv_path = live_review_server["csv_path"]

    requests.post(
        f"{base_url}/action",
        data={"action": "skip"},
        timeout=2,
        allow_redirects=True,
    ).raise_for_status()
    requests.post(
        f"{base_url}/action",
        data={"action": "skip"},
        timeout=2,
        allow_redirects=True,
    ).raise_for_status()

    page = requests.get(f"{base_url}/", timeout=2)
    page.raise_for_status()
    assert _current_filename(page.text) == "unintentional_blur_candidates_0.jpg"
    assert "Reclassify" in page.text
    assert "intentional_blur" in page.text

    response = requests.post(
        f"{base_url}/action",
        data={"action": "reclassify", "reclassify_target": "intentional_blur"},
        timeout=2,
        allow_redirects=True,
    )
    assert response.status_code == 200

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    row = rows[-1]
    assert row["filename"] == "unintentional_blur_candidates_0.jpg"
    assert row["human_decision"] == "reclassify"
    assert row["target_class"] == "intentional_blur"
    assert row["original_triage_class"] == "unintentional_blur"

    train_dir = tmp_path / "train"
    holdout_dir = tmp_path / "holdout"
    monkeypatch.setattr(finalize_dataset, "TRAIN_DIR", str(train_dir))
    monkeypatch.setattr(finalize_dataset, "HOLDOUT_DIR", str(holdout_dir))

    finalized_rows = finalize_dataset.read_rows(Path(csv_path))
    grouped = finalize_dataset.confirmed_rows_by_class(finalized_rows)
    train_rows, holdout_rows = finalize_dataset.split_rows(grouped["intentional_blur"])
    finalize_dataset.copy_rows(train_rows, train_dir)
    finalize_dataset.copy_rows(holdout_rows, holdout_dir)

    assert not (holdout_dir / "unintentional_blur" / row["filename"]).exists()
    assert (holdout_dir / "intentional_blur" / row["filename"]).exists()
