from __future__ import annotations

import csv
import re

import requests


def _current_image_id(html: str) -> str:
    match = re.search(r'/image/([^"]+)"', html)
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
