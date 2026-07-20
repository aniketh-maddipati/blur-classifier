"""
Flask review app for fast candidate culling plus an explicit finalize step.

Review mode:
    python cull_review_app.py review dataset_candidates/sharp_candidates 40

Finalize mode:
    python cull_review_app.py finalize results/cull_review.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import os
import shutil
import subprocess
import tempfile
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from flask import Flask, Response, redirect, render_template_string, request, send_file, url_for
except ModuleNotFoundError:  # pragma: no cover - keeps finalize mode usable without Flask installed
    Flask = None  # type: ignore[assignment]
    Response = Any  # type: ignore[misc,assignment]
    redirect = render_template_string = request = send_file = url_for = None  # type: ignore[assignment]

from PIL import Image, UnidentifiedImageError

from classify import classify_image, parse_classification, resize_dimensions
from config import CLASSES, HOLDOUT_DIR, MAX_IMAGE_DIMENSION, MODEL_NAME, TRAIN_DIR
from exif_analyzer import ShotMetadata, extract_metadata

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".arw", ".nef", ".cr2", ".raw"}
CSV_FIELDS = [
    "filename",
    "source_path",
    "triage_bucket",
    "original_triage_class",
    "target_class",
    "aperture",
    "shutter_speed",
    "iso",
    "focal_length",
    "focal_length_group",
    "human_decision",
    "indoor_or_outdoor",
    "review_duration_seconds",
    "zero_shot_guess",
    "agrees_with_human",
    "first_display_timestamp",
    "timestamp",
    "session_id",
]
REVIEW_DECISIONS = {"confirm", "reject", "reclassify", "skip"}
FOCAL_GROUPS = ("35", "85", "zoom")
TRIAGE_TO_CLASS = {
    "intentional_blur_candidates": "intentional_blur",
    "unintentional_blur_candidates": "unintentional_blur",
    "sharp_candidates": "sharp",
    "intentional_blur": "intentional_blur",
    "unintentional_blur": "unintentional_blur",
    "sharp": "sharp",
}
SUMMARY_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cull Review</title>
  <style>
    :root {
      --bg: #f6f1e8;
      --panel: rgba(255, 252, 247, 0.92);
      --ink: #1f1a17;
      --muted: #6c6259;
      --accent: #99582a;
      --accent-2: #588157;
      --warn: #bc4749;
      --line: rgba(31, 26, 23, 0.12);
      --shadow: 0 18px 45px rgba(80, 52, 28, 0.14);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Iowan Old Style", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(153, 88, 42, 0.12), transparent 34%),
        linear-gradient(135deg, #f6f1e8, #efe4d1 55%, #e4d1b9);
    }
    .shell {
      width: min(1600px, calc(100vw - 28px));
      margin: 18px auto;
      display: grid;
      grid-template-columns: minmax(360px, 1fr) 360px;
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .image-wrap {
      padding: 16px;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 72vh;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.65), rgba(239,228,209,0.5)),
        repeating-linear-gradient(
          45deg,
          rgba(31, 26, 23, 0.02),
          rgba(31, 26, 23, 0.02) 10px,
          rgba(255, 255, 255, 0.02) 10px,
          rgba(255, 255, 255, 0.02) 20px
        );
    }
    img {
      max-width: 100%;
      max-height: calc(100vh - 180px);
      border-radius: 16px;
      box-shadow: 0 20px 40px rgba(31, 26, 23, 0.18);
      background: #ddd4c6;
    }
    .controls {
      padding: 14px 16px 18px;
      border-top: 1px solid var(--line);
    }
    .button-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 10px;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font: inherit;
      cursor: pointer;
      color: white;
      background: var(--accent);
      box-shadow: 0 10px 22px rgba(153, 88, 42, 0.22);
    }
    button.alt { background: #7f5539; }
    button.soft { background: var(--accent-2); }
    button.warn { background: var(--warn); }
    button.ghost { background: #8c8c8c; }
    .meta, .summary {
      padding: 18px 20px;
    }
    h1, h2, h3 { margin: 0 0 10px; }
    h1 { font-size: 1.7rem; }
    h2 { font-size: 1.15rem; margin-top: 18px; }
    h3 { font-size: 1rem; margin-top: 14px; }
    .muted { color: var(--muted); }
    .pill {
      display: inline-block;
      margin: 0 8px 8px 0;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(88, 129, 87, 0.12);
      color: #36543b;
      font-size: 0.95rem;
    }
    .pill.pending {
      background: rgba(153, 88, 42, 0.15);
      color: #6c3d18;
    }
    ul {
      margin: 8px 0 0 18px;
      padding: 0;
    }
    li { margin: 6px 0; }
    .flash {
      margin: 0 20px 12px;
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(188, 71, 73, 0.10);
      color: #7c2628;
    }
    .done {
      padding: 44px 22px;
      text-align: center;
    }
    .small { font-size: 0.95rem; }
    @media (max-width: 1100px) {
      .shell { grid-template-columns: 1fr; }
      .image-wrap { min-height: 56vh; }
    }
  </style>
</head>
<body>
  {% if current %}
  <div class="shell">
    <section class="panel">
      <div class="meta">
        <h1>{{ current.filename }}</h1>
        <div class="pill">{{ current.triage_bucket }}</div>
        <div class="pill">{{ current.target_class }}</div>
        <div class="pill">{{ current.focal_length_group }}</div>
        {% if current.indoor_or_outdoor %}
        <div class="pill">{{ current.indoor_or_outdoor }} selected</div>
        {% else %}
        <div class="pill pending">Indoor/outdoor not set</div>
        {% endif %}
        <p class="muted small">
          Aperture {{ current.aperture_display }} · Shutter {{ current.shutter_display }} ·
          ISO {{ current.iso_display }} · Focal {{ current.focal_display }}
        </p>
      </div>
      {% if flash_message %}
      <div class="flash">{{ flash_message }}</div>
      {% endif %}
      <div class="image-wrap">
        <img src="{{ url_for('image_file', image_id=current.image_id) }}" alt="{{ current.filename }}">
      </div>
      <div class="controls">
        <form method="post" action="{{ url_for('act') }}">
          <div class="button-row">
            <button class="soft" type="submit" name="action" value="confirm">Confirm (C)</button>
            <button class="warn" type="submit" name="action" value="reject">Reject (R)</button>
            <button class="alt" type="submit" name="action" value="indoor">Indoor (I)</button>
            <button class="alt" type="submit" name="action" value="outdoor">Outdoor (O)</button>
            <button class="ghost" type="submit" name="action" value="skip">Skip (S)</button>
            <button class="ghost" type="submit" name="action" value="undo">Undo last (U)</button>
          </div>
        </form>
        <details>
          <summary>Reclassify</summary>
          <div class="button-row">
            {% for reclassify_target in current.reclassify_options %}
            <form method="post" action="{{ url_for('act') }}">
              <input type="hidden" name="action" value="reclassify">
              <input type="hidden" name="reclassify_target" value="{{ reclassify_target }}">
              <button class="alt" type="submit">As {{ reclassify_target }}</button>
            </form>
            {% endfor %}
          </div>
        </details>
        <p class="muted small">
          Indoor/outdoor is sticky for the current image until you confirm, reject, or skip it.
          Confirm, reject, and reclassify all log immediately, then record a silent zero-shot baseline guess.
        </p>
      </div>
    </section>
    <aside class="panel">
      <div class="summary">
        <h2>{{ summary.reviewed_so_far }} / {{ summary.target_count }} reviewed</h2>
        <p class="muted small">
          {{ summary.remaining_count }} remaining in this sampled set · session {{ session_id }}
        </p>

        <h2>Sampling</h2>
        <ul>
          {% for line in sampling_report %}
          <li>{{ line }}</li>
          {% endfor %}
        </ul>

        <h2>Confirm Rate By Triage Bucket</h2>
        <ul>
          {% for line in summary.confirm_rate_lines %}
          <li>{{ line }}</li>
          {% endfor %}
        </ul>

        <h2>Indoor / Outdoor By Class</h2>
        <ul>
          {% for line in summary.indoor_outdoor_lines %}
          <li>{{ line }}</li>
          {% endfor %}
        </ul>

        <h2>Focal Group Split By Class</h2>
        <ul>
          {% for line in summary.focal_group_lines %}
          <li>{{ line }}</li>
          {% endfor %}
        </ul>

        <h2>Human / Zero-Shot Agreement By Class</h2>
        <ul>
          {% for line in summary.agreement_lines %}
          <li>{{ line }}</li>
          {% endfor %}
        </ul>

        <h2>Timing</h2>
        <ul>
          <li>Average review duration: {{ summary.average_review_duration_display }}</li>
          {% for line in summary.slowest_lines %}
          <li>{{ line }}</li>
          {% endfor %}
        </ul>
      </div>
    </aside>
  </div>
  <script>
    window.addEventListener("keydown", (event) => {
      if (event.target && ["INPUT", "TEXTAREA"].includes(event.target.tagName)) return;
      const map = { c: "confirm", r: "reject", i: "indoor", o: "outdoor", s: "skip", u: "undo" };
      const action = map[event.key.toLowerCase()];
      if (!action) return;
      const form = document.createElement("form");
      form.method = "post";
      form.action = "{{ url_for('act') }}";
      const input = document.createElement("input");
      input.type = "hidden";
      input.name = "action";
      input.value = action;
      form.appendChild(input);
      document.body.appendChild(form);
      form.submit();
    });
  </script>
  {% else %}
  <div class="shell">
    <section class="panel done">
      <h1>Review complete</h1>
      <p class="muted">Everything in this sampled set has already been logged.</p>
      <p class="small">Run <code>python cull_review_app.py finalize {{ csv_path }}</code> when you are ready to copy confirmed files into train/holdout.</p>
    </section>
    <aside class="panel">
      <div class="summary">
        <h2>{{ summary.reviewed_so_far }} / {{ summary.target_count }} reviewed</h2>
        <ul>
          {% for line in summary.confirm_rate_lines %}
          <li>{{ line }}</li>
          {% endfor %}
        </ul>
        <h2>Timing</h2>
        <ul>
          <li>Average review duration: {{ summary.average_review_duration_display }}</li>
          {% for line in summary.slowest_lines %}
          <li>{{ line }}</li>
          {% endfor %}
        </ul>
      </div>
    </aside>
  </div>
  {% endif %}
</body>
</html>
"""


@dataclass(frozen=True)
class Candidate:
    image_id: str
    path: Path
    filename: str
    triage_bucket: str
    target_class: str
    metadata: ShotMetadata
    focal_length_group: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def reclassify_options_for(target_class: str) -> list[str]:
    return [cls for cls in CLASSES if cls != target_class]


def normalize_number(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{value}"


def display_value(value: float | int | None, prefix: str = "", suffix: str = "") -> str:
    if value is None or value == "":
        return "unknown"
    return f"{prefix}{value}{suffix}"


def classify_focal_group(focal_length: float | None) -> str | None:
    if focal_length is None:
        return None
    if 33 <= focal_length <= 40:
        return "35"
    if 80 <= focal_length <= 90:
        return "85"
    if 28 <= focal_length <= 75:
        return "zoom"
    return None


def infer_triage_bucket(candidates_root: Path, image_path: Path) -> str:
    relative_parts = image_path.relative_to(candidates_root).parts
    for part in relative_parts[:-1]:
        if part in TRIAGE_TO_CLASS:
            return part
    if image_path.parent.name in TRIAGE_TO_CLASS:
        return image_path.parent.name
    return candidates_root.name


def infer_target_class(triage_bucket: str) -> str:
    if triage_bucket not in TRIAGE_TO_CLASS:
        raise ValueError(
            f"Cannot map triage bucket {triage_bucket!r} to a dataset class. "
            "Expected a *_candidates folder or one of the class names."
        )
    return TRIAGE_TO_CLASS[triage_bucket]


def iter_candidate_paths(candidates_root: Path) -> list[Path]:
    return sorted(
        path for path in candidates_root.rglob("*")
        if (path.is_file() or path.is_symlink()) and path.suffix.lower() in IMAGE_EXTS
    )


def ensure_csv(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    ensure_csv(csv_path)
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def rewrite_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def format_shutter_speed(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds >= 1:
        return f"{seconds:g}s"
    return f"1/{round(1 / seconds)}s"


def prepare_candidates(candidates_root: Path, reviewed_filenames: set[str]) -> tuple[list[Candidate], list[str]]:
    candidates: list[Candidate] = []
    sampling_notes: list[str] = []
    skipped_reviewed = 0
    skipped_unknown_group = 0
    skipped_unreadable = 0
    skipped_unsupported_bucket = 0

    for path in iter_candidate_paths(candidates_root):
        if path.name in reviewed_filenames:
            skipped_reviewed += 1
            continue

        triage_bucket = infer_triage_bucket(candidates_root, path)
        try:
            target_class = infer_target_class(triage_bucket)
        except ValueError:
            skipped_unsupported_bucket += 1
            continue
        try:
            metadata = extract_metadata(str(path.resolve(strict=False)))
        except Exception:
            skipped_unreadable += 1
            continue
        focal_group = classify_focal_group(metadata.focal_length)
        if focal_group is None:
            skipped_unknown_group += 1
            continue

        candidate = Candidate(
            image_id=hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16],
            path=path,
            filename=path.name,
            triage_bucket=triage_bucket,
            target_class=target_class,
            metadata=metadata,
            focal_length_group=focal_group,
        )
        candidates.append(candidate)

    if skipped_reviewed:
        sampling_notes.append(f"Skipped {skipped_reviewed} file(s) already present in the CSV.")
    if skipped_unknown_group:
        sampling_notes.append(
            f"Skipped {skipped_unknown_group} file(s) outside the 35/85/28-75 focal-length groups."
        )
    if skipped_unreadable:
        sampling_notes.append(f"Skipped {skipped_unreadable} unreadable or missing file(s).")
    if skipped_unsupported_bucket:
        sampling_notes.append(
            f"Skipped {skipped_unsupported_bucket} file(s) from unsupported triage buckets such as borderline/unknown."
        )

    return candidates, sampling_notes


def stratified_sample(candidates: list[Candidate], target_count: int) -> tuple[list[Candidate], list[str]]:
    grouped: dict[str, list[Candidate]] = {group: [] for group in FOCAL_GROUPS}
    for candidate in candidates:
        grouped[candidate.focal_length_group].append(candidate)

    for bucket in grouped.values():
        bucket.sort(key=lambda item: item.filename)

    desired = {group: target_count // len(FOCAL_GROUPS) for group in FOCAL_GROUPS}
    for group in FOCAL_GROUPS[:target_count % len(FOCAL_GROUPS)]:
        desired[group] += 1

    selected: dict[str, list[Candidate]] = {group: [] for group in FOCAL_GROUPS}
    shortfalls: dict[str, int] = {}

    for group in FOCAL_GROUPS:
        take = min(desired[group], len(grouped[group]))
        selected[group] = grouped[group][:take]
        if take < desired[group]:
            shortfalls[group] = desired[group] - take

    deficit = target_count - sum(len(items) for items in selected.values())
    extras_taken: dict[str, int] = defaultdict(int)
    if deficit > 0:
        for group in FOCAL_GROUPS:
            if deficit <= 0:
                break
            spare = grouped[group][len(selected[group]):]
            if not spare:
                continue
            take = min(len(spare), deficit)
            selected[group].extend(spare[:take])
            extras_taken[group] += take
            deficit -= take

    flattened: list[Candidate] = []
    for index in range(max(len(selected[group]) for group in FOCAL_GROUPS) if candidates else 0):
        for group in FOCAL_GROUPS:
            if index < len(selected[group]):
                flattened.append(selected[group][index])

    report = [
        f"{group}: available {len(grouped[group])}, sampled {len(selected[group])}, initial target {desired[group]}"
        for group in FOCAL_GROUPS
    ]
    for group, shortfall in shortfalls.items():
        report.append(f"{group}: short by {shortfall}; backfilled from other focal-length groups.")
    for group, extra in extras_taken.items():
        report.append(f"{group}: supplied {extra} extra file(s) to cover other groups.")
    if deficit > 0:
        report.append(f"Only {len(flattened)} eligible file(s) available for a target of {target_count}.")
    return flattened[:target_count], report


def load_review_rows_for_scope(csv_rows: list[dict[str, str]], triage_buckets: set[str]) -> list[dict[str, str]]:
    return [row for row in csv_rows if row.get("triage_bucket") in triage_buckets]


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def build_summary(csv_rows: list[dict[str, str]], target_count: int, triage_buckets: set[str]) -> dict[str, Any]:
    scoped_rows = load_review_rows_for_scope(csv_rows, triage_buckets)
    reviewed_rows = [row for row in scoped_rows if row.get("human_decision") in REVIEW_DECISIONS]
    decision_rows = [row for row in scoped_rows if row.get("human_decision") in {"confirm", "reject", "reclassify"}]
    reviewed_so_far = len(reviewed_rows)

    confirm_by_bucket: dict[str, dict[str, int]] = defaultdict(lambda: {"confirm": 0, "total": 0})
    indoor_outdoor_by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"indoor": 0, "outdoor": 0, "unset": 0})
    focal_by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"35": 0, "85": 0, "zoom": 0, "other": 0})
    agreement_by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"agree": 0, "total": 0})
    durations: list[tuple[float, dict[str, str]]] = []

    for row in decision_rows:
        bucket = row.get("triage_bucket", "unknown")
        target_class = row.get("target_class") or TRIAGE_TO_CLASS.get(bucket, "unknown")
        decision = row.get("human_decision")
        confirm_by_bucket[bucket]["total"] += 1
        if decision == "confirm":
            confirm_by_bucket[bucket]["confirm"] += 1

        scene = row.get("indoor_or_outdoor") or "unset"
        indoor_outdoor_by_class[target_class][scene] = indoor_outdoor_by_class[target_class].get(scene, 0) + 1

        focal_group = row.get("focal_length_group") or "other"
        focal_by_class[target_class][focal_group if focal_group in FOCAL_GROUPS else "other"] += 1

        agrees = row.get("agrees_with_human")
        if agrees in {"True", "False"}:
            agreement_by_class[target_class]["total"] += 1
            if agrees == "True":
                agreement_by_class[target_class]["agree"] += 1

        duration = parse_float(row.get("review_duration_seconds"))
        if duration is not None:
            durations.append((duration, row))

    average_duration = sum(duration for duration, _ in durations) / len(durations) if durations else 0.0
    slowest = sorted(durations, key=lambda item: item[0], reverse=True)[:5]

    confirm_rate_lines = []
    for bucket in sorted(confirm_by_bucket):
        total = confirm_by_bucket[bucket]["total"]
        confirm = confirm_by_bucket[bucket]["confirm"]
        rate = (confirm / total) if total else 0.0
        confirm_rate_lines.append(f"{bucket}: {confirm}/{total} confirmed ({rate:.1%})")
    if not confirm_rate_lines:
        confirm_rate_lines = ["No confirm/reject decisions logged yet."]

    indoor_outdoor_lines = []
    for target_class in sorted(indoor_outdoor_by_class):
        counts = indoor_outdoor_by_class[target_class]
        total = sum(counts.values()) or 1
        indoor_outdoor_lines.append(
            f"{target_class}: indoor {counts['indoor']} ({counts['indoor']/total:.1%}), "
            f"outdoor {counts['outdoor']} ({counts['outdoor']/total:.1%}), unset {counts['unset']} ({counts['unset']/total:.1%})"
        )
    if not indoor_outdoor_lines:
        indoor_outdoor_lines = ["No indoor/outdoor data yet."]

    focal_group_lines = []
    for target_class in sorted(focal_by_class):
        counts = focal_by_class[target_class]
        total = sum(counts.values()) or 1
        focal_group_lines.append(
            f"{target_class}: 35mm {counts['35']} ({counts['35']/total:.1%}), "
            f"85mm {counts['85']} ({counts['85']/total:.1%}), zoom {counts['zoom']} ({counts['zoom']/total:.1%}), "
            f"other {counts['other']} ({counts['other']/total:.1%})"
        )
    if not focal_group_lines:
        focal_group_lines = ["No focal-length split yet."]

    agreement_lines = []
    for target_class in sorted(agreement_by_class):
        total = agreement_by_class[target_class]["total"]
        agree = agreement_by_class[target_class]["agree"]
        rate = (agree / total) if total else 0.0
        agreement_lines.append(f"{target_class}: {agree}/{total} agreement ({rate:.1%})")
    if not agreement_lines:
        agreement_lines = ["No human/zero-shot agreement data yet."]

    slowest_lines = [
        f"Slowest: {row.get('filename')} at {duration:.2f}s ({row.get('human_decision')})"
        for duration, row in slowest
    ]
    if not slowest_lines:
        slowest_lines = ["No completed decisions yet for slowest list."]

    return {
        "reviewed_so_far": reviewed_so_far,
        "target_count": target_count,
        "remaining_count": max(target_count - reviewed_so_far, 0),
        "confirm_rate_lines": confirm_rate_lines,
        "indoor_outdoor_lines": indoor_outdoor_lines,
        "focal_group_lines": focal_group_lines,
        "agreement_lines": agreement_lines,
        "average_review_duration_seconds": average_duration,
        "average_review_duration_display": f"{average_duration:.2f}s",
        "slowest_lines": slowest_lines,
    }


def build_final_report(csv_rows: list[dict[str, str]]) -> str:
    triage_buckets = {row["triage_bucket"] for row in csv_rows if row.get("triage_bucket")}
    summary = build_summary(csv_rows, len(csv_rows), triage_buckets)
    durations = [
        parse_float(row.get("review_duration_seconds")) or 0.0
        for row in csv_rows
        if row.get("human_decision") in REVIEW_DECISIONS
    ]
    total_time = sum(durations)
    reviewed = len(durations)
    speed = (reviewed / total_time * 60.0) if total_time > 0 else 0.0
    lines = [
        f"Reviewed: {summary['reviewed_so_far']}",
        "Confirm rate by triage bucket:",
        *[f"  - {line}" for line in summary["confirm_rate_lines"]],
        "Indoor/outdoor split by class:",
        *[f"  - {line}" for line in summary["indoor_outdoor_lines"]],
        "Focal-length-group split by class:",
        *[f"  - {line}" for line in summary["focal_group_lines"]],
        "Human/zero-shot agreement by class:",
        *[f"  - {line}" for line in summary["agreement_lines"]],
        f"Average review duration per image: {summary['average_review_duration_display']}",
        f"Total review time: {total_time:.2f}s",
        f"Review speed: {speed:.2f} images/minute",
        "Slowest decisions:",
        *[f"  - {line}" for line in summary["slowest_lines"]],
    ]
    return "\n".join(lines)


def make_preview_cache_path(source_path: Path) -> Path:
    stat = source_path.stat()
    token = hashlib.sha1(f"{source_path.resolve(strict=False)}:{stat.st_mtime_ns}".encode("utf-8")).hexdigest()
    return Path(tempfile.gettempdir()) / f"cull-review-preview-{token}.jpg"


def render_via_sips(source_path: Path) -> Path:
    preview_path = make_preview_cache_path(source_path)
    if preview_path.exists():
        return preview_path
    subprocess.run(
        ["sips", "-s", "format", "jpeg", str(source_path), "--out", str(preview_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return preview_path


def image_to_jpeg_bytes(source_path: Path, max_dimension: int | None = None) -> bytes:
    try:
        image = Image.open(source_path)
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        converted = render_via_sips(source_path)
        image = Image.open(converted)

    width, height = image.size
    if max_dimension is not None:
        new_width, new_height = resize_dimensions(width, height, max_dimension)
        if (new_width, new_height) != (width, height):
            image = image.resize((new_width, new_height), Image.LANCZOS)

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return buffer.read()


def zero_shot_guess_for(candidate: Candidate) -> tuple[str, str]:
    raw = classify_image(image_to_jpeg_bytes(candidate.path, max_dimension=MAX_IMAGE_DIMENSION), MODEL_NAME)
    parsed = parse_classification(raw) or ""
    return parsed, raw


def classify_agreement(target_class: str, decision: str, zero_shot_guess: str) -> str:
    if decision not in {"confirm", "reject", "reclassify"} or not zero_shot_guess:
        return ""
    if decision == "reject":
        agrees = zero_shot_guess != target_class
    else:
        agrees = zero_shot_guess == target_class
    return str(agrees)


def copy_confirmed_images(csv_rows: list[dict[str, str]], train_dir: Path, holdout_dir: Path) -> None:
    confirmed_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in csv_rows:
        if row.get("human_decision") in {"confirm", "reclassify"}:
            confirmed_by_class[row["target_class"]].append(row)

    for target_class, rows in confirmed_by_class.items():
        rows.sort(key=lambda row: row.get("timestamp", ""))
        holdout_rows = rows[-10:]
        train_rows = rows[:-10]

        for split_name, split_rows, split_dir in [
            ("train", train_rows, train_dir / target_class),
            ("holdout", holdout_rows, holdout_dir / target_class),
        ]:
            split_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            for row in split_rows:
                source_path = Path(row["source_path"])
                if not source_path.exists():
                    raise FileNotFoundError(
                        f"Cannot finalize {row['filename']} into {split_name}: missing source file {source_path}"
                    )
                destination = split_dir / row["filename"]
                shutil.copy2(source_path, destination)
                copied += 1
            print(f"{target_class} -> {split_name}: copied {copied} file(s) into {split_dir}")


def build_app(
    candidates_root: Path,
    target_count: int,
    csv_path: Path,
) -> Flask:
    if Flask is None:
        raise ModuleNotFoundError(
            "Flask is not installed. Run `pip install -r requirements.txt` before starting review mode."
        )

    existing_rows = read_csv_rows(csv_path)
    reviewed_filenames = {row["filename"] for row in existing_rows if row.get("filename")}
    triage_buckets_in_scope = {
        infer_triage_bucket(candidates_root, path)
        for path in iter_candidate_paths(candidates_root)
        if infer_triage_bucket(candidates_root, path) in TRIAGE_TO_CLASS
    }
    scope_reviewed = load_review_rows_for_scope(existing_rows, triage_buckets_in_scope)
    remaining_target = max(target_count - len(scope_reviewed), 0)
    prepared_candidates, prepare_notes = prepare_candidates(candidates_root, reviewed_filenames)
    sampled_candidates, sampling_report = stratified_sample(prepared_candidates, remaining_target)
    sampling_report = prepare_notes + sampling_report
    if remaining_target < target_count and scope_reviewed:
        sampling_report.insert(
            0,
            f"{len(scope_reviewed)} file(s) in this candidate scope were already logged; sampling the remaining {remaining_target}.",
        )

    app = Flask(__name__)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    state: dict[str, Any] = {
        "session_id": uuid.uuid4().hex[:12],
        "csv_path": csv_path,
        "target_count": target_count,
        "triage_buckets_in_scope": triage_buckets_in_scope,
        "queue": sampled_candidates,
        "candidate_lookup": {candidate.filename: candidate for candidate in prepared_candidates + sampled_candidates},
        "image_lookup": {
            candidate.image_id: candidate for candidate in prepared_candidates + sampled_candidates
        },
        "display_times": {},
        "current_scene": {},
        "flash_message": "",
        "sampling_report": sampling_report or ["No sampling notes."],
    }

    def current_candidate() -> Candidate | None:
        queue: list[Candidate] = state["queue"]
        return queue[0] if queue else None

    def current_scene(candidate: Candidate | None) -> str:
        if candidate is None:
            return ""
        return state["current_scene"].get(candidate.filename, "")

    def consume_flash() -> str:
        message = state["flash_message"]
        state["flash_message"] = ""
        return message

    @app.get("/")
    def index() -> str:
        candidate = current_candidate()
        if candidate and candidate.filename not in state["display_times"]:
            state["display_times"][candidate.filename] = now_iso()

        csv_rows = read_csv_rows(state["csv_path"])
        summary = build_summary(csv_rows, state["target_count"], state["triage_buckets_in_scope"])

        current_payload = None
        if candidate:
            current_payload = {
                "image_id": candidate.image_id,
                "filename": candidate.filename,
                "triage_bucket": candidate.triage_bucket,
                "target_class": candidate.target_class,
                "focal_length_group": candidate.focal_length_group,
                "aperture_display": display_value(candidate.metadata.aperture, prefix="f/"),
                "shutter_display": format_shutter_speed(candidate.metadata.shutter_speed),
                "iso_display": display_value(candidate.metadata.iso),
                "focal_display": display_value(candidate.metadata.focal_length, suffix="mm"),
                "indoor_or_outdoor": current_scene(candidate),
                "reclassify_options": reclassify_options_for(candidate.target_class),
            }

        return render_template_string(
            SUMMARY_TEMPLATE,
            current=current_payload,
            summary=summary,
            sampling_report=state["sampling_report"],
            flash_message=consume_flash(),
            session_id=state["session_id"],
            csv_path=state["csv_path"],
        )

    @app.get("/image/<image_id>")
    def image_file(image_id: str) -> Response:
        # The browser can still finish fetching the previous <img src> after a
        # confirm/reject/skip POST has already popped that candidate off the
        # queue, so image lookup cannot rely on the live queue alone.
        candidate = state["image_lookup"].get(image_id)
        if candidate is not None:
            if not candidate.path.exists():
                raise FileNotFoundError(f"Image source is missing: {candidate.path}")
            return send_file(io.BytesIO(image_to_jpeg_bytes(candidate.path)), mimetype="image/jpeg")
        raise FileNotFoundError(f"Unknown image id: {image_id}")

    @app.post("/action")
    def act() -> Response:
        action = request.form.get("action", "")
        if action == "undo":
            rows = read_csv_rows(state["csv_path"])
            if not rows:
                state["flash_message"] = "Nothing to undo yet."
                return redirect(url_for("index"))
            undone = rows.pop()
            rewrite_csv(state["csv_path"], rows)
            filename = undone.get("filename", "")
            candidate = state["candidate_lookup"].get(filename)
            if candidate is not None and all(item.filename != filename for item in state["queue"]):
                state["queue"].insert(0, candidate)
                state["display_times"].pop(filename, None)
                state["current_scene"].pop(filename, None)
            state["flash_message"] = f"Undid {filename or 'the last row'}."
            return redirect(url_for("index"))

        candidate = current_candidate()
        if candidate is None:
            state["flash_message"] = "No images left in this sampled set."
            return redirect(url_for("index"))

        if action in {"indoor", "outdoor"}:
            state["current_scene"][candidate.filename] = action
            return redirect(url_for("index"))

        corrected_target_class = candidate.target_class
        if action == "reclassify":
            corrected_target_class = request.form.get("reclassify_target", "").strip()
            if corrected_target_class not in reclassify_options_for(candidate.target_class):
                state["flash_message"] = f"Unknown reclassify target: {corrected_target_class or 'missing'}"
                return redirect(url_for("index"))
        elif action not in REVIEW_DECISIONS:
            state["flash_message"] = f"Unknown action: {action}"
            return redirect(url_for("index"))

        started_at = state["display_times"].get(candidate.filename)
        if started_at is None:
            started_at = now_iso()
            state["display_times"][candidate.filename] = started_at
        clicked_at = now_iso()
        duration = (
            datetime.fromisoformat(clicked_at) - datetime.fromisoformat(started_at)
        ).total_seconds()

        zero_shot_guess = ""
        if action in {"confirm", "reject", "reclassify"}:
            try:
                zero_shot_guess, _ = zero_shot_guess_for(candidate)
            except Exception:
                zero_shot_guess = "unavailable"

        row = {
            "filename": candidate.filename,
            "source_path": str(candidate.path.resolve(strict=False)),
            "triage_bucket": candidate.triage_bucket,
            "original_triage_class": candidate.target_class,
            "target_class": corrected_target_class,
            "aperture": normalize_number(candidate.metadata.aperture),
            "shutter_speed": normalize_number(candidate.metadata.shutter_speed),
            "iso": normalize_number(candidate.metadata.iso),
            "focal_length": normalize_number(candidate.metadata.focal_length),
            "focal_length_group": candidate.focal_length_group,
            "human_decision": action,
            "indoor_or_outdoor": current_scene(candidate),
            "review_duration_seconds": f"{duration:.6f}",
            "zero_shot_guess": zero_shot_guess,
            "agrees_with_human": classify_agreement(corrected_target_class, action, zero_shot_guess),
            "first_display_timestamp": started_at,
            "timestamp": clicked_at,
            "session_id": state["session_id"],
        }
        append_csv_row(state["csv_path"], row)
        state["queue"].pop(0)
        state["display_times"].pop(candidate.filename, None)
        state["current_scene"].pop(candidate.filename, None)
        return redirect(url_for("index"))

    return app


def run_review(args: argparse.Namespace) -> None:
    candidates_root = Path(args.candidates_folder)
    if not candidates_root.exists():
        raise FileNotFoundError(f"Candidates folder does not exist: {candidates_root}")
    if args.target_count <= 0:
        raise ValueError("target_count must be greater than zero.")

    csv_path = Path(args.csv)
    ensure_csv(csv_path)
    app = build_app(candidates_root, args.target_count, csv_path)
    print(f"Candidates folder: {candidates_root}")
    print(f"CSV log: {csv_path}")
    print(f"Target count: {args.target_count}")
    print(f"Starting Flask review app on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


def run_finalize(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv)
    rows = read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    copy_confirmed_images(rows, Path(TRAIN_DIR), Path(HOLDOUT_DIR))
    print()
    print(build_final_report(rows))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    review = subparsers.add_parser("review", help="Start the Flask review app.")
    review.add_argument("candidates_folder", help="Folder containing candidate images or candidate subfolders.")
    review.add_argument("target_count", type=int, help="How many images to sample evenly across focal groups.")
    review.add_argument("--csv", default="results/cull_review.csv", help="CSV log path.")
    review.add_argument("--host", default="127.0.0.1", help="Flask host.")
    review.add_argument("--port", default=5000, type=int, help="Flask port.")
    review.set_defaults(func=run_review)

    finalize = subparsers.add_parser("finalize", help="Copy confirmed files into train/holdout and print metrics.")
    finalize.add_argument("csv", help="CSV log produced by review mode.")
    finalize.set_defaults(func=run_finalize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
