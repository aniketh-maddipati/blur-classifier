"""
exif_analyzer.py
-----------------
Core diagnostic engine for the photo-culling / camera-coaching app.

Responsibilities:
    1. Extract key EXIF metadata from an image file using `exifread`.
    2. Normalize that metadata into plain Python types (floats/ints).
    3. Run a rule-based diagnostic pass to catch three common shooting
       errors: motion blur, unnecessary noise, and dangerously thin
       depth of field.
    4. Return a single JSON-serializable payload combining metadata +
       human-readable "next time" advice.

Dependencies:
    pip install exifread
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from fractions import Fraction
from typing import Optional

import exifread


# --------------------------------------------------------------------------
# 1. DATA MODELS
# --------------------------------------------------------------------------

@dataclass
class ShotMetadata:
    """Normalized EXIF values we actually care about for diagnostics."""
    aperture: Optional[float] = None       # f-number, e.g. 2.8
    shutter_speed: Optional[float] = None  # in seconds, e.g. 0.008 (== 1/125s)
    iso: Optional[int] = None
    focal_length: Optional[float] = None   # in mm
    camera_model: Optional[str] = None


@dataclass
class DiagnosticIssue:
    """A single detected shooting problem."""
    code: str
    severity: str          # "low" | "medium" | "high"
    message: str            # plain-English explanation of what happened
    advice: str              # plain-English "next time" fix


@dataclass
class ShotReport:
    file_path: str
    metadata: ShotMetadata
    issues: list[DiagnosticIssue] = field(default_factory=list)
    primary_advice: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# --------------------------------------------------------------------------
# 2. EXIF EXTRACTION
# --------------------------------------------------------------------------

def _ratio_to_float(value) -> Optional[float]:
    """exifread returns Ratio-like objects for fractional tags; normalize to float."""
    if value is None:
        return None
    try:
        # exifread tags stringify like "1/125" or "28"
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return None


def extract_metadata(image_path: str) -> ShotMetadata:
    """Reads the EXIF block from `image_path` and returns normalized values."""
    with open(image_path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    aperture = _ratio_to_float(tags.get("EXIF FNumber"))
    shutter_speed = _ratio_to_float(tags.get("EXIF ExposureTime"))
    focal_length = _ratio_to_float(tags.get("EXIF FocalLength"))

    iso_tag = tags.get("EXIF ISOSpeedRatings")
    iso = int(str(iso_tag)) if iso_tag else None

    camera_model_tag = tags.get("Image Model")
    camera_model = str(camera_model_tag).strip() if camera_model_tag else None

    return ShotMetadata(
        aperture=aperture,
        shutter_speed=shutter_speed,
        iso=iso,
        focal_length=focal_length,
        camera_model=camera_model,
    )


# --------------------------------------------------------------------------
# 3. DIAGNOSTIC RULES
# --------------------------------------------------------------------------
# Tunable thresholds — expose these as user/app settings later, or derive
# CROP_FACTOR automatically from camera_model via a lookup table.

HIGH_ISO_THRESHOLD = 800
FAST_SHUTTER_THRESHOLD = 1 / 250     # seconds; "fast" enough that high ISO wasn't required
WIDE_APERTURE_THRESHOLD = 2.8        # f-number at/under this counts as "wide open"
LONG_FOCAL_LENGTH_THRESHOLD = 85     # mm; portrait/tele range where thin-DOF risk rises
DANGEROUS_APERTURE_THRESHOLD = 2.0   # f-number under this is "wide open" for DOF purposes
CROP_FACTOR = 1.0                    # 1.0 = full frame; 1.5/1.6 = APS-C; 2.0 = m4/3


def _check_motion_blur(m: ShotMetadata) -> Optional[DiagnosticIssue]:
    """
    Reciprocal Rule: for hand-held shots, shutter speed should be at least
    1 / (focal_length * crop_factor) to avoid camera-shake blur.
    e.g. at 50mm full-frame, you want 1/50s or faster.
    """
    if not m.shutter_speed or not m.focal_length:
        return None

    effective_focal_length = m.focal_length * CROP_FACTOR
    safe_shutter_speed = 1 / effective_focal_length  # in seconds

    if m.shutter_speed > safe_shutter_speed:
        safe_display = f"1/{round(1 / safe_shutter_speed)}s"
        actual_display = _seconds_to_shutter_string(m.shutter_speed)
        return DiagnosticIssue(
            code="MOTION_BLUR_RISK",
            severity="high",
            message=(
                f"Your shutter speed ({actual_display}) was slower than the "
                f"recommended minimum for a {m.focal_length:.0f}mm lens "
                f"({safe_display}), so hand-shake blur is likely."
            ),
            advice=(
                f"Next time, keep your shutter speed at {safe_display} or faster "
                f"when shooting hand-held at {m.focal_length:.0f}mm. Raise your ISO "
                f"or open your aperture to compensate for the light loss."
            ),
        )
    return None


def _check_unnecessary_noise(m: ShotMetadata) -> Optional[DiagnosticIssue]:
    """
    Flags high ISO usage when the shutter was already fast and/or the
    aperture was already wide — meaning there was room to lower ISO and
    reduce noise without under-exposing the shot.
    """
    if not m.iso or m.iso < HIGH_ISO_THRESHOLD:
        return None

    shutter_was_fast = m.shutter_speed is not None and m.shutter_speed <= FAST_SHUTTER_THRESHOLD
    aperture_was_wide = m.aperture is not None and m.aperture <= WIDE_APERTURE_THRESHOLD

    if shutter_was_fast or aperture_was_wide:
        reasons = []
        if shutter_was_fast:
            reasons.append(f"your shutter speed was already fast ({_seconds_to_shutter_string(m.shutter_speed)})")
        if aperture_was_wide:
            reasons.append(f"your aperture was already wide (f/{m.aperture:g})")
        reason_str = " and ".join(reasons)

        return DiagnosticIssue(
            code="UNNECESSARY_NOISE",
            severity="medium",
            message=(
                f"ISO {m.iso} likely added visible noise/grain, but {reason_str}, "
                f"so there was headroom to shoot at a lower ISO."
            ),
            advice=(
                "Next time, lower your ISO manually before the camera pushes it up "
                "automatically — you had enough shutter speed or aperture budget to "
                "spare without risking blur or an unintentionally soft background."
            ),
        )
    return None


def _check_dangerous_dof(m: ShotMetadata) -> Optional[DiagnosticIssue]:
    """
    Flags a wide-open aperture on a longer lens, where depth of field
    becomes razor-thin and small focus misses (e.g. missing the eyes in
    a portrait) become likely.
    """
    if not m.aperture or not m.focal_length:
        return None

    if m.aperture < DANGEROUS_APERTURE_THRESHOLD and m.focal_length >= LONG_FOCAL_LENGTH_THRESHOLD:
        return DiagnosticIssue(
            code="DANGEROUS_DOF",
            severity="medium",
            message=(
                f"Shooting at f/{m.aperture:g} on a {m.focal_length:.0f}mm lens gives "
                f"an extremely thin plane of focus, raising the risk of a soft or "
                f"missed subject."
            ),
            advice=(
                f"Next time at {m.focal_length:.0f}mm, consider closing down to "
                f"f/{DANGEROUS_APERTURE_THRESHOLD:.1f}\u2013f/4 for portraits or moving "
                f"subjects, and let shutter speed or ISO carry the rest of your exposure."
            ),
        )
    return None


def _seconds_to_shutter_string(seconds: float) -> str:
    """Formats a shutter speed in seconds as a human-readable fraction string."""
    if seconds >= 1:
        return f"{seconds:g}s"
    denominator = round(1 / seconds)
    return f"1/{denominator}s"


# --------------------------------------------------------------------------
# 4. ORCHESTRATION
# --------------------------------------------------------------------------

_RULES = [_check_motion_blur, _check_unnecessary_noise, _check_dangerous_dof]


def diagnose(metadata: ShotMetadata) -> list[DiagnosticIssue]:
    """Runs every diagnostic rule and collects any issues found."""
    return [issue for rule in _RULES if (issue := rule(metadata)) is not None]


def _pick_primary_advice(issues: list[DiagnosticIssue]) -> Optional[str]:
    """Surfaces the single most important piece of advice for the UI headline."""
    if not issues:
        return None
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    ranked = sorted(issues, key=lambda i: severity_rank.get(i.severity, 3))
    return ranked[0].advice


def analyze_shot(image_path: str) -> ShotReport:
    """Main entry point: extract metadata, diagnose, and build the report."""
    metadata = extract_metadata(image_path)
    issues = diagnose(metadata)
    primary_advice = _pick_primary_advice(issues)

    return ShotReport(
        file_path=image_path,
        metadata=metadata,
        issues=issues,
        primary_advice=primary_advice,
    )


# --------------------------------------------------------------------------
# 5. CLI / STANDALONE TEST
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python exif_analyzer.py <path_to_image.jpg>")
        sys.exit(1)

    report = analyze_shot(sys.argv[1])
    print(report.to_json())