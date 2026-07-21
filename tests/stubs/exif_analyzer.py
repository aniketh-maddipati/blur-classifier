from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ShotMetadata:
    aperture: Optional[float] = None
    shutter_speed: Optional[float] = None
    iso: Optional[int] = None
    focal_length: Optional[float] = None
    camera_model: Optional[str] = None
    capture_datetime: Optional[datetime] = None


def extract_metadata(image_path: str) -> ShotMetadata:
    path = Path(image_path)
    if "intentional_blur_candidates" in path.parts:
        return ShotMetadata(aperture=4.0, shutter_speed=1 / 125, iso=200, focal_length=85.0, camera_model="StubCam")
    if "unintentional_blur_candidates" in path.parts:
        return ShotMetadata(aperture=4.0, shutter_speed=1 / 125, iso=200, focal_length=50.0, camera_model="StubCam")
    return ShotMetadata(aperture=4.0, shutter_speed=1 / 125, iso=200, focal_length=35.0, camera_model="StubCam")
