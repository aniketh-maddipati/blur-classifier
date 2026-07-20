from exif_analyzer import (
    DANGEROUS_APERTURE_THRESHOLD,
    FAST_SHUTTER_THRESHOLD,
    HIGH_ISO_THRESHOLD,
    LONG_FOCAL_LENGTH_THRESHOLD,
    ShotMetadata,
    _check_dangerous_dof,
    _check_motion_blur,
    _check_unnecessary_noise,
)


def test_motion_blur_fires_when_shutter_is_slower_than_safe_threshold():
    issue = _check_motion_blur(ShotMetadata(shutter_speed=1 / 40, focal_length=50))

    assert issue is not None
    assert issue.code == "MOTION_BLUR_RISK"


def test_motion_blur_does_not_fire_when_shutter_is_at_or_faster_than_threshold():
    assert _check_motion_blur(ShotMetadata(shutter_speed=1 / 50, focal_length=50)) is None
    assert _check_motion_blur(ShotMetadata(shutter_speed=1 / 100, focal_length=50)) is None


def test_motion_blur_does_not_fire_without_shutter_or_focal_length():
    assert _check_motion_blur(ShotMetadata(shutter_speed=None, focal_length=50)) is None
    assert _check_motion_blur(ShotMetadata(shutter_speed=1 / 40, focal_length=None)) is None


def test_unnecessary_noise_requires_high_iso_plus_fast_shutter_or_wide_aperture():
    assert _check_unnecessary_noise(ShotMetadata(iso=HIGH_ISO_THRESHOLD - 1, shutter_speed=FAST_SHUTTER_THRESHOLD)) is None
    assert _check_unnecessary_noise(ShotMetadata(iso=HIGH_ISO_THRESHOLD, shutter_speed=1 / 125, aperture=4.0)) is None

    fast_shutter_issue = _check_unnecessary_noise(
        ShotMetadata(iso=HIGH_ISO_THRESHOLD, shutter_speed=FAST_SHUTTER_THRESHOLD, aperture=4.0)
    )
    wide_aperture_issue = _check_unnecessary_noise(
        ShotMetadata(iso=HIGH_ISO_THRESHOLD, shutter_speed=1 / 125, aperture=2.8)
    )

    assert fast_shutter_issue is not None
    assert fast_shutter_issue.code == "UNNECESSARY_NOISE"
    assert wide_aperture_issue is not None
    assert wide_aperture_issue.code == "UNNECESSARY_NOISE"


def test_dangerous_dof_requires_both_wide_aperture_and_long_focal_length():
    assert _check_dangerous_dof(
        ShotMetadata(aperture=DANGEROUS_APERTURE_THRESHOLD, focal_length=LONG_FOCAL_LENGTH_THRESHOLD)
    ) is None
    assert _check_dangerous_dof(
        ShotMetadata(aperture=1.8, focal_length=LONG_FOCAL_LENGTH_THRESHOLD - 1)
    ) is None

    issue = _check_dangerous_dof(
        ShotMetadata(aperture=1.8, focal_length=LONG_FOCAL_LENGTH_THRESHOLD)
    )

    assert issue is not None
    assert issue.code == "DANGEROUS_DOF"
