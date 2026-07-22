import pytest

from blur_labels import CLASSES, normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("intentional_blur", "intentional_blur"),
        ("Intentional Blur", "intentional_blur"),
        ("  UNINTENTIONAL BLUR  ", "unintentional_blur"),
        ("sharp", "sharp"),
        (" Sharp ", "sharp"),
    ],
)
def test_normalize_accepts_spaced_and_underscored_labels(raw, expected):
    assert normalize(raw) == expected


def test_normalize_rejects_unknown_labels():
    with pytest.raises(ValueError, match="Unknown blur label"):
        normalize("soft focus")


def test_classes_are_canonical_underscored_names():
    assert CLASSES == ("intentional_blur", "unintentional_blur", "sharp")
