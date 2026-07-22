from __future__ import annotations

import re


CLASSES = ("intentional_blur", "unintentional_blur", "sharp")

_SEPARATOR_RE = re.compile(r"[\s_]+")
_CANONICAL_BY_ALIAS = {
    _SEPARATOR_RE.sub("_", class_name.strip().lower()): class_name
    for class_name in CLASSES
}


def normalize(label: str) -> str:
    normalized = _SEPARATOR_RE.sub("_", label.strip().lower())
    try:
        return _CANONICAL_BY_ALIAS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown blur label {label!r}; expected one of {list(CLASSES)}") from exc


def candidates() -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for class_name in sorted(CLASSES, key=len, reverse=True):
        values.append((class_name, class_name))
        values.append((class_name.replace("_", " "), class_name))
    return values


def find_in_text(text: str) -> str | None:
    normalized = text.strip().lower()
    for needle, class_name in candidates():
        if needle in normalized:
            return class_name
    return None
