# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Locate the coat reference that conditions appearance.

The reference is an image on disk, typically a cropped animal or a pattern
mask. Decoding and encoding belong to the coat-pattern module, which is still
an open experiment entry point.
"""

from __future__ import annotations

from pathlib import Path

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_PREFERRED = ("reference.png", "reference.jpg", "reference.jpeg")


def find_coat_reference(sequence_dir: Path) -> Path | None:
    """Return the coat image for ``sequence_dir``, if one was provided."""

    coat_dir = Path(sequence_dir) / "coat"
    if not coat_dir.is_dir():
        return None
    for name in _PREFERRED:
        candidate = coat_dir / name
        if candidate.is_file():
            return candidate
    images = sorted(
        (path for path in coat_dir.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES),
        key=lambda path: path.name,
    )
    if not images:
        return None
    return images[0]
