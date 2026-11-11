# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Records shared by the loader and the training entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Camera:
    """One calibrated view.

    ``c2w`` maps camera coordinates to the sequence world frame.
    """

    name: str
    width: int
    height: int
    K: np.ndarray
    c2w: np.ndarray

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "K": self.K.tolist(),
            "c2w": self.c2w.tolist(),
        }


@dataclass(frozen=True)
class GaitLabel:
    """One optional annotation on a frame. The name comes from the study."""

    frame: int
    timestamp: float
    name: str

    def to_dict(self) -> dict:
        return {"frame": self.frame, "timestamp": self.timestamp, "name": self.name}


@dataclass
class Sequence:
    """One synchronized walking clip.

    Image bytes stay on disk. ``frames`` lists the sorted file for each
    camera so a later experiment can choose its own reader.
    """

    sequence_id: str
    root: Path
    fps: float
    cameras: dict[str, Camera]
    frames: dict[str, list[Path]]
    cow_id: str | None = None
    labels: list[GaitLabel] = field(default_factory=list)
    coat_path: Path | None = None

    @property
    def num_frames(self) -> int:
        if not self.frames:
            return 0
        return len(next(iter(self.frames.values())))

    def to_dict(self) -> dict:
        return {
            "sequence_id": self.sequence_id,
            "root": str(self.root),
            "fps": self.fps,
            "cow_id": self.cow_id,
            "num_frames": self.num_frames,
            "cameras": {name: camera.to_dict() for name, camera in self.cameras.items()},
            "image_paths": {
                name: [str(path) for path in paths] for name, paths in self.frames.items()
            },
            "labels": [label.to_dict() for label in self.labels],
            "coat_path": None if self.coat_path is None else str(self.coat_path),
        }
