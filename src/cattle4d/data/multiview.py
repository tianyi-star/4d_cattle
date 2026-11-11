# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Read one multi-camera sequence from the on-disk layout.

Expected tree, also shown in the README::

    <sequence_id>/
      cameras.yaml
      gait.json              # optional
      coat/reference.jpg     # optional
      views/<camera_name>/*.jpg

``cameras.yaml`` follows ``configs/example_sequence/cameras.yaml``.
Frame files for every camera must line up by sorted filename. This module
checks that contract and does not invent missing views.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from cattle4d.data.coat import find_coat_reference
from cattle4d.data.schema import Camera, GaitLabel, Sequence

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def validate_sequence(sequence_dir: Path) -> list[str]:
    """Return human-readable problems. An empty list means the clip is usable."""

    sequence_dir = Path(sequence_dir)
    problems: list[str] = []
    if not sequence_dir.is_dir():
        return [f"Sequence directory does not exist: {sequence_dir}"]
    camera_path = sequence_dir / "cameras.yaml"
    if not camera_path.is_file():
        problems.append("Missing cameras.yaml.")
        cameras: dict[str, Camera] = {}
    else:
        try:
            cameras = _read_cameras(camera_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            problems.append(f"cameras.yaml could not be read: {exc}")
            cameras = {}
    view_root = sequence_dir / "views"
    if not view_root.is_dir():
        problems.append("Missing views/ directory.")
        return problems
    counts: dict[str, int] = {}
    for name in cameras:
        view_dir = view_root / name
        if not view_dir.is_dir():
            problems.append(f"Missing views/{name}/ for a camera declared in cameras.yaml.")
            continue
        counts[name] = len(_list_frames(view_dir))
        if counts[name] == 0:
            problems.append(f"views/{name}/ has no images.")
    extra = sorted(
        path.name for path in view_root.iterdir() if path.is_dir() and path.name not in cameras
    )
    for name in extra:
        problems.append(f"views/{name}/ has no matching camera in cameras.yaml.")
    positive = [count for count in counts.values() if count > 0]
    if len(set(positive)) > 1:
        detail = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        problems.append(f"Cameras do not share a frame count ({detail}).")
    gait_path = sequence_dir / "gait.json"
    if gait_path.exists() and not gait_path.is_file():
        problems.append("gait.json exists but is not a file.")
    elif gait_path.is_file():
        try:
            _read_gait(gait_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            problems.append(f"gait.json could not be read: {exc}")
    return problems


def load_sequence(sequence_dir: Path) -> Sequence:
    """Load metadata and frame paths. Image pixels are left on disk."""

    sequence_dir = Path(sequence_dir)
    problems = validate_sequence(sequence_dir)
    if problems:
        joined = "\n".join(f"- {item}" for item in problems)
        raise ValueError(f"Sequence {sequence_dir.name} is not ready:\n{joined}")
    cameras = _read_cameras(sequence_dir / "cameras.yaml")
    frames = {name: _list_frames(sequence_dir / "views" / name) for name in cameras}
    document = _read_camera_document(sequence_dir / "cameras.yaml")
    fps = float(document.get("fps", 1.0))
    cow_id, labels = _read_gait(sequence_dir / "gait.json")
    return Sequence(
        sequence_id=sequence_dir.name,
        root=sequence_dir,
        fps=fps,
        cameras=cameras,
        frames=frames,
        cow_id=cow_id,
        labels=labels,
        coat_path=find_coat_reference(sequence_dir),
    )


def _read_camera_document(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"{path} must be a mapping.")
    return document


def _read_cameras(path: Path) -> dict[str, Camera]:
    document = _read_camera_document(path)
    raw = document.get("cameras")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("cameras.yaml needs a non-empty 'cameras' mapping.")
    cameras: dict[str, Camera] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Camera {name} must be a mapping.")
        camera = Camera(
            name=str(name),
            width=_positive_int(spec.get("width"), f"{name}.width"),
            height=_positive_int(spec.get("height"), f"{name}.height"),
            K=_matrix(spec.get("K"), (3, 3), f"{name}.K"),
            c2w=_matrix(spec.get("c2w"), (4, 4), f"{name}.c2w"),
        )
        if not np.isfinite(camera.K).all() or not np.isfinite(camera.c2w).all():
            raise ValueError(f"Camera {name} has a non-finite K or c2w.")
        cameras[camera.name] = camera
    return cameras


def _read_gait(path: Path) -> tuple[str | None, list[GaitLabel]]:
    if not path.is_file():
        return None, []
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("gait.json must be an object.")
    cow_id = document.get("cow_id")
    if cow_id is not None:
        cow_id = str(cow_id)
    labels: list[GaitLabel] = []
    for item in document.get("labels", []):
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError("Each gait label needs a 'name'.")
        labels.append(
            GaitLabel(
                frame=int(item.get("frame", 0)),
                timestamp=float(item.get("timestamp", 0.0)),
                name=str(item["name"]),
            )
        )
    return cow_id, labels


def _list_frames(view_dir: Path) -> list[Path]:
    return sorted(
        (path for path in view_dir.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES),
        key=lambda path: path.name,
    )


def _matrix(value: object, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    return array


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive integer.")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return number
