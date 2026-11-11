# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Pinhole helpers for the multi-camera rig.

These functions only apply a calibrated projection. They do not estimate a
camera and they do not score a reconstruction.
"""

from __future__ import annotations

import numpy as np


def invert_rigid(c2w: np.ndarray) -> np.ndarray:
    """Invert a camera-to-world rigid transform.

    Parameters
    ----------
    c2w :
        Array of shape ``(4, 4)``. The upper-left 3x3 block is a rotation.
    """

    matrix = _as_matrix(c2w, (4, 4), "c2w")
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    world_to_cam = np.eye(4, dtype=np.float64)
    world_to_cam[:3, :3] = rotation.T
    world_to_cam[:3, 3] = -rotation.T @ translation
    return world_to_cam


def project_points(
    points_xyz: np.ndarray,
    K: np.ndarray,
    world_to_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points with a pinhole camera.

    Parameters
    ----------
    points_xyz :
        Array of shape ``(N, 3)`` in the same world frame as the cameras.
    K :
        Intrinsics, shape ``(3, 3)``.
    world_to_cam :
        Extrinsics, shape ``(4, 4)``.

    Returns
    -------
    pixels, depth
        ``pixels`` has shape ``(N, 2)`` in ``(x, y)`` order. ``depth`` has
        shape ``(N,)`` and is the camera-frame ``z``. Points with non-positive
        depth are still returned; the caller decides whether they are valid.
    """

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_xyz must have shape (N, 3), got {points.shape}.")
    intrinsics = _as_matrix(K, (3, 3), "K")
    extrinsics = _as_matrix(world_to_cam, (4, 4), "world_to_cam")
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    camera = (extrinsics @ homogeneous.T).T[:, :3]
    depth = camera[:, 2]
    projected = (intrinsics @ camera.T).T
    pixels = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    valid = depth != 0.0
    pixels[valid] = projected[valid, :2] / depth[valid, None]
    return pixels, depth


def _as_matrix(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    return array
