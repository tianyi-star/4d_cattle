# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Shape checks used before an unimplemented stage returns.

The checks describe the tensor contract. They are not a network.
"""

from __future__ import annotations

import torch


def check_image_batch(images: torch.Tensor) -> None:
    """Require a floating-point batch of RGB images, ``[B, 3, H, W]``."""

    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor.")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"images must have shape [B, 3, H, W], got {tuple(images.shape)}.")
    if images.dtype not in (torch.float16, torch.float32, torch.float64):
        raise ValueError(f"images must be floating point, got {images.dtype}.")
    if images.shape[0] == 0 or images.shape[2] == 0 or images.shape[3] == 0:
        raise ValueError("images has an empty batch or spatial size.")


def check_mask_batch(masks: torch.Tensor, batch: int) -> None:
    """Require a single-channel mask aligned with an image batch."""

    if not isinstance(masks, torch.Tensor):
        raise TypeError("masks must be a torch.Tensor.")
    if masks.ndim != 4 or masks.shape[0] != batch or masks.shape[1] != 1:
        raise ValueError(
            f"masks must have shape [{batch}, 1, H, W], got {tuple(masks.shape)}."
        )


def check_time_batch(times: torch.Tensor) -> None:
    """Require per-sample timestamps, ``[B]`` or ``[B, T]``."""

    if not isinstance(times, torch.Tensor):
        raise TypeError("times must be a torch.Tensor.")
    if times.ndim not in (1, 2) or times.shape[0] == 0:
        raise ValueError(f"times must have shape [B] or [B, T], got {tuple(times.shape)}.")
    if times.dtype not in (torch.float16, torch.float32, torch.float64):
        raise ValueError(f"times must be floating point, got {times.dtype}.")
