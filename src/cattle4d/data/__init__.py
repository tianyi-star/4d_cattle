# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Sequence layout for multi-view cattle gait."""

from cattle4d.data.dataset import GaitSequenceDataset
from cattle4d.data.multiview import load_sequence, validate_sequence
from cattle4d.data.schema import Camera, GaitLabel, Sequence

__all__ = [
    "Camera",
    "GaitLabel",
    "GaitSequenceDataset",
    "Sequence",
    "load_sequence",
    "validate_sequence",
]
