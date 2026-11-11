# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Model interfaces for the gait pipeline.

Importing a class does not load a checkpoint. Each ``forward`` method is an
experiment entry point and raises ``NotImplementedError`` until it is filled in.
"""

from cattle4d.models.coat_pattern import CoatPatternEncoder
from cattle4d.models.denoiser import GaussianDenoiser
from cattle4d.models.gaussian4d import Gaussian4D, GaussianState
from cattle4d.models.generator import GaitGenerator
from cattle4d.models.recognizer import GaitRecognizer

__all__ = [
    "CoatPatternEncoder",
    "Gaussian4D",
    "GaussianDenoiser",
    "GaussianState",
    "GaitGenerator",
    "GaitRecognizer",
]
