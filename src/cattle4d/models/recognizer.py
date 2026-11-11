# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Gait recognition head.

The head reads the denoised dynamic state and scores the label set declared
in ``model.recognizer.labels``. The label names are part of the annotation
protocol. This repository does not fix a taxonomy and does not report a score.
"""

from __future__ import annotations

import torch
from torch import nn

from cattle4d.models.gaussian4d import GaussianState


class GaitRecognizer(nn.Module):
    """Classify a gait from a denoised Gaussian trajectory.

    Parameters
    ----------
    labels :
        Ordered class names. An empty tuple means the protocol is not set yet;
        ``forward`` still explains where the classifier goes, and it refuses
        to invent a class list.
    """

    def __init__(self, labels: tuple[str, ...]) -> None:
        super().__init__()
        if any(not isinstance(name, str) or not name for name in labels):
            raise ValueError("labels must be non-empty strings.")
        self.labels = tuple(labels)

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def forward(self, state: GaussianState) -> torch.Tensor:
        """Return logits of shape ``[B, num_classes]``.

        Implement the readout from deformed means, or from rendered features,
        in this method. The stub allocates no classifier and returns nothing.
        """

        if not isinstance(state, GaussianState):
            raise TypeError("state must be a GaussianState.")
        state.require_initialized()
        if not self.labels:
            raise ValueError(
                "model.recognizer.labels is empty. Set the gait label names "
                "in the config before calling the recognizer."
            )
        raise NotImplementedError(
            "GaitRecognizer.forward is the recognition entry point. "
            f"It should return logits over {list(self.labels)}."
        )
