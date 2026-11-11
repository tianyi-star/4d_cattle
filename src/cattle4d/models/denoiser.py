# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Denoising stage between deformation and generation.

Dynamic Gaussians fitted to a walking animal pick up floaters and jitter in
the canonical coordinates. This module is the place to suppress that noise,
either on the Gaussian attributes or on the rendered frames, before replay
and prediction. The concrete denoiser is an open experiment.
"""

from __future__ import annotations

from torch import nn

from cattle4d.models.gaussian4d import GaussianState


class GaussianDenoiser(nn.Module):
    """Clean a deformed Gaussian state.

    Parameters
    ----------
    hidden_dim :
        Width reserved for a later denoising network. No layers are built.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        self.hidden_dim = int(hidden_dim)

    def forward(self, state: GaussianState) -> GaussianState:
        """Return a denoised copy of ``state``.

        The input must already carry initialized attributes. The method does
        not modify them in place and does not apply a filter yet.
        """

        if not isinstance(state, GaussianState):
            raise TypeError("state must be a GaussianState.")
        state.require_initialized()
        raise NotImplementedError(
            "GaussianDenoiser.forward is the denoising entry point. "
            "Suppress unstable canonical or deformed attributes here "
            "before generation."
        )
