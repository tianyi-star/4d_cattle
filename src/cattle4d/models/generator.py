# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Generation of a walking replay and of a future gait.

Replay asks the denoised 4D Gaussians for the observed timestamps. Prediction
asks for timestamps past the observed window. Both paths share the denoised
state so the coat and the body stay consistent. Filling either path in is
the generative experiment.
"""

from __future__ import annotations

import torch
from torch import nn

from cattle4d.models.checks import check_time_batch
from cattle4d.models.gaussian4d import GaussianState


class GaitGenerator(nn.Module):
    """Produce replay frames and forecast frames from a denoised state.

    Parameters
    ----------
    history :
        Number of observed frames the forecast may condition on. ``None``
        until the experiment sets ``model.generator.history``.
    horizon :
        Number of future frames to predict. ``None`` until
        ``model.generator.horizon`` is set.
    """

    def __init__(self, history: int | None, horizon: int | None) -> None:
        super().__init__()
        self.history = _optional_positive(history, "history")
        self.horizon = _optional_positive(horizon, "horizon")

    def replay(self, state: GaussianState, cameras: dict, times: torch.Tensor) -> dict:
        """Render the observed window.

        Returns a mapping from camera name to images. Implement this by
        calling the 4D Gaussian renderer on ``times``, or by a learned
        refinement of that rendering. The stub does not write images.
        """

        self._check(state, cameras, times)
        raise NotImplementedError(
            "GaitGenerator.replay is the walking-replay entry point. "
            "Write observed-time frames here. No frames are synthesized by this stub."
        )

    def predict(
        self,
        state: GaussianState,
        cameras: dict,
        future_times: torch.Tensor,
    ) -> dict:
        """Forecast the gait past the observed window.

        ``future_times`` uses the same clock as the sequence. The length of
        the last dimension should match ``horizon`` once that value is set.
        """

        self._check(state, cameras, future_times)
        if self.horizon is not None and future_times.ndim == 2:
            if future_times.shape[1] != self.horizon:
                raise ValueError(
                    f"future_times has T={future_times.shape[1]}, "
                    f"but model.generator.horizon is {self.horizon}."
                )
        raise NotImplementedError(
            "GaitGenerator.predict is the gait-forecast entry point. "
            "Write future frames or future Gaussian states here."
        )

    def _check(self, state: GaussianState, cameras: dict, times: torch.Tensor) -> None:
        if not isinstance(state, GaussianState):
            raise TypeError("state must be a GaussianState.")
        state.require_initialized()
        if not isinstance(cameras, dict) or not cameras:
            raise ValueError("cameras must be a non-empty mapping.")
        check_time_batch(times)


def _optional_positive(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer or null.")
    return value
