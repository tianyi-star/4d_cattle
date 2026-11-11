# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Ordered research pipeline.

The call order is the method:

    coat pattern -> 4D Gaussians -> denoising -> generation

Gait recognition reads the denoised state and does not sit on the generative
path. ``forward`` writes that order down. Execution stops at the first stage
that is still an open entry point, which today is the coat encoder.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from cattle4d.models.coat_pattern import CoatPatternEncoder
from cattle4d.models.denoiser import GaussianDenoiser
from cattle4d.models.gaussian4d import Gaussian4D
from cattle4d.models.generator import GaitGenerator
from cattle4d.models.recognizer import GaitRecognizer


class Cattle4DPipeline(nn.Module):
    """Hold the five stages and expose one training/inference call."""

    def __init__(
        self,
        coat_embed_dim: int,
        sh_degree: int,
        denoiser_hidden_dim: int,
        history: int | None,
        horizon: int | None,
        labels: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.coat = CoatPatternEncoder(coat_embed_dim)
        self.gaussians = Gaussian4D(sh_degree, coat_embed_dim)
        self.denoiser = GaussianDenoiser(denoiser_hidden_dim)
        self.generator = GaitGenerator(history, horizon)
        self.recognizer = GaitRecognizer(labels)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run one batch through the documented order.

        Parameters
        ----------
        batch :
            Keys consumed once the stages are implemented:

            ``coat_images`` ``[B, 3, H, W]``, optional ``coat_masks``,
            ``points_xyz`` ``[N, 3]`` or a per-batch cloud, ``times``
            ``[B, T]``, ``future_times`` ``[B, horizon]``, and ``cameras``.

        Returns
        -------
        dict
            ``replay``, ``prediction``, and ``logits`` once every stage exists.
        """

        self._require_batch_keys(batch)
        coat = self.coat(batch["coat_images"], batch.get("coat_masks"))
        canonical = self.gaussians.initialize_from_points(batch["points_xyz"])
        deformed = self.gaussians.deform(canonical, batch["times"], coat)
        denoised = self.denoiser(deformed)
        replay = self.generator.replay(denoised, batch["cameras"], batch["times"])
        prediction = self.generator.predict(
            denoised, batch["cameras"], batch["future_times"]
        )
        logits = self.recognizer(denoised)
        return {"replay": replay, "prediction": prediction, "logits": logits}

    def describe(self) -> str:
        """Summarize the constructed graph. This does not run the stages."""

        if self.recognizer.labels:
            label_text = ", ".join(self.recognizer.labels)
        else:
            label_text = "unset"
        history = "unset" if self.generator.history is None else str(self.generator.history)
        horizon = "unset" if self.generator.horizon is None else str(self.generator.horizon)
        return "\n".join(
            [
                "stages: coat_pattern -> gaussian4d -> denoiser -> generator",
                "side branch: denoised state -> gait_recognizer",
                f"coat_embed_dim: {self.coat.embed_dim}",
                f"sh_degree: {self.gaussians.sh_degree}",
                f"denoiser_hidden_dim: {self.denoiser.hidden_dim}",
                f"history: {history}",
                f"horizon: {horizon}",
                f"recognition labels: {label_text}",
                "parameters: none allocated",
                "weights: none released; stage forwards are unimplemented",
            ]
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _require_batch_keys(batch: dict[str, Any]) -> None:
        required = ("coat_images", "points_xyz", "times", "future_times", "cameras")
        if not isinstance(batch, dict):
            raise TypeError("batch must be a dict.")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError("batch is missing keys: " + ", ".join(missing))


def build_pipeline(config: dict[str, Any]) -> Cattle4DPipeline:
    """Construct the pipeline from the experiment YAML. No weights are loaded."""

    model = config["model"]
    labels = tuple(str(name) for name in model["recognizer"]["labels"])
    generator = model.get("generator", {})
    denoiser = model.get("denoiser", {})
    hidden = denoiser.get("hidden_dim", 128)
    pipeline = Cattle4DPipeline(
        coat_embed_dim=int(model["coat_embed_dim"]),
        sh_degree=int(model["gaussian"]["sh_degree"]),
        denoiser_hidden_dim=int(hidden),
        history=generator.get("history"),
        horizon=generator.get("horizon"),
        labels=labels,
    )
    return pipeline
