# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Dynamic Gaussians for a walking animal.

The intended representation follows the canonical-plus-deformation reading of
4D Gaussian Splatting: a set of 3D Gaussians in a rest pose, and a map that
sends those Gaussians to each timestamp of the gait. Rendering from a known
camera then supports both replay of observed time and, downstream, a forecast.

No rasterizer is vendored here, and no point cloud is initialized. Fill in
``initialize_from_points``, ``deform``, and ``render``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cattle4d.models.checks import check_time_batch


@dataclass
class GaussianState:
    """Attributes of one set of Gaussians at one time, or in the rest pose.

    An empty state, with every tensor left as ``None``, is the value returned
    before initialization. It is not a reconstruction and it must not be
    written out as a result.
    """

    means: torch.Tensor | None = None
    scales: torch.Tensor | None = None
    quaternions: torch.Tensor | None = None
    opacities: torch.Tensor | None = None
    features: torch.Tensor | None = None
    times: torch.Tensor | None = None

    def require_initialized(self) -> None:
        missing = [
            name
            for name in ("means", "scales", "quaternions", "opacities", "features")
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                "GaussianState has no "
                + ", ".join(missing)
                + ". Initialize the canonical Gaussians from the sequence before rendering."
            )


class Gaussian4D(nn.Module):
    """Canonical Gaussians and the deformation that plays a gait.

    Parameters
    ----------
    sh_degree :
        Spherical-harmonics degree reserved for the appearance features.
        Stored for the experiment; no coefficient tensor is allocated.
    coat_embed_dim :
        Width of the coat embedding that conditions deformation.
    """

    def __init__(self, sh_degree: int, coat_embed_dim: int) -> None:
        super().__init__()
        if sh_degree < 0:
            raise ValueError("sh_degree must be zero or positive.")
        if coat_embed_dim <= 0:
            raise ValueError("coat_embed_dim must be positive.")
        self.sh_degree = int(sh_degree)
        self.coat_embed_dim = int(coat_embed_dim)

    def canonical_state(self) -> GaussianState:
        """Return the rest-pose container. It stays empty until initialized."""

        return GaussianState()

    def initialize_from_points(self, points_xyz: torch.Tensor) -> GaussianState:
        """Build the canonical Gaussians from a 3D sample of the animal.

        Parameters
        ----------
        points_xyz :
            Float tensor ``[N, 3]`` in the sequence world frame. A usual source
            is a fused multi-view point cloud, or a sample conditioned on the
            coat embedding. This method does not sample either source.

        Returns
        -------
        GaussianState
            Populated canonical attributes. Implement the initialization here.
        """

        if not isinstance(points_xyz, torch.Tensor):
            raise TypeError("points_xyz must be a torch.Tensor.")
        if points_xyz.ndim != 2 or points_xyz.shape[1] != 3 or points_xyz.shape[0] == 0:
            raise ValueError(f"points_xyz must have shape [N, 3], got {tuple(points_xyz.shape)}.")
        raise NotImplementedError(
            "Gaussian4D.initialize_from_points is the canonical-Gaussian entry point. "
            "Do not load a checked-in weight file; this repository has none."
        )

    def deform(
        self,
        canonical: GaussianState,
        times: torch.Tensor,
        coat_embed: torch.Tensor,
    ) -> GaussianState:
        """Move the rest pose to each requested timestamp.

        Parameters
        ----------
        canonical :
            Rest-pose state from ``initialize_from_points``.
        times :
            Float tensor ``[B]`` or ``[B, T]``, in seconds or in a normalized
            gait phase. The unit is fixed by the experiment config.
        coat_embed :
            Float tensor ``[B, coat_embed_dim]`` from ``CoatPatternEncoder``.

        Returns
        -------
        GaussianState
            Deformed attributes. ``times`` on the returned state should record
            the query time. Implement the deformation field here.
        """

        if not isinstance(canonical, GaussianState):
            raise TypeError("canonical must be a GaussianState.")
        canonical.require_initialized()
        check_time_batch(times)
        if not isinstance(coat_embed, torch.Tensor):
            raise TypeError("coat_embed must be a torch.Tensor.")
        if coat_embed.ndim != 2 or coat_embed.shape[1] != self.coat_embed_dim:
            raise ValueError(
                f"coat_embed must have shape [B, {self.coat_embed_dim}], "
                f"got {tuple(coat_embed.shape)}."
            )
        if coat_embed.shape[0] != times.shape[0]:
            raise ValueError("coat_embed and times must share the batch dimension.")
        raise NotImplementedError(
            "Gaussian4D.deform is the dynamic-Gaussian entry point. "
            "Map (means, time, coat embedding) to per-frame attribute deltas."
        )

    def render(
        self,
        state: GaussianState,
        cameras: dict,
        times: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Rasterize ``state`` for each camera at ``times``.

        Parameters
        ----------
        state :
            Deformed, and usually denoised, Gaussians.
        cameras :
            Mapping of camera name to the calibration dict produced by
            ``Sequence.to_dict``.
        times :
            The timestamps ``state`` was deformed to.

        Returns
        -------
        dict
            Camera name to a float image tensor ``[B, T, 3, H, W]`` or
            ``[B, 3, H, W]``. Implement the rasterizer, or the call into an
            external one, here. No renderer is bundled.
        """

        if not isinstance(state, GaussianState):
            raise TypeError("state must be a GaussianState.")
        state.require_initialized()
        check_time_batch(times)
        if not isinstance(cameras, dict) or not cameras:
            raise ValueError("cameras must be a non-empty mapping of calibrated views.")
        raise NotImplementedError(
            "Gaussian4D.render is the splatting entry point. "
            "Connect a Gaussian rasterizer here; none is vendored in this repository."
        )
