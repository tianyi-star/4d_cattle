# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Coat-pattern encoder.

The coat is the stable appearance cue that should stay attached to an animal
while the body moves. A later experiment can drop in a ``timm`` backbone, or
an ``ultralytics`` detector that crops the animal before the backbone. This
file only fixes the call signature.
"""

from __future__ import annotations

import torch
from torch import nn

from cattle4d.models.checks import check_image_batch, check_mask_batch


class CoatPatternEncoder(nn.Module):
    """Map a coat reference to an identity appearance vector.

    Parameters
    ----------
    embed_dim :
        Width of the vector consumed by the deformation field. The value is
        stored so the rest of the pipeline can agree on a width. No layers
        are allocated, and no pretrained backbone is downloaded.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        self.embed_dim = int(embed_dim)

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode coat references.

        Parameters
        ----------
        images :
            Float tensor ``[B, 3, H, W]``. Values are expected in ``[0, 1]``,
            which the caller is responsible for scaling.
        masks :
            Optional float tensor ``[B, 1, H, W]`` that limits the pattern to
            the animal. ``None`` means the crop is already the coat region.

        Returns
        -------
        torch.Tensor
            Embedding ``[B, embed_dim]``. The current body does not return
            one; implement the encoder here.
        """

        check_image_batch(images)
        if masks is not None:
            check_mask_batch(masks, images.shape[0])
        raise NotImplementedError(
            "CoatPatternEncoder.forward is the coat-pattern entry point. "
            "Add the encoder here. A timm backbone or an ultralytics crop "
            "can be constructed in this method; neither is loaded by default."
        )
