"""Project the frozen MIPDS vision and text features into a joint representation."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from mipds_language.pretrained import PretrainedNotes, VERSION as TEXT_VERSION


VISION_VERSION = "MIPDS-Vision-ResNet50-exercise-v1-0d59d2a0080d"
FUSION_VERSION = "MIPDS-Fusion-ProjectedConcat-256x2-v1"
PROJECTION_WIDTH = 256
FUSED_WIDTH = 512


@dataclass(frozen=True)
class FusionOutput:
    fused_embedding: torch.Tensor
    modality_mask: torch.Tensor
    has_information: torch.Tensor
    text_truncated: torch.Tensor
    model_version: str = FUSION_VERSION


class ProjectedConcatFusion(nn.Module):
    """Align by clip ID, project each modality, then concatenate vision before text."""

    def __init__(self):
        super().__init__()
        self.vision_projection = nn.Sequential(
            nn.Linear(2048, PROJECTION_WIDTH), nn.LayerNorm(PROJECTION_WIDTH), nn.GELU(),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(768, PROJECTION_WIDTH), nn.LayerNorm(PROJECTION_WIDTH), nn.GELU(),
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        language: PretrainedNotes,
        *,
        vision_clip_ids: Sequence[str],
        note_clip_ids: Sequence[str],
        vision_present: torch.Tensor | None = None,
        vision_version: str = VISION_VERSION,
    ) -> FusionOutput:
        if vision_version != VISION_VERSION or language.model_version != TEXT_VERSION:
            raise ValueError("Fusion requires the recorded vision and pretrained language versions.")
        if vision_features.ndim != 2 or vision_features.shape[1] != 2048:
            raise ValueError("Expected vision features of shape B × 2048.")
        count = vision_features.shape[0]
        if vision_features.dtype != torch.float32:
            raise ValueError("Vision features must use float32.")
        if language.pooled_embedding.shape != (count, 768) or language.pooled_embedding.dtype != np.float32:
            raise ValueError("Expected language features of shape B × 768 in float32.")
        if len(vision_clip_ids) != count or len(note_clip_ids) != count:
            raise ValueError("Supply one clip ID for each modality row.")
        if tuple(vision_clip_ids) != tuple(note_clip_ids) or any(not isinstance(i, str) or not i for i in vision_clip_ids):
            raise ValueError("The vision and note rows must have matching nonempty clip IDs.")
        if any(flag.shape != (count,) or flag.dtype != np.bool_
               for flag in (language.has_text, language.truncated)):
            raise ValueError("Language flags must be boolean vectors of length B.")

        device = vision_features.device
        if vision_present is None:
            vision_present = torch.ones(count, dtype=torch.bool, device=device)
        if vision_present.shape != (count,) or vision_present.dtype != torch.bool or vision_present.device != device:
            raise ValueError("Vision presence must be a boolean vector on the feature device.")
        text_present = torch.as_tensor(language.has_text, device=device)
        text_features = torch.as_tensor(language.pooled_embedding, device=device)
        vision_input = torch.where(vision_present[:, None], vision_features, torch.zeros_like(vision_features))
        text_input = torch.where(text_present[:, None], text_features, torch.zeros_like(text_features))
        if not torch.isfinite(vision_input).all() or not torch.isfinite(text_input).all():
            raise ValueError("Present modality features must be finite.")
        vision = self.vision_projection(vision_input) * vision_present[:, None]
        text = self.text_projection(text_input) * text_present[:, None]
        present = torch.stack((vision_present, text_present), dim=1)
        return FusionOutput(
            torch.cat((vision, text), dim=1), present, present.any(dim=1),
            torch.as_tensor(language.truncated, device=device).clone(),
        )
