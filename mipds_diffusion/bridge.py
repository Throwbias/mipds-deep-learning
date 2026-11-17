"""Route confirmed text from MIPDS to the diffusion model's own CLIP encoder.

The M10 vector validates source availability. It is not mapped to CLIP tokens
or passed to the denoiser as an untrained soft condition.
"""

from dataclasses import dataclass
from typing import Sequence

import torch

from mipds_fusion.fusion import FUSION_VERSION, FusionOutput


@dataclass(frozen=True)
class ReviewedPrompt:
    clip_id: str
    prompt: str
    vision_present: bool
    text_present: bool
    source_text_truncated: bool
    source_fusion_version: str = FUSION_VERSION


def prepare_prompt(
    fusion: FusionOutput, source_clip_ids: Sequence[str], *, row: int,
    trainer_text: str, confirmed: bool,
) -> ReviewedPrompt:
    """Keep source metadata while requiring an explicitly confirmed description."""
    if confirmed is not True:
        raise ValueError("Confirm the source context and the description before generation.")
    if not isinstance(trainer_text, str) or not trainer_text.strip():
        raise ValueError("Supply a nonempty description approved for illustration.")
    if fusion.model_version != FUSION_VERSION:
        raise ValueError("Use the recorded Milestone 10 fusion interface.")
    features = fusion.fused_embedding
    if features.ndim != 2 or features.shape[1] != 512 or features.dtype != torch.float32:
        raise ValueError("Fusion must supply float32 B × 512 features.")
    count = features.shape[0]
    if len(source_clip_ids) != count or any(not isinstance(i, str) or not i for i in source_clip_ids):
        raise ValueError("Supply one source clip ID per fusion row.")
    if type(row) is not int or not 0 <= row < count:
        raise ValueError("The selected source row is outside the fusion batch.")
    for flag, shape in ((fusion.modality_mask, (count, 2)),
                        (fusion.has_information, (count,)), (fusion.text_truncated, (count,))):
        if flag.shape != shape or flag.dtype != torch.bool or flag.device != features.device:
            raise ValueError("Fusion flags must match the feature rows and device.")
    if not torch.isfinite(features).all():
        raise ValueError("Fusion features must be finite.")
    if not torch.equal(fusion.has_information, fusion.modality_mask.any(1)):
        raise ValueError("Fusion availability must agree with the modality mask.")
    if not fusion.has_information[row]:
        raise ValueError("The requested source row has neither modality available.")
    for index in (0, 1):
        block = features[:, index * 256:(index + 1) * 256]
        if block[~fusion.modality_mask[:, index]].any():
            raise ValueError("Absent modalities must have zero feature blocks.")
    return ReviewedPrompt(
        source_clip_ids[row], trainer_text.strip(), bool(fusion.modality_mask[row, 0]),
        bool(fusion.modality_mask[row, 1]), bool(fusion.text_truncated[row]),
    )
