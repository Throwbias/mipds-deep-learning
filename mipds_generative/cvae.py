"""An untrained conditional VAE for 64-pixel exercise setup proposals."""

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn

from mipds_fusion.fusion import FUSION_VERSION, FusionOutput


VERSION = "MIPDS-Generative-CVAE64-z32-c128-v1"
IMAGE_SIZE = 64
LATENT_WIDTH = 32
CONDITION_WIDTH = 128


@dataclass(frozen=True)
class PosteriorOutput:
    reconstruction: torch.Tensor
    mean: torch.Tensor
    log_variance: torch.Tensor


@dataclass(frozen=True)
class GeneratedSetups:
    pixels: torch.Tensor
    clip_ids: tuple[str, ...]
    modality_mask: torch.Tensor
    text_truncated: torch.Tensor
    seed: int
    model_version: str = VERSION
    fusion_version: str = FUSION_VERSION
    requires_trainer_review: bool = True


class ConditionalSetupVAE(nn.Module):
    """Use a fixed normal prior and inject the same condition into both paths.

    This module specifies the architecture. Constructing it initializes weights;
    it does not load a trained exercise illustration generator.
    """

    def __init__(self):
        super().__init__()
        self.condition_adapter = nn.Sequential(
            nn.Linear(515, 256), nn.SiLU(),
            nn.Linear(256, CONDITION_WIDTH), nn.LayerNorm(CONDITION_WIDTH), nn.SiLU(),
        )
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.SiLU(), nn.Flatten(1),
        )
        self.posterior = nn.Sequential(nn.Linear(4096 + CONDITION_WIDTH, 256), nn.SiLU())
        self.mean_head = nn.Linear(256, LATENT_WIDTH)
        self.log_variance_head = nn.Linear(256, LATENT_WIDTH)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_WIDTH + CONDITION_WIDTH, 4096), nn.SiLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def condition_input(self, fusion: FusionOutput) -> torch.Tensor:
        """Validate the M10 contract and append vision, text, and truncation flags."""
        features = fusion.fused_embedding
        device = next(self.parameters()).device
        if fusion.model_version != FUSION_VERSION:
            raise ValueError("Use the recorded Milestone 10 fusion interface.")
        if features.ndim != 2 or features.shape[1] != 512 or features.shape[0] < 1:
            raise ValueError("Fusion features must have shape B × 512 with B positive.")
        if features.dtype != torch.float32 or features.device != device or not torch.isfinite(features).all():
            raise ValueError("Fusion features must be finite float32 values on the model device.")
        count = features.shape[0]
        for flag, shape in (
            (fusion.modality_mask, (count, 2)),
            (fusion.has_information, (count,)),
            (fusion.text_truncated, (count,)),
        ):
            if flag.shape != shape or flag.dtype != torch.bool or flag.device != device:
                raise ValueError("Fusion flags must be boolean tensors with matching rows and device.")
        if not torch.equal(fusion.has_information, fusion.modality_mask.any(dim=1)):
            raise ValueError("Fusion availability disagrees with the modality mask.")
        if not fusion.has_information.all():
            raise ValueError("Filter rows without either modality before requesting generation.")
        for index in (0, 1):
            block = features[:, index * 256:(index + 1) * 256]
            if block[~fusion.modality_mask[:, index]].any():
                raise ValueError("An absent modality must have a zero feature block.")
        return torch.cat((
            features, fusion.modality_mask.to(torch.float32),
            fusion.text_truncated[:, None].to(torch.float32),
        ), dim=1)

    def condition(self, fusion: FusionOutput) -> torch.Tensor:
        return self.condition_adapter(self.condition_input(fusion))

    def _validate_images(self, images: torch.Tensor, count: int) -> None:
        if images.shape != (count, 3, IMAGE_SIZE, IMAGE_SIZE) or images.dtype != torch.float32:
            raise ValueError("Images must be float32 B × 3 × 64 × 64 RGB tensors.")
        if images.device != next(self.parameters()).device or not torch.isfinite(images).all():
            raise ValueError("Images must be finite and on the model device.")
        if (images < 0).any() or (images > 1).any():
            raise ValueError("The generative path uses RGB values in [0, 1].")

    def encode(self, images: torch.Tensor, fusion: FusionOutput):
        context = self.condition(fusion)
        self._validate_images(images, context.shape[0])
        state = self.posterior(torch.cat((self.image_encoder(images), context), dim=1))
        return self.mean_head(state), self.log_variance_head(state).clamp(-12, 12), context

    def decode(self, latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or latent.ndim != 2:
            raise ValueError("Latent and context inputs must both be matrices.")
        count = context.shape[0]
        if count < 1 or context.shape != (count, CONDITION_WIDTH) or latent.shape != (count, LATENT_WIDTH):
            raise ValueError("Supply matching B × 32 latent and B × 128 context tensors.")
        device = next(self.parameters()).device
        if any(value.dtype != torch.float32 or value.device != device or not torch.isfinite(value).all()
               for value in (latent, context)):
            raise ValueError("Latent and context tensors must be finite float32 values on the model device.")
        return self.decoder(torch.cat((latent, context), dim=1))

    def forward(self, images: torch.Tensor, fusion: FusionOutput, *, noise: torch.Tensor | None = None):
        mean, log_variance, context = self.encode(images, fusion)
        if noise is None:
            noise = torch.randn_like(mean)
        if (noise.shape != mean.shape or noise.dtype != mean.dtype or noise.device != mean.device
                or not torch.isfinite(noise).all()):
            raise ValueError("Posterior noise must match the finite B × 32 mean tensor.")
        latent = mean + (0.5 * log_variance).exp() * noise
        return PosteriorOutput(self.decode(latent, context), mean, log_variance)

    @torch.inference_mode()
    def sample(self, fusion: FusionOutput, *, clip_ids: Sequence[str], seed: int) -> GeneratedSetups:
        context = self.condition(fusion)
        if len(clip_ids) != context.shape[0] or any(not isinstance(i, str) or not i for i in clip_ids):
            raise ValueError("Supply the same nonempty clip IDs used by the fusion call.")
        generator = torch.Generator(device=context.device).manual_seed(seed)
        latent = torch.randn((context.shape[0], LATENT_WIDTH), generator=generator, device=context.device)
        return GeneratedSetups(
            self.decode(latent, context), tuple(clip_ids), fusion.modality_mask.clone(),
            fusion.text_truncated.clone(), seed,
        )


def vae_loss(output: PosteriorOutput, images: torch.Tensor, *, beta: float = 1.0):
    """Average per-image unit-variance Gaussian NLL and normal-prior KL.

    The reconstruction term omits the constant Gaussian normalization. Both
    terms sum over their per-image dimensions before averaging over the batch.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("The KL coefficient must be finite and nonnegative.")
    if images.ndim != 4:
        raise ValueError("Loss targets must be B × 3 × 64 × 64 images.")
    count = images.shape[0]
    if (count < 1 or images.shape != (count, 3, IMAGE_SIZE, IMAGE_SIZE)
            or images.shape != output.reconstruction.shape
            or output.mean.shape != (count, LATENT_WIDTH) or output.log_variance.shape != output.mean.shape):
        raise ValueError("Loss inputs must have matching nonempty image and posterior batches.")
    if any(value.dtype != torch.float32 or value.device != images.device or not torch.isfinite(value).all()
           for value in (images, output.reconstruction, output.mean, output.log_variance)):
        raise ValueError("Loss inputs must be finite float32 tensors on the same device.")
    reconstruction = 0.5 * (output.reconstruction - images).square().flatten(1).sum(1).mean()
    kl = 0.5 * (output.mean.square() + output.log_variance.exp() - 1 - output.log_variance).sum(1).mean()
    return {"total": reconstruction + beta * kl, "reconstruction": reconstruction, "kl": kl}
