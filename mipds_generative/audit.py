"""Check the M11 architecture using real fused inputs and an analytic image fixture."""

import argparse
from dataclasses import replace
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file, save_file
import torch

from mipds_fusion.audit import EMBEDDINGS_SHA256, parameter_digest
from mipds_fusion.fusion import ProjectedConcatFusion
from mipds_language.preprocessing import NoteTokenizer
from mipds_language.pretrained import PretrainedNoteEncoder, file_digest
from .cvae import ConditionalSetupVAE, PosteriorOutput, VERSION, vae_loss


REPO = Path(__file__).resolve().parents[1]
SEED = 5352


def rejected(action):
    try:
        action()
    except ValueError:
        return True
    return False


def run(vision_embeddings, model_dir, output_dir):
    torch.set_num_threads(2)
    prior_audit = json.loads((REPO / "experiments/m10_fusion/audit.json").read_text())
    if file_digest(vision_embeddings) != EMBEDDINGS_SHA256:
        raise ValueError("Use the original M6 cached embedding archive.")
    with np.load(vision_embeddings, allow_pickle=False) as archive:
        index = {frame: row for row, frame in enumerate(archive["frame_ids"])}
        selected = [index[pair["frame_id"]] for pair in prior_audit["pairs"]]
        vision = torch.from_numpy(archive["embeddings"][selected].copy())
    clip_ids = [pair["clip_id"] for pair in prior_audit["pairs"]]
    notes = [pair["fictional_note"] for pair in prior_audit["pairs"]]
    tokenizer = NoteTokenizer(model_dir / "vocab.txt")
    encoder = PretrainedNoteEncoder(model_dir)
    text_before = parameter_digest(encoder.model)
    language = encoder.encode(tokenizer.pack(notes))
    torch.manual_seed(SEED)
    fusion = ProjectedConcatFusion().eval().requires_grad_(False)
    fusion_before = parameter_digest(fusion)
    if fusion_before != prior_audit["fusion_state_sha256"]:
        raise ValueError("The seeded fusion state differs from Milestone 10.")

    def fuse(text=language, **kwargs):
        return fusion(vision, text, vision_clip_ids=clip_ids, note_clip_ids=clip_ids, **kwargs)

    fused = fuse()
    no_text = fuse(replace(language, has_text=np.zeros(7, dtype=bool)))
    no_vision = fuse(vision_present=torch.zeros(7, dtype=torch.bool))
    neither = fuse(replace(language, has_text=np.zeros(7, dtype=bool)),
                   vision_present=torch.zeros(7, dtype=torch.bool))
    changed_notes = list(notes)
    changed_notes[0] = changed_notes[0].replace("left", "right")
    altered = fuse(encoder.encode(tokenizer.pack(changed_notes)))

    torch.manual_seed(SEED)
    model = ConditionalSetupVAE().eval()
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    initial_digest = parameter_digest(model)
    # An analytic color grid exercises image tensors. It is not training data
    # and it is not presented as an exercise illustration or generated result.
    ramp = torch.linspace(0, 1, 64)
    x = ramp[None, :].expand(64, 64)
    y = ramp[:, None].expand(64, 64)
    fixture = torch.stack((x, y, 0.5 * (x + y))).unsqueeze(0).repeat(7, 1, 1, 1)
    posterior = model(fixture, fused, noise=torch.zeros(7, 32))
    losses = vae_loss(posterior, fixture)
    generated = model.sample(fused, clip_ids=clip_ids, seed=SEED)
    repeated = model.sample(fused, clip_ids=clip_ids, seed=SEED)
    other_seed = model.sample(fused, clip_ids=clip_ids, seed=SEED + 1)
    other_condition = model.sample(altered, clip_ids=clip_ids, seed=SEED)
    changed_condition_l2 = float((generated.pixels[0] - other_condition.pixels[0]).norm())
    changed_seed_l2 = float((generated.pixels - other_seed.pixels).norm())
    packed = model.condition_input(fused)
    context = model.condition(fused)
    one_fused = replace(
        fused, fused_embedding=fused.fused_embedding[:1], modality_mask=fused.modality_mask[:1],
        has_information=fused.has_information[:1], text_truncated=fused.text_truncated[:1],
    )
    single = model(fixture[:1], one_fused, noise=torch.zeros(1, 32))
    empty = replace(
        fused, fused_embedding=fused.fused_embedding[:0], modality_mask=fused.modality_mask[:0],
        has_information=fused.has_information[:0], text_truncated=fused.text_truncated[:0],
    )
    ideal = PosteriorOutput(fixture.clone(), torch.zeros(7, 32), torch.zeros(7, 32))
    beta_zero = vae_loss(posterior, fixture, beta=0)
    checks = {
        "cached vision and initial fusion state match M10": True,
        "condition concatenation has 515 float32 channels": packed.shape == (7, 515)
        and packed.dtype == torch.float32,
        "condition fields retain their documented order": torch.equal(packed[:, :512], fused.fused_embedding)
        and torch.equal(packed[:, 512:514], fused.modality_mask.float())
        and torch.equal(packed[:, 514], fused.text_truncated.float()),
        "condition adapter returns finite 128-wide states": context.shape == (7, 128)
        and torch.isfinite(context).all(),
        "posterior mean and log variance have 32 channels": posterior.mean.shape == (7, 32)
        and posterior.log_variance.shape == (7, 32),
        "reconstruction has finite float32 RGB shape and range": posterior.reconstruction.shape == (7, 3, 64, 64)
        and posterior.reconstruction.dtype == torch.float32
        and torch.isfinite(posterior.reconstruction).all()
        and (posterior.reconstruction >= 0).all() and (posterior.reconstruction <= 1).all(),
        "prior generation has finite RGB shape and range": generated.pixels.shape == (7, 3, 64, 64)
        and torch.isfinite(generated.pixels).all()
        and (generated.pixels >= 0).all() and (generated.pixels <= 1).all(),
        "same seed reproduces generated tensors": torch.equal(generated.pixels, repeated.pixels),
        "different latent samples reach the output": changed_seed_l2 > 0,
        "a text change reaches the decoder with fixed noise": changed_condition_l2 > 0,
        "output preserves source IDs flags seed and review requirement": generated.clip_ids == tuple(clip_ids)
        and torch.equal(generated.modality_mask, fused.modality_mask)
        and torch.equal(generated.text_truncated, fused.text_truncated)
        and generated.seed == SEED and generated.requires_trainer_review,
        "vision-only and text-only inputs are supported": model.sample(no_text, clip_ids=clip_ids, seed=SEED).pixels.shape
        == model.sample(no_vision, clip_ids=clip_ids, seed=SEED).pixels.shape == (7, 3, 64, 64),
        "a row with both modalities absent is rejected": rejected(lambda: model.condition(neither)),
        "wrong fusion versions and widths are rejected": rejected(lambda: model.condition(replace(fused, model_version="wrong")))
        and rejected(lambda: model.condition(replace(fused, fused_embedding=fused.fused_embedding[:, :256]))),
        "nonfinite fusion features and inconsistent availability are rejected": rejected(
            lambda: model.condition(replace(fused, fused_embedding=torch.full((7, 512), float("nan"))))
        ) and rejected(lambda: model.condition(replace(fused, has_information=torch.zeros(7, dtype=torch.bool)))),
        "nonzero absent feature blocks are rejected": rejected(lambda: model.condition(
            replace(fused, modality_mask=torch.tensor([[True, False]] * 7))
        )),
        "wrong image sizes and numeric ranges are rejected": rejected(lambda: model(fixture[:, :, :32], fused))
        and rejected(lambda: model(fixture + 1, fused)),
        "nonfinite images and wrong dtypes are rejected": rejected(lambda: model(torch.full_like(fixture, float("nan")), fused))
        and rejected(lambda: model(fixture.double(), fused)),
        "wrong latent and noise dimensions are rejected": rejected(lambda: model.decode(torch.zeros(7, 16), context))
        and rejected(lambda: model(fixture, fused, noise=torch.zeros(7, 16))),
        "empty batches and missing clip IDs are rejected": rejected(lambda: model.condition(empty))
        and rejected(lambda: model.sample(fused, clip_ids=clip_ids[:1], seed=SEED)),
        "deterministic single and batch reconstructions agree": torch.allclose(
            single.reconstruction[0], posterior.reconstruction[0], rtol=1e-5, atol=1e-6
        ),
        "Gaussian and KL losses are finite and nonnegative": all(torch.isfinite(value) and value >= 0 for value in losses.values()),
        "ideal reconstruction and standard-normal posterior have zero loss": vae_loss(ideal, fixture)["total"] == 0,
        "KL weighting follows the specified loss": torch.allclose(losses["total"], losses["reconstruction"] + losses["kl"])
        and torch.equal(beta_zero["total"], losses["reconstruction"]),
        "negative and nonfinite KL coefficients are rejected": rejected(lambda: vae_loss(posterior, fixture, beta=-1))
        and rejected(lambda: vae_loss(posterior, fixture, beta=float("nan"))),
    }

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    gradients = {}
    for name in ("condition_adapter", "image_encoder", "posterior", "mean_head", "log_variance_head", "decoder"):
        parameters = tuple(getattr(model, name).parameters())
        gradients[name] = float(torch.cat([parameter.grad.flatten() for parameter in parameters]).norm())
    checks["all six generative parameter groups receive finite nonzero gradients"] = all(
        np.isfinite(value) and value > 0 for value in gradients.values()
    )
    optimizer.step()
    checks["a temporary optimizer step changes the VAE state"] = parameter_digest(model) != initial_digest
    model.load_state_dict(initial)
    checks["the untrained VAE state is restored"] = parameter_digest(model) == initial_digest
    checks["language and fusion weights stay fixed without gradients"] = (
        parameter_digest(encoder.model) == text_before and parameter_digest(fusion) == fusion_before
        and all(parameter.grad is None for parameter in encoder.model.parameters())
        and all(parameter.grad is None for parameter in fusion.parameters())
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "generator_initial_state.safetensors"
    save_file(model.state_dict(), str(checkpoint))
    model.load_state_dict(load_file(str(checkpoint)))
    checks["checkpoint reload reproduces prior generation"] = torch.equal(
        generated.pixels, model.sample(fused, clip_ids=clip_ids, seed=SEED).pixels
    )
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise AssertionError({name: passed for name, passed in checks.items() if not passed})
    result = {
        "model_version": VERSION, "seed": SEED, "trained_on_paired_examples": False,
        "image_fixture": "Analytic RGB coordinate grid for tensor and gradient checks only",
        "quality_evaluation": "Not performed. Initialized outputs are not exercise illustrations.",
        "source_pairs": "Seven M10 held-out vision vectors with fictional notes, used only for interface checks",
        "vision_archive_sha256": EMBEDDINGS_SHA256, "fusion_state_sha256": fusion_before,
        "text_encoder_state_sha256": text_before, "generator_state_sha256": initial_digest,
        "checkpoint_sha256": file_digest(checkpoint),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_groups": {
            name: sum(parameter.numel() for parameter in getattr(model, name).parameters()) for name in gradients
        },
        "input_shapes": {"fusion": [7, 512], "condition_input": [7, 515], "context": [7, 128], "latent": [7, 32]},
        "output_shape": list(generated.pixels.shape), "checks": checks,
        "gradient_norms": gradients,
        "untrained_output_sensitivity_l2": {"changed_note_same_noise": changed_condition_l2, "changed_noise": changed_seed_l2},
        "fixture_objective": {name: float(value.detach()) for name, value in losses.items()},
        "software": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "safetensors")},
    }
    (output_dir / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Milestone 11 passed {len(checks)} architecture checks. No paired-data training or image-quality evaluation.", flush=True)
    print("Parameters", result["parameters"], "output", result["output_shape"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-embeddings", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.vision_embeddings, args.model_dir, args.output)


if __name__ == "__main__":
    main()
