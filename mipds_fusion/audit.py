"""Run Milestone 10 integration on cached M6 vision features and fictional notes."""

import argparse
import csv
from dataclasses import replace
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file, save_file
import torch

from mipds_language.preprocessing import NoteTokenizer
from mipds_language.pretrained import PretrainedNoteEncoder, file_digest
from .fusion import FUSION_VERSION, VISION_VERSION, ProjectedConcatFusion


EMBEDDINGS_SHA256 = "6d3ed8e9fa1a878b3f79472c8ef20b6cfee62975fd1437d7018860161236333f"
SEED = 5352
REPO = Path(__file__).resolve().parents[1]
NOTES = (
    "Dumbbell side bend, 25 lb for 10. My left side felt steadier.",
    "Bent-over row, 35 lb for 8. I changed the bench angle.",
    "Bicep curl, 15 lb for 9. I shortened the last rep.",
    "Push-up, 12 reps. My feet stayed in the same place.",
    "Shoulder press, 20 lb for 8. I used an upright backrest.",
    "Squat, 45 lb for 6. The camera was on the right and I moved left on the last rep.",
    "Triceps extension, 30 lb for 10. " + "I filmed the next set from the same spot. " * 8 +
    "I kept my upper arm still through the final rep.",
)


def parameter_digest(model):
    result = sha256()
    for name, value in model.state_dict().items():
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def _rejected(action):
    try:
        action()
    except ValueError:
        return True
    return False


def run(embeddings_path, model_dir, output_dir):
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    source_hash = file_digest(embeddings_path)
    if source_hash != EMBEDDINGS_SHA256:
        raise ValueError("Use the recorded cached Milestone 6 embedding export.")
    manifest_path = REPO / "experiments/m4_recovered/classification_manifest.csv"
    m4 = json.loads((REPO / "experiments/m4_recovered/m4_run_record.json").read_text())
    if file_digest(manifest_path) != m4["classification_manifest_sha256"]:
        raise ValueError("The split manifest differs from the recovered experiment.")
    with manifest_path.open(newline="") as stream:
        manifest = {row["frame_id"]: row for row in csv.DictReader(stream)}
    with np.load(embeddings_path, allow_pickle=False) as archive:
        frame_ids = archive["frame_ids"].copy()
        embeddings = archive["embeddings"].copy()
        true_indices = archive["true_indices"].copy()
    expected_ids = sorted(key for key, row in manifest.items() if row["split"] == "test")
    if frame_ids.tolist() != expected_ids or embeddings.shape != (570, 2048):
        raise ValueError("The cached vectors do not match the full held-out split.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Cached vision features must be finite.")
    labels = m4["classes"]
    if any(manifest[frame]["label"] != labels[int(label)] for frame, label in zip(frame_ids, true_indices)):
        raise ValueError("Cached label indices differ from the original class order.")

    selected = [int(np.flatnonzero(true_indices == label)[0]) for label in range(len(labels))]
    chosen_frames = frame_ids[selected].tolist()
    clip_ids = [manifest[frame]["video_path"] for frame in chosen_frames]
    vision = torch.from_numpy(embeddings[selected].copy())
    tokenizer = NoteTokenizer(model_dir / "vocab.txt")
    encoder = PretrainedNoteEncoder(model_dir)
    frozen_text_before = parameter_digest(encoder.model)
    language = encoder.encode(tokenizer.pack(NOTES))
    torch.manual_seed(SEED)
    model = ProjectedConcatFusion().eval()
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    original_model_digest = parameter_digest(model)

    def fuse(values=vision, text=language, **kwargs):
        return model(values, text, vision_clip_ids=clip_ids, note_clip_ids=clip_ids, **kwargs)

    output = fuse()
    repeated = fuse()
    missing_text = replace(language, has_text=np.zeros(len(NOTES), dtype=bool))
    no_text = fuse(text=missing_text)
    no_vision = fuse(vision_present=torch.zeros(len(NOTES), dtype=torch.bool))
    neither = fuse(text=missing_text, vision_present=torch.zeros(len(NOTES), dtype=torch.bool))
    absent_nan = fuse(
        values=torch.full_like(vision, float("nan")),
        text=replace(missing_text, pooled_embedding=np.full_like(language.pooled_embedding, np.nan)),
        vision_present=torch.zeros(len(NOTES), dtype=torch.bool),
    )
    changed_note = list(NOTES)
    changed_note[0] = changed_note[0].replace("left", "right")
    altered_language = encoder.encode(tokenizer.pack(changed_note))
    altered_text = fuse(text=altered_language)
    perturbed_vision = vision.clone()
    perturbed_vision[0, 0] += 0.1
    altered_image = fuse(values=perturbed_vision)
    text_change = float(torch.linalg.vector_norm(altered_text.fused_embedding[0] - output.fused_embedding[0]).detach())
    vision_change = float(torch.linalg.vector_norm(altered_image.fused_embedding[0] - output.fused_embedding[0]).detach())

    one = replace(
        language, token_states=language.token_states[:1], pooled_embedding=language.pooled_embedding[:1],
        attention_mask=language.attention_mask[:1], has_text=language.has_text[:1], truncated=language.truncated[:1],
    )
    one_output = model(vision[:1], one, vision_clip_ids=clip_ids[:1], note_clip_ids=clip_ids[:1])
    empty = replace(
        language, token_states=language.token_states[:0], pooled_embedding=language.pooled_embedding[:0],
        attention_mask=language.attention_mask[:0], has_text=language.has_text[:0], truncated=language.truncated[:0],
    )
    empty_output = model(vision[:0], empty, vision_clip_ids=[], note_clip_ids=[])

    checks = {
        "cached vision archive and complete test membership match": True,
        "actual language features have the expected width": language.pooled_embedding.shape == (7, 768),
        "joint features have shape 7 by 512 and float32": output.fused_embedding.shape == (7, 512)
        and output.fused_embedding.dtype == torch.float32,
        "both-present flags are true and output is finite": output.modality_mask.all()
        and torch.isfinite(output.fused_embedding).all(),
        "identical inputs produce identical outputs": torch.equal(output.fused_embedding, repeated.fused_embedding),
        "missing text zeros its block and preserves vision": not no_text.fused_embedding[:, 256:].any()
        and torch.equal(no_text.fused_embedding[:, :256], output.fused_embedding[:, :256]),
        "missing vision zeros its block and preserves text": not no_vision.fused_embedding[:, :256].any()
        and torch.equal(no_vision.fused_embedding[:, 256:], output.fused_embedding[:, 256:]),
        "both missing produces zero output and false availability": not neither.fused_embedding.any()
        and not neither.has_information.any(),
        "nonfinite placeholders in absent modalities are masked": torch.isfinite(absent_nan.fused_embedding).all()
        and not absent_nan.fused_embedding.any(),
        "a present nonfinite vector is rejected": _rejected(lambda: fuse(values=torch.full_like(vision, float("nan")))),
        "incorrect vision dimensions are rejected": _rejected(lambda: fuse(values=vision[:, :768])),
        "legacy 256-wide language features are rejected": _rejected(
            lambda: fuse(text=replace(language, pooled_embedding=language.pooled_embedding[:, :256]))
        ),
        "mismatched clip IDs are rejected": _rejected(
            lambda: model(vision, language, vision_clip_ids=clip_ids, note_clip_ids=clip_ids[::-1])
        ),
        "wrong model versions are rejected": _rejected(lambda: fuse(vision_version="different"))
        and _rejected(lambda: fuse(text=replace(language, model_version="different"))),
        "text truncation flags survive the handoff": torch.equal(output.text_truncated, torch.as_tensor(language.truncated))
        and language.truncated[-1],
        "a word change reaches only the text block": text_change > 0
        and torch.equal(altered_text.fused_embedding[:, :256], output.fused_embedding[:, :256]),
        "a vision perturbation reaches only the vision block": vision_change > 0
        and torch.equal(altered_image.fused_embedding[:, 256:], output.fused_embedding[:, 256:]),
        "individual and batched calls agree": torch.allclose(
            one_output.fused_embedding[0], output.fused_embedding[0], rtol=1e-5, atol=1e-6
        ),
        "empty batches preserve the output contract": empty_output.fused_embedding.shape == (0, 512),
    }

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad(set_to_none=True)
    loss = fuse().fused_embedding.square().mean()
    loss.backward()
    gradients = {
        "vision_projection": float(model.vision_projection[0].weight.grad.norm()),
        "text_projection": float(model.text_projection[0].weight.grad.norm()),
    }
    optimizer.step()
    checks["both projection weights receive finite nonzero gradients"] = all(
        np.isfinite(value) and value > 0 for value in gradients.values()
    )
    checks["an optimizer step changes the fusion state"] = parameter_digest(model) != original_model_digest
    model.load_state_dict(initial)
    checks["the original untrained fusion state is restored"] = parameter_digest(model) == original_model_digest
    checks["pretrained language weights and gradients are fixed"] = (
        parameter_digest(encoder.model) == frozen_text_before
        and all(param.grad is None for param in encoder.model.parameters())
    )
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise AssertionError({name: passed for name, passed in checks.items() if not passed})

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "fusion_initial_state.safetensors"
    save_file(model.state_dict(), str(checkpoint))
    model.load_state_dict(load_file(str(checkpoint)))
    checks["saved fusion weights reproduce the forward output"] = torch.equal(
        fuse().fused_embedding, output.fused_embedding
    )
    if not checks["saved fusion weights reproduce the forward output"]:
        raise AssertionError("Checkpoint reload changed the fusion output.")

    result = {
        "fusion_version": FUSION_VERSION, "seed": SEED, "trained_on_paired_examples": False,
        "vision_version": VISION_VERSION, "language_version": language.model_version,
        "vision_archive_sha256": source_hash, "manifest_sha256": file_digest(manifest_path),
        "text_encoder_state_sha256": frozen_text_before,
        "fusion_state_sha256": original_model_digest, "checkpoint_sha256": file_digest(checkpoint),
        "fusion_parameters": sum(param.numel() for param in model.parameters()),
        "batch_size": 7, "vision_input_shape": [7, 2048], "text_input_shape": [7, 768],
        "fused_output_shape": [7, 512], "checks": checks, "projection_gradient_norms": gradients,
        "input_sensitivity_l2": {"word_left_to_right": text_change, "vision_channel_plus_0_1": vision_change},
        "pairs": [
            {"frame_id": frame, "exercise_label": label, "clip_id": clip, "fictional_note": note}
            for frame, label, clip, note in zip(chosen_frames, labels, clip_ids, NOTES)
        ],
        "software": {
            name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "safetensors")
        },
    }
    (output_dir / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Milestone 10 passed {len(checks)} checks using seven real vision vectors and fictional notes.", flush=True)
    print("Fusion parameters", result["fusion_parameters"], "output", result["fused_output_shape"], flush=True)
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
