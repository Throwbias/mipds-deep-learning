"""Verify pinned pretrained features and a head trained on the M8 fictional notes."""

import argparse
from dataclasses import replace
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from .audit import LABELS, SEED, TEST, TRAIN, VALIDATION, examples
from .preprocessing import NoteTokenizer
from .pretrained import (
    MODEL_FILE_SHA256, MODEL_ID, MODEL_REVISION, VERSION, PretrainedNoteEncoder, file_digest,
)


def parameter_digest(model):
    digest = sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_head(train, validation, labels, val_labels):
    """Fit only a linear readout and select its epoch with validation cross-entropy."""
    torch.manual_seed(SEED)
    mean = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), np.float32(1e-6))
    train_x = torch.from_numpy((train - mean) / scale)
    val_x = torch.from_numpy((validation - mean) / scale)
    gold = torch.tensor(labels, dtype=torch.long)
    val_gold = torch.tensor(val_labels, dtype=torch.long)
    head = torch.nn.Linear(train.shape[1], len(LABELS))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=0.01)
    history = []
    best_loss, best_epoch, best = float("inf"), None, None
    for epoch in range(201):
        with torch.no_grad():
            val_scores = head(val_x)
            val_loss = float(torch.nn.functional.cross_entropy(val_scores, val_gold))
            accuracy = float((val_scores.argmax(dim=1) == val_gold).float().mean())
        history.append({"epoch": epoch, "validation_loss": val_loss, "validation_accuracy": accuracy})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best = {name: value.detach().clone() for name, value in head.state_dict().items()}
        if epoch < 200:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(train_x), gold)
            loss.backward()
            optimizer.step()
    head.load_state_dict(best)
    return head.eval(), mean, scale, best_epoch, history


def run(model_dir, output_dir):
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    model_dir = Path(model_dir)
    checkpoint_digest = MODEL_FILE_SHA256
    tokenizer = NoteTokenizer(model_dir / "vocab.txt")
    encoder = PretrainedNoteEncoder(model_dir)
    original_digest = parameter_digest(encoder.model)
    notes = [None, "", "Squat, 35lb for 8. I moved left when I stood up.",
             "No shift on that rep.", " ".join(["squat"] * 74) + " No heel lift on the right."]
    packed = tokenizer.pack(notes)
    result = encoder.encode(packed, batch_size=2, capture_attention=True)
    single = encoder.encode(tokenizer.pack([notes[2]]))
    missing = encoder.encode(tokenizer.pack([None, " "]), capture_attention=True)
    zero = encoder.encode(tokenizer.pack([]), capture_attention=True)
    pads = packed.attention_mask == 0
    ids = packed.input_ids
    content = (packed.attention_mask == 1) & (ids != 101) & (ids != 102)
    content_means = (result.token_states * content[:, :, None]).sum(axis=1) / np.maximum(
        content.sum(axis=1, keepdims=True), 1
    )
    altered_segments = encoder.encode(replace(packed, token_type_ids=1 - packed.token_type_ids))
    repeat = encoder.encode(packed, batch_size=2)

    checks = {
        "official tensor file hash and vocabulary hash matched": True,
        "base encoder has 66362880 parameters": sum(p.numel() for p in encoder.model.parameters()) == 66_362_880,
        "all base parameters frozen": all(not p.requires_grad for p in encoder.model.parameters()),
        "token states and pooled shape": result.token_states.shape == (5, 64, 768)
        and result.pooled_embedding.shape == (5, 768),
        "all returned feature tensors are float32": result.token_states.dtype == np.float32
        and result.pooled_embedding.dtype == np.float32,
        "missing text and padding states are zero": not result.token_states[pads].any()
        and not result.pooled_embedding[:2].any(),
        "pooled vectors equal content-only masked means": np.allclose(
            result.pooled_embedding, content_means, rtol=1e-5, atol=1e-6
        ),
        "attention shape and row normalization": result.attention.shape == (6, 5, 12, 64, 64)
        and np.allclose(result.attention.sum(axis=-1), packed.attention_mask[None, :, None, :], atol=1e-6),
        "missing batch and zero-size batch are finite": np.isfinite(missing.token_states).all()
        and not missing.token_states.any() and not missing.attention.any()
        and zero.token_states.shape == (0, 64, 768),
        "single-note and batched inference agree": np.allclose(
            result.token_states[2], single.token_states[0], rtol=1e-4, atol=2e-5
        ),
        "evaluation produces deterministic repeat features": np.array_equal(result.token_states, repeat.token_states),
        "legacy segment IDs have no effect": np.array_equal(
            result.token_states, altered_segments.token_states
        ),
        "truncation flag survives and all outputs are finite": result.truncated[4]
        and np.isfinite(result.token_states).all() and np.isfinite(result.pooled_embedding).all(),
    }
    groups = [
        examples(name, template)
        for name, template in (("train", TRAIN), ("validation", VALIDATION), ("test", TEST))
    ]
    feature_sets = []
    for group in groups:
        encoded = encoder.encode(tokenizer.pack([row.text for row in group]), batch_size=8)
        feature_sets.append(encoded.pooled_embedding)
    labels = [np.array([row.label for row in group]) for group in groups]
    head, mean, scale, epoch, history = train_head(*feature_sets[:2], *labels[:2])
    with torch.no_grad():
        test_logits = head(torch.from_numpy((feature_sets[2] - mean) / scale)).numpy()
    predicted = test_logits.argmax(axis=1)
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int32)
    np.add.at(confusion, (labels[2], predicted), 1)
    diag = confusion.diagonal()
    f1 = float(np.mean(2 * diag / np.maximum(confusion.sum(axis=0) + confusion.sum(axis=1), 1)))
    checks["encoder weights unchanged after extraction and readout training"] = (
        parameter_digest(encoder.model) == original_digest
    )
    checks["frozen encoder accumulated no gradients"] = all(
        param.grad is None for param in encoder.model.parameters()
    )
    if not all(checks.values()):
        raise AssertionError({key: value for key, value in checks.items() if not value})

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "readout.npz", weight=head.weight.detach().numpy(),
        bias=head.bias.detach().numpy(), feature_mean=mean, feature_scale=scale,
    )
    path = output_dir / "readout.npz"
    with np.load(path, allow_pickle=False) as saved:
        reloaded_scores = (feature_sets[2] - saved["feature_mean"]) / saved["feature_scale"] @ saved["weight"].T + saved["bias"]
    checks["serialized readout reproduces predictions"] = np.array_equal(reloaded_scores.argmax(axis=1), predicted)
    if not checks["serialized readout reproduces predictions"]:
        raise AssertionError("Readout reload changed the predictions.")
    checks = {name: bool(passed) for name, passed in checks.items()}

    evidence = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "model_version": VERSION,
        "weight_file_sha256": checkpoint_digest, "encoder_state_sha256": original_digest,
        "encoder_parameters": sum(param.numel() for param in encoder.model.parameters()),
        "fine_tuned_encoder_parameters": 0, "readout_parameters": sum(p.numel() for p in head.parameters()),
        "readout_sha256": file_digest(path), "software": {
            name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "safetensors")
        },
        "note_lengths": packed.untruncated_length.tolist(), "checks": checks,
        "feature_shapes": {"token_states": [5, 64, 768], "pooled_embedding": [5, 768]},
        "attention_shape": list(result.attention.shape),
        "fictional_readout": {
            "split_sizes": [len(group) for group in groups], "label_order": LABELS,
            "same_notes_as_milestone_8": True, "selected_epoch": epoch,
            "optimizer": "AdamW", "learning_rate": 0.01, "weight_decay": 0.01,
            "maximum_epochs": 200, "seed": SEED,
            "validation": history, "accuracy": float((predicted == labels[2]).mean()),
            "macro_f1": f1, "confusion": confusion.tolist(),
            "prediction_ids": [row.key for row in groups[2]], "predicted_labels": predicted.tolist(),
        },
    }
    (output_dir / "audit.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        f"Verified {len(checks)} checks. Frozen encoder readout: {int(diag.sum())}/{len(predicted)} "
        f"fictional notes, macro-F1 {f1:.4f}. Validation selected epoch {epoch}.",
        flush=True,
    )
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.model_dir, args.output)


if __name__ == "__main__":
    main()
