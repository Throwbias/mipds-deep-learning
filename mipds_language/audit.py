"""Train and check an encoder on fictional MIPDS coaching notes.

Run python -m mipds_language.audit --vocab PATH --output DIRECTORY.
Requires the encoder's NumPy dependency plus Autograd 1.8.0 for training.
"""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .encoder import BLOCK_COUNT, HEAD_COUNT, WIDTH, _encode_rows, encode_notes, initialize_encoder
from .preprocessing import CLS, SEP, SEQUENCE_LENGTH, VOCAB_SHA256, NoteTokenizer, normalize_note


LABELS = ("left", "right", "no_shift_reported")
SEED = 5352
EPOCHS = 8
BATCH_SIZE = 12
LEARNING_RATE = 0.0003

TRAIN = (
    (
        "Set 2, 45lb for 8 squats.", "On rack 2, 3×8 at 22.5kg.", "Set 1, 6 reps.",
        "8 reps. Filmed from my right side.", "5 reps. Filmed from my left side.",
        "Bodyweight squat with a pause.", "Set 3 at 65lb, full range.",
        "20kg goblet squat, flat shoes.", "DB reverse lunge, 10 reps.",
        "Used a box under the bench.", "Warm-up set, no added weight.", "2 sets at 35lb today.",
    ),
    (
        "I shifted {side} out of the bottom.",
        "My weight moved {side} near lockout.",
        "I drifted toward my {side} side on the last rep.",
        "My balance felt stable on the {other}. I leaned {side} after the pause.",
        "I didn't lean {other}. I nudged {side} as I stood.",
    ),
    (
        "Weight stayed even on both feet.", "No shift on the last rep.",
        "I did not shift either way.", "I didn't drift {decoy} as I stood. I stayed centered.",
        "My weight stayed even through that set.",
    ),
)
VALIDATION = (
    ("75 lb for 5 reps in the first set.", "Used the rack in the back.", "10kg in each hand.", "Paused the last rep."),
    (
        "I went {side} as I stood up from the bottom.",
        "I didn't move {other}. I ended up on the {side} side on rep 5.",
        "I felt a shift toward my {side} after the pause.",
    ),
    (
        "I didn't go {decoy}. Both sides felt even.", "I stayed centered near the bottom.",
        "No shift after the pause.",
    ),
)
TEST = (
    ("At 35lb, 8 squats.", "We filmed set 2 of 3×8.", "Goblet squat at 22.5kg.",
     "Set 2, 65 lb. The camera was on the left."),
    (
        "I moved {side} when I stood up.",
        "I tipped {side} after leaving the bottom.",
        "I didn't drift {other}. I moved {side} off the box.",
    ),
    (
        "I didn't shift to either side.", "I felt centered even at the bottom.",
        "I stayed steady with no shift after rep 6.",
    ),
)


@dataclass(frozen=True)
class ProbeNote:
    key: str
    text: str
    label: int


def examples(split, templates):
    contexts, directional, neutral = templates
    rows = []
    for label, name in enumerate(LABELS):
        variants = neutral if label == 2 else directional
        for variant_index, variant in enumerate(variants, 1):
            for context_index, context in enumerate(contexts, 1):
                phrase = variant.format(
                    side=name,
                    other=LABELS[1 - label] if label < 2 else "",
                    decoy=LABELS[(variant_index + context_index) % 2],
                )
                rows.append(ProbeNote(f"{split}-{name}-{variant_index}-{context_index}", f"{context} {phrase}", label))
    return rows


def _check_contract(tokenizer, params):
    note = "DB RDL 3×8 at 52.5lbs. Didn't shift left."
    long_note = " ".join(["squat"] * 74) + " No heel lift on the right."
    batch = tokenizer.pack([note, None, "", long_note])
    enc = encode_notes(params, batch, capture_attention=True)
    both = tokenizer.pack([None, " "])
    empty = encode_notes(params, both, capture_attention=True)
    single = encode_notes(params, tokenizer.pack([note]))
    padding = batch.attention_mask == 0
    id_text = [tokenizer.vocabulary[t] for t in tokenizer.tokenize(note)]
    checks = {
        "input tensors right-shaped and int32": all(
            a.shape == (4, SEQUENCE_LENGTH) and a.dtype == np.int32
            for a in (batch.input_ids, batch.attention_mask, batch.token_type_ids)
        ),
        "load notation, negation, and direction tokens": "rd" in id_text and "##l" in id_text and
        all(piece in id_text for piece in ("3", "x", "8", "52", ".", "5", "lbs", "didn", "left")),
        "untruncated input has one separator": batch.input_ids[0, 0] == CLS and
        np.count_nonzero(batch.input_ids[0] == SEP) == 1 and not batch.token_type_ids[0].any(),
        "overflow keeps two separators and final cue": batch.truncated[3] and
        np.count_nonzero(batch.input_ids[3] == SEP) == 2 and
        np.array_equal(batch.input_ids[3, -8:-1], tokenizer.tokenize("No heel lift on the right.")),
        "encoder token, pooled, and attention shapes": enc.token_states.shape == (4, 64, WIDTH) and
        enc.cls_embedding.shape == (4, WIDTH) and enc.attention.shape == (BLOCK_COUNT, 4, HEAD_COUNT, 64, 64),
        "active queries sum to one and inactive queries sum to zero": np.allclose(
            enc.attention.sum(axis=-1), batch.attention_mask[None, :, None, :], atol=1e-6
        ),
        "padding keys get zero attention": not np.any(
            enc.attention * padding[None, :, None, None, :]
        ),
        "padded states and missing outputs are zero": not np.any(enc.token_states[padding]) and
        not np.any(enc.cls_embedding[~batch.has_text]),
        "all-missing batch bypass is finite": np.isfinite(empty.token_states).all() and
        not np.any(empty.token_states) and not np.any(empty.attention),
        "extra batch padding leaves the first note unchanged": np.allclose(
            single.token_states[0], enc.token_states[0], atol=2e-5
        ),
        "vocabulary pad embedding stays zero": np.all(params["word_embedding"][0] == 0),
    }
    return {name: bool(passed) for name, passed in checks.items()}


def _labels(notes):
    return np.array([item.label for item in notes], dtype=np.int32)


def _cross_entropy(logits, gold, xp):
    shifted = logits - xp.max(logits, axis=1, keepdims=True)
    return xp.mean(xp.log(xp.sum(xp.exp(shifted), axis=1)) - shifted[np.arange(len(gold)), gold])


def _classify(params, packed, *, capture_attention=False):
    encoded = encode_notes(params, packed, capture_attention=capture_attention)
    scores = encoded.cls_embedding @ params["probe_weight"] + params["probe_bias"]
    return scores, encoded


def camera_phrase_probe(params, tokenizer, original):
    variant = original.replace("The camera was on the left.", "The camera was on the right.")
    if variant == original:
        raise ValueError("The camera intervention requires the original left-camera phrase.")
    scores, _ = _classify(params, tokenizer.pack([original, variant]))
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return [
        {
            "camera_side": side, "text": text, "prediction": LABELS[int(p.argmax())],
            "probabilities": {label: float(value) for label, value in zip(LABELS, p)},
        }
        for side, text, p in zip(("left", "right"), (original, variant), probabilities)
    ]


def _train(tokenizer, train_notes, val_notes, initial):
    import autograd.numpy as anp
    from autograd import grad
    from autograd.misc.optimizers import adam

    train, val = (tokenizer.pack([r.text for r in rows]) for rows in (train_notes, val_notes))
    train_gold, val_gold = _labels(train_notes), _labels(val_notes)
    rng = np.random.default_rng(SEED + 1)
    batches = [
        shuffled[start:start + BATCH_SIZE]
        for shuffled in (rng.permutation(len(train_gold)) for _ in range(EPOCHS))
        for start in range(0, len(train_gold), BATCH_SIZE)
    ]
    batches_per_epoch = (len(train_gold) + BATCH_SIZE - 1) // BATCH_SIZE

    def loss(params, indices):
        mask = train.attention_mask[indices]
        length = int(mask.sum(axis=1).max())
        states, _ = _encode_rows(
            params, train.input_ids[indices, :length], mask[:, :length],
            train.token_type_ids[indices, :length], xp=anp,
        )
        logits = anp.matmul(states[:, 0], params["probe_weight"]) + params["probe_bias"]
        return _cross_entropy(logits, train_gold[indices], anp)

    gradient = grad(loss)
    history = []
    best = None
    best_loss = float("inf")

    def evaluate_epoch(params, epoch):
        nonlocal best, best_loss
        scores, _ = _classify(params, val)
        val_loss = float(_cross_entropy(scores, val_gold, np))
        val_accuracy = float(np.mean(scores.argmax(axis=1) == val_gold))
        history.append({"epoch": epoch, "loss": val_loss, "accuracy": val_accuracy})
        if val_loss < best_loss:
            best_loss = val_loss
            best = {name: value.astype(np.float32, copy=True) for name, value in params.items()}
        print(f"Epoch {epoch} | validation loss {val_loss:.4f} | accuracy {val_accuracy:.1%}", flush=True)

    def save_at_epoch(params, step, _gradient):
        if step % batches_per_epoch == 0:
            evaluate_epoch(params, step // batches_per_epoch)

    last = adam(
        lambda params, step: gradient(params, batches[step]), initial,
        step_size=LEARNING_RATE, num_iters=len(batches), callback=save_at_epoch,
    )
    evaluate_epoch(last, EPOCHS)
    return best, history


def _save_weights(path, params):
    tmp = NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False)
    scratch = Path(tmp.name)
    try:
        with tmp:
            np.savez_compressed(tmp, **params)
        scratch.replace(path)
    finally:
        scratch.unlink(missing_ok=True)


def run(vocab, output):
    tokenizer = NoteTokenizer(vocab)
    train_notes, val_notes, test_notes = (
        examples(split, templates) for split, templates in (
            ("train", TRAIN), ("validation", VALIDATION), ("test", TEST)
        )
    )
    groups = [{normalize_note(item.text) for item in rows} for rows in (train_notes, val_notes, test_notes)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Probe splits contain duplicate normalized notes.")
    initial = initialize_encoder(SEED)
    rng = np.random.default_rng(SEED + 2)
    initial["probe_weight"] = rng.standard_normal((WIDTH, len(LABELS)), dtype=np.float32) * np.float32(0.02)
    initial["probe_bias"] = np.zeros(len(LABELS), dtype=np.float32)
    trained, history = _train(tokenizer, train_notes, val_notes, initial)
    checks = _check_contract(tokenizer, trained)
    checks["both learned attention projections changed"] = all(
        float(np.linalg.norm(trained[f"block{block}_qkv_weight"] - initial[f"block{block}_qkv_weight"])) > 0
        for block in range(BLOCK_COUNT)
    )
    if not all(checks.values()):
        raise AssertionError({test: passed for test, passed in checks.items() if not passed})

    test_batch = tokenizer.pack([item.text for item in test_notes])
    scores, _ = _classify(trained, test_batch)
    guesses = scores.argmax(axis=1)
    gold = _labels(test_notes)
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int32)
    np.add.at(confusion, (gold, guesses), 1)
    tp = confusion.diagonal()
    macro_f1 = float(np.mean(2 * tp / (confusion.sum(axis=0) + confusion.sum(axis=1))))
    keys = ("test-left-1-1", "test-right-2-2", "test-no_shift_reported-1-3", "test-left-3-4", "test-right-3-4")
    test_by_key = {item.key: item for item in test_notes}
    cases = [test_by_key[key] for key in keys]
    case_batch = tokenizer.pack([item.text for item in cases])
    case_scores, case_encoding = _classify(trained, case_batch, capture_attention=True)
    case_results = []
    for index, item in enumerate(cases):
        length = int(case_batch.attention_mask[index].sum())
        case_results.append({
            "id": item.key, "text": item.text,
            "gold": LABELS[item.label], "prediction": LABELS[int(case_scores[index].argmax())],
            "tokens": [tokenizer.vocabulary[t] for t in case_batch.input_ids[index, :length]],
            "cls_attention": case_encoding.attention[:, index, :, 0, :length].tolist(),
        })

    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "language_encoder.npz"
    _save_weights(checkpoint, trained)
    with np.load(checkpoint, allow_pickle=False) as restored:
        restored_scores, _ = _classify(restored, case_batch)
    checks["checkpoint reload reproduces probe scores"] = bool(np.array_equal(restored_scores, case_scores))
    if not checks["checkpoint reload reproduces probe scores"]:
        raise AssertionError("Loaded checkpoint did not reproduce the audit logits.")
    result = {
        "seed": SEED, "numpy": np.__version__, "vocabulary_sha256": VOCAB_SHA256,
        "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "encoder_parameters": sum(v.size for k, v in trained.items() if not k.startswith("probe_")),
        "probe_parameters": sum(v.size for k, v in trained.items() if k.startswith("probe_")),
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "labels": LABELS, "split_sizes": [len(train_notes), len(val_notes), len(test_notes)],
        "split_digest": sha256("\n".join(row.text for row in [*train_notes, *val_notes, *test_notes]).encode()).hexdigest(),
        "validation": history, "selected_epoch": min(history, key=lambda item: item["loss"])["epoch"],
        "test_accuracy": float(np.mean(guesses == gold)), "test_macro_f1": macro_f1,
        "test_confusion": confusion.tolist(), "checks": checks, "attention_cases": case_results,
        "camera_phrase_swap": camera_phrase_probe(trained, tokenizer, test_by_key["test-right-3-4"].text),
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"Held-out fictional notes: {int(tp.sum())}/{len(gold)} correct. "
        f"Macro-F1 {macro_f1:.4f}. {len(checks)} verification checks passed.",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=Path, required=True, help="Pinned BERT Base Uncased vocab.txt")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the audit and checkpoint")
    args = parser.parse_args()
    run(args.vocab, args.output)


if __name__ == "__main__":
    main()
