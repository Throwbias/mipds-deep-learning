"""A BERT-compatible input, two-block bidirectional Transformer for Milestone 8."""

from dataclasses import dataclass
from math import sqrt

import numpy as np

from .preprocessing import PackedNotes, SEQUENCE_LENGTH, VOCAB_SIZE


WIDTH = 256
HEAD_COUNT = 4
HEAD_WIDTH = WIDTH // HEAD_COUNT
BLOCK_COUNT = 2
FF_WIDTH = 512


@dataclass(frozen=True)
class EncodedNotes:
    token_states: np.ndarray
    cls_embedding: np.ndarray
    attention_mask: np.ndarray
    has_text: np.ndarray
    truncated: np.ndarray
    attention: np.ndarray | None


def initialize_encoder(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def embedding(count):
        return (rng.standard_normal((count, WIDTH), dtype=np.float32) * np.float32(0.02))

    def linear(fan_in, fan_out):
        scale = np.float32(sqrt(6.0 / (fan_in + fan_out)))
        return rng.uniform(-scale, scale, (fan_in, fan_out)).astype(np.float32)

    params = {
        "word_embedding": embedding(VOCAB_SIZE),
        "position_embedding": embedding(SEQUENCE_LENGTH),
        "segment_embedding": embedding(2),
        "final_gain": np.ones(WIDTH, dtype=np.float32),
        "final_bias": np.zeros(WIDTH, dtype=np.float32),
    }
    params["word_embedding"][0] = 0
    for block in range(BLOCK_COUNT):
        prefix = f"block{block}_"
        for name, fan_in, fan_out in (
            ("qkv", WIDTH, 3 * WIDTH), ("out", WIDTH, WIDTH),
            ("ff1", WIDTH, FF_WIDTH), ("ff2", FF_WIDTH, WIDTH),
        ):
            params[prefix + name + "_weight"] = linear(fan_in, fan_out)
            params[prefix + name + "_bias"] = np.zeros(fan_out, dtype=np.float32)
        for name in ("attention", "ff"):
            params[prefix + name + "_gain"] = np.ones(WIDTH, dtype=np.float32)
            params[prefix + name + "_offset"] = np.zeros(WIDTH, dtype=np.float32)
    return params


def _layer_norm(values, gain, offset, xp):
    centered = values - xp.mean(values, axis=-1, keepdims=True)
    variance = xp.mean(centered * centered, axis=-1, keepdims=True)
    return centered / xp.sqrt(variance + np.float32(1e-5)) * gain + offset


def _gelu(values, xp):
    inner = np.float32(sqrt(2 / np.pi)) * (values + np.float32(0.044715) * values ** 3)
    return np.float32(0.5) * values * (1 + xp.tanh(inner))


def _encode_rows(params, ids, mask, segments, *, xp=np, capture_attention=False):
    """Forward path shared by the CPU audit and inference, for rows that contain text."""
    batch, length = ids.shape
    active = mask[:, :, None].astype(np.float32)
    key_mask = mask[:, None, None, :] == 1
    query_mask = mask[:, None, :, None].astype(np.float32)
    states = (
        params["word_embedding"][ids]
        + params["position_embedding"][None, :length]
        + params["segment_embedding"][segments]
    ) * active
    maps = []
    for block in range(BLOCK_COUNT):
        prefix = f"block{block}_"
        attention_input = _layer_norm(
            states, params[prefix + "attention_gain"], params[prefix + "attention_offset"], xp
        )
        qkv = xp.matmul(attention_input, params[prefix + "qkv_weight"]) + params[prefix + "qkv_bias"]
        heads = xp.transpose(xp.reshape(qkv, (batch, length, 3, HEAD_COUNT, HEAD_WIDTH)), (2, 0, 3, 1, 4))
        queries, keys, values = heads[0], heads[1], heads[2]
        scores = xp.matmul(queries, xp.swapaxes(keys, -2, -1)) / np.float32(sqrt(HEAD_WIDTH))
        scores = xp.where(key_mask, scores, np.float32(-1e9))
        centered_scores = scores - xp.max(scores, axis=-1, keepdims=True)
        exp_scores = xp.exp(centered_scores)
        weights = exp_scores / xp.sum(exp_scores, axis=-1, keepdims=True) * query_mask
        context = xp.matmul(weights, values)
        context = xp.reshape(xp.transpose(context, (0, 2, 1, 3)), (batch, length, WIDTH))
        states = (
            states + xp.matmul(context, params[prefix + "out_weight"]) + params[prefix + "out_bias"]
        ) * active
        ff_input = _layer_norm(states, params[prefix + "ff_gain"], params[prefix + "ff_offset"], xp)
        hidden = _gelu(xp.matmul(ff_input, params[prefix + "ff1_weight"]) + params[prefix + "ff1_bias"], xp)
        states = (states + xp.matmul(hidden, params[prefix + "ff2_weight"]) + params[prefix + "ff2_bias"]) * active
        if capture_attention:
            maps.append(weights)
    return _layer_norm(states, params["final_gain"], params["final_bias"], xp) * active, maps


def encode_notes(params, notes: PackedNotes, *, capture_attention=False) -> EncodedNotes:
    """Return fixed-width token states and CLS embeddings. Skip missing rows."""
    batch = len(notes.has_text)
    states = np.zeros((batch, SEQUENCE_LENGTH, WIDTH), dtype=np.float32)
    attention = None
    if capture_attention:
        attention = np.zeros(
            (BLOCK_COUNT, batch, HEAD_COUNT, SEQUENCE_LENGTH, SEQUENCE_LENGTH), dtype=np.float32
        )
    active = np.flatnonzero(notes.has_text)
    if active.size:
        mask = notes.attention_mask[active]
        length = int(mask.sum(axis=1).max())
        encoded, maps = _encode_rows(
            params, notes.input_ids[active, :length], mask[:, :length],
            notes.token_type_ids[active, :length], capture_attention=capture_attention,
        )
        states[active, :length] = encoded
        if attention is not None:
            attention[:, active, :, :length, :length] = np.stack(maps)
    return EncodedNotes(states, states[:, 0], notes.attention_mask, notes.has_text, notes.truncated, attention)
