"""Milestone 9 frozen DistilBERT features with the Milestone 7 note contract."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from transformers import DistilBertModel

from .preprocessing import CLS, SEP, PackedNotes, SEQUENCE_LENGTH, VOCAB_SIZE


MODEL_ID = "distilbert/distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
MODEL_FILE_SHA256 = "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063"
PRETRAINED_WIDTH = 768
PRETRAINED_LAYERS = 6
PRETRAINED_HEADS = 12
VERSION = f"MIPDS-Language-DistilBERT-{MODEL_REVISION[:12]}-mean-v1"


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PretrainedNotes:
    token_states: np.ndarray
    pooled_embedding: np.ndarray
    attention_mask: np.ndarray
    has_text: np.ndarray
    truncated: np.ndarray
    attention: np.ndarray | None
    model_version: str = VERSION


def _validate_notes(notes: PackedNotes):
    shape = notes.input_ids.shape
    if len(shape) != 2 or shape[1] != SEQUENCE_LENGTH:
        raise ValueError("Expected B × 64 packed note IDs.")
    tensors = (notes.input_ids, notes.attention_mask, notes.token_type_ids)
    if any(t.shape != shape or t.dtype != np.int32 for t in tensors):
        raise ValueError("Note IDs, attention mask, and segment IDs must be matching int32 tensors.")
    flags = (notes.has_text, notes.truncated)
    if any(t.shape != (shape[0],) or t.dtype != np.bool_ for t in flags):
        raise ValueError("Presence and truncation flags must be boolean vectors of length B.")
    if np.any(notes.input_ids < 0) or np.any(notes.input_ids >= VOCAB_SIZE):
        raise ValueError("A note ID is outside the pinned vocabulary.")
    if not np.isin(notes.attention_mask, (0, 1)).all():
        raise ValueError("Attention masks must contain 0 or 1.")
    if not np.isin(notes.token_type_ids, (0, 1)).all():
        raise ValueError("Segment IDs must contain 0 or 1.")
    lengths = notes.attention_mask.sum(axis=1)
    expected = np.arange(SEQUENCE_LENGTH)[None, :] < lengths[:, None]
    if not np.array_equal(notes.attention_mask, expected):
        raise ValueError("Notes must be right padded.")
    if not np.array_equal(notes.has_text, lengths > 0):
        raise ValueError("Presence flags disagree with the attention mask.")


class PretrainedNoteEncoder:
    """Expose frozen token states and masked content means in a new 768-wide interface."""

    def __init__(self, source: str | Path = MODEL_ID, *, device: str = "cpu"):
        location = str(source)
        local = Path(location).is_dir()
        if local and file_digest(Path(location) / "model.safetensors") != MODEL_FILE_SHA256:
            raise ValueError("The local weights differ from the pinned upstream checkpoint.")
        options = {} if local else {"revision": MODEL_REVISION}
        self.model, loading = DistilBertModel.from_pretrained(
            location, **options, use_safetensors=True, attn_implementation="eager",
            output_loading_info=True, local_files_only=local,
        )
        if loading["missing_keys"] or loading["mismatched_keys"] or loading["error_msgs"]:
            raise ValueError("The pretrained encoder did not load every expected tensor.")
        expected = {
            "model_type": "distilbert", "dim": PRETRAINED_WIDTH,
            "n_layers": PRETRAINED_LAYERS, "n_heads": PRETRAINED_HEADS,
            "hidden_dim": 3072, "vocab_size": VOCAB_SIZE, "max_position_embeddings": 512,
        }
        if any(getattr(self.model.config, key) != value for key, value in expected.items()):
            raise ValueError("The supplied checkpoint does not match the frozen DistilBERT interface.")
        self.device = torch.device(device)
        self.model.to(self.device, dtype=torch.float32)
        self.model.requires_grad_(False)
        self.model.eval()

    def encode(
        self, notes: PackedNotes, *, batch_size: int = 8, capture_attention: bool = False,
    ) -> PretrainedNotes:
        _validate_notes(notes)
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        count = len(notes.has_text)
        states = np.zeros((count, SEQUENCE_LENGTH, PRETRAINED_WIDTH), dtype=np.float32)
        pooled = np.zeros((count, PRETRAINED_WIDTH), dtype=np.float32)
        maps = None
        if capture_attention:
            maps = np.zeros(
                (PRETRAINED_LAYERS, count, PRETRAINED_HEADS, SEQUENCE_LENGTH, SEQUENCE_LENGTH),
                dtype=np.float32,
            )
        self.model.eval()
        active_rows = np.flatnonzero(notes.has_text)
        with torch.inference_mode():
            for start in range(0, len(active_rows), batch_size):
                rows = active_rows[start:start + batch_size]
                input_ids = torch.as_tensor(notes.input_ids[rows], device=self.device, dtype=torch.long)
                mask = torch.as_tensor(notes.attention_mask[rows], device=self.device, dtype=torch.long)
                output = self.model(
                    input_ids=input_ids, attention_mask=mask,
                    output_attentions=capture_attention, return_dict=True,
                )
                encoded = output.last_hidden_state * mask[:, :, None]
                content = (mask == 1) & (input_ids != CLS) & (input_ids != SEP)
                mean = (encoded * content[:, :, None]).sum(dim=1) / content.sum(dim=1).clamp_min(1)[:, None]
                states[rows] = encoded.cpu().numpy()
                pooled[rows] = mean.cpu().numpy()
                if maps is not None:
                    attention = torch.stack(output.attentions)
                    attention *= mask[None, :, None, :, None] * mask[None, :, None, None, :]
                    maps[:, rows] = attention.cpu().numpy()
        return PretrainedNotes(
            states, pooled, notes.attention_mask.copy(), notes.has_text.copy(),
            notes.truncated.copy(), maps,
        )
