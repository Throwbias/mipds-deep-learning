"""Pack a note using the frozen Milestone 7 WordPiece vocabulary and layout."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Sequence
import unicodedata

import numpy as np


SEQUENCE_LENGTH = 64
VOCAB_SIZE = 30_522
VOCAB_SHA256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
PAD, UNK, CLS, SEP = 0, 100, 101, 102
HEAD_PIECES, TAIL_PIECES = 31, 30

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
_MULTIPLIER = re.compile(r"(?<=\d)\s*[xX×]\s*(?=\d)")
_LOAD_UNIT = re.compile(r"(?<=\d)(lbs?|kgs?)\b", flags=re.IGNORECASE)
# CJK ranges used by the original BERT BasicTokenizer.
_CJK_RANGES = (
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F), (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF),
    (0xF900, 0xFAFF), (0x2F800, 0x2FA1F),
)


def normalize_note(note: str) -> str:
    normalized = unicodedata.normalize("NFKC", note).translate(_QUOTES)
    normalized = _MULTIPLIER.sub(" x ", normalized)
    normalized = _LOAD_UNIT.sub(r" \1", normalized)
    return " ".join(normalized.split())


def _basic_tokens(text: str) -> list[str]:
    """Apply the uncased BERT cleaning, accent stripping, and punctuation rules."""
    cleaned = []
    for char in text:
        point = ord(char)
        if point in (0, 0xFFFD) or unicodedata.category(char).startswith("C"):
            continue
        if any(start <= point <= end for start, end in _CJK_RANGES):
            cleaned.append(f" {char} ")
        else:
            cleaned.append(char)
    result = []
    for word in "".join(cleaned).split():
        chunk = []
        for char in unicodedata.normalize("NFD", word.lower()):
            category = unicodedata.category(char)
            if category == "Mn":
                continue
            point = ord(char)
            punctuation = (
                33 <= point <= 47 or 58 <= point <= 64 or 91 <= point <= 96
                or 123 <= point <= 126 or category.startswith("P")
            )
            if punctuation:
                if chunk:
                    result.append("".join(chunk))
                    chunk = []
                result.append(char)
            else:
                chunk.append(char)
        if chunk:
            result.append("".join(chunk))
    return result


@dataclass(frozen=True)
class PackedNotes:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    token_type_ids: np.ndarray
    has_text: np.ndarray
    truncated: np.ndarray
    untruncated_length: np.ndarray
    pretrim_unknown_count: np.ndarray


class NoteTokenizer:
    def __init__(self, vocabulary_path: str | Path):
        data = Path(vocabulary_path).read_bytes()
        if sha256(data).hexdigest() != VOCAB_SHA256:
            raise ValueError("Use the pinned BERT Base Uncased vocabulary from Milestone 7.")
        self.vocabulary = tuple(data.decode("utf-8").splitlines())
        self._to_id = {piece: index for index, piece in enumerate(self.vocabulary)}

    def _word_ids(self, word: str) -> list[int]:
        if len(word) > 200:
            return [UNK]
        result = []
        start = 0
        while start < len(word):
            prefix = "" if start == 0 else "##"
            for end in range(len(word), start, -1):
                match = self._to_id.get(prefix + word[start:end])
                if match is not None:
                    result.append(match)
                    start = end
                    break
            else:
                return [UNK]
        return result

    def tokenize(self, note: str) -> list[int]:
        return [piece for word in _basic_tokens(normalize_note(note)) for piece in self._word_ids(word)]

    def pack(self, notes: Sequence[str | None]) -> PackedNotes:
        count = len(notes)
        input_ids = np.zeros((count, SEQUENCE_LENGTH), dtype=np.int32)
        masks = np.zeros_like(input_ids)
        segments = np.zeros_like(input_ids)
        has_text = np.zeros(count, dtype=bool)
        truncated = np.zeros(count, dtype=bool)
        lengths = np.zeros(count, dtype=np.int32)
        unknowns = np.zeros_like(lengths)

        for index, note in enumerate(notes):
            pieces = self.tokenize(note) if note is not None else []
            if not pieces:
                continue
            has_text[index] = True
            lengths[index] = len(pieces) + 2
            unknowns[index] = pieces.count(UNK)
            if lengths[index] > SEQUENCE_LENGTH:
                truncated[index] = True
                packed = [CLS, *pieces[:HEAD_PIECES], SEP, *pieces[-TAIL_PIECES:], SEP]
                segments[index, HEAD_PIECES + 2:] = 1
            else:
                packed = [CLS, *pieces, SEP]
            input_ids[index, :len(packed)] = packed
            masks[index, :len(packed)] = 1

        return PackedNotes(input_ids, masks, segments, has_text, truncated, lengths, unknowns)
