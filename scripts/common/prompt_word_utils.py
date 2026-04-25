from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoProcessor

WORD_START_MARKER = "\u0120"


def strip_token_piece(piece: str) -> str:
    # Rimuove il marker di spazio iniziale usato da Qwen nei pezzi di token.
    piece = str(piece)
    if piece.startswith(WORD_START_MARKER):
        return piece[1:]
    return piece


def normalize_word_text(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value).strip().lower())


def canonicalize_word_text(value: str) -> str:
    normalized = normalize_word_text(value)
    if len(normalized) > 4 and normalized.endswith("ies"):
        return normalized[:-3] + "y"
    if len(normalized) > 4 and normalized.endswith("es"):
        return normalized[:-2]
    if len(normalized) > 4 and normalized.endswith("s"):
        return normalized[:-1]
    return normalized


def is_special_piece(piece: str) -> bool:
    stripped = strip_token_piece(piece).strip()
    return not stripped or (stripped.startswith("<|") and stripped.endswith("|>"))


def piece_is_word_like(piece: str) -> bool:
    return bool(normalize_word_text(strip_token_piece(piece)))


@lru_cache(maxsize=4)
def load_tokenizer(model_name: str):
    # Nei run storici i pezzi raw del tokenizer possono mancare.
    return AutoProcessor.from_pretrained(model_name).tokenizer


def token_pieces_from_metadata(metadata: dict) -> list[str]:
    token_pieces = metadata.get("generated_token_pieces")
    if isinstance(token_pieces, list) and token_pieces:
        return [str(piece) for piece in token_pieces]

    token_ids = metadata.get("generated_token_ids", [])
    model_name = metadata.get("model_name")
    if not model_name or not token_ids:
        return [str(token) for token in metadata.get("generated_token_labels", [])]

    tokenizer = load_tokenizer(str(model_name))
    return [str(tokenizer.convert_ids_to_tokens(int(token_id))) for token_id in token_ids]


def build_word_groups(step_records: list[dict], token_pieces: list[str] | None = None) -> list[dict]:
    # Raggruppo i token in parole usando i marker del tokenizer.
    # Esempio:
    # - "to", "ilet" -> "toilet"
    # - "the", "Ġtoilet" -> due parole separate
    # Raggruppa i pezzi del tokenizer in parole usando i marker di inizio parola.
    groups: list[dict] = []
    current: list[dict] = []

    def flush_current() -> None:
        nonlocal current
        if not current:
            return
        combined_label = "".join(piece["piece_text"] for piece in current).strip()
        groups.append({
            "word_index": len(groups),
            "word_label": combined_label,
            "canonical_word_label": canonicalize_word_text(combined_label),
            "source_step_indices": [piece["step_idx"] for piece in current],
            "source_token_labels": [piece["token_label"] for piece in current],
            "source_token_pieces": [piece["token_piece"] for piece in current],
            "source_heatmap_paths": [piece["heatmap_path"] for piece in current],
        })
        current = []

    for idx, step in enumerate(step_records):
        token_piece = token_pieces[idx] if token_pieces and idx < len(token_pieces) else step.get("token_piece") or step.get("token_label", "")
        token_piece = str(token_piece)
        piece_text = strip_token_piece(token_piece)
        if is_special_piece(token_piece):
            flush_current()
            continue
        if not piece_is_word_like(token_piece):
            flush_current()
            continue

        new_piece = {
            "step_idx": int(step.get("step_idx", idx)),
            "token_label": str(step.get("token_label", "")),
            "token_piece": token_piece,
            "piece_text": piece_text,
            "heatmap_path": str(step.get("heatmap_path", "")),
        }

        starts_new_word = token_piece.startswith(WORD_START_MARKER)
        if not current:
            current = [new_piece]
            continue
        if starts_new_word:
            flush_current()
            current = [new_piece]
            continue
        current.append(new_piece)

    flush_current()
    return groups


def build_word_lookup(metadata: dict) -> dict[int, dict]:
    word_records = metadata.get("word_records")
    if isinstance(word_records, list) and word_records:
        return {int(word["word_index"]): word for word in word_records}

    step_records = metadata.get("step_records", [])
    token_pieces = token_pieces_from_metadata(metadata)
    words = build_word_groups(step_records, token_pieces)
    return {int(word["word_index"]): word for word in words}


def estimate_heatmap_rgb(overlay_path: str | Path, original_image_path: str | Path) -> np.ndarray:
    overlay = np.asarray(Image.open(overlay_path).convert("RGB"), dtype=np.float32)
    width, height = int(overlay.shape[1]), int(overlay.shape[0])
    raw_image = np.asarray(Image.open(original_image_path).convert("RGB").resize((width, height), Image.BILINEAR), dtype=np.float32)
    return np.clip((2.0 * overlay) - raw_image, 0.0, 255.0)


def combined_word_heatmap(word_record: dict, original_image_path: str | Path) -> np.ndarray | None:
    heatmap_paths = [Path(path) for path in word_record.get("source_heatmap_paths", []) if str(path)]
    valid_paths = [path for path in heatmap_paths if path.exists()]
    if not valid_paths:
        return None
    maps = [estimate_heatmap_rgb(path, original_image_path) for path in valid_paths]
    if not maps:
        return None
    return np.maximum.reduce(maps)
