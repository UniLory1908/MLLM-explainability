from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prompt_word_utils import build_word_groups, canonicalize_word_text, load_saliency_map, token_pieces_from_metadata
from scripts.dashboard.config import DashboardConfig, resolve_project_path, to_project_rel


OFFICIAL_FIX256 = {
    ("000000030213", "baseline_neutral"),
    ("000000393226", "extra_knowledge_context"),
    ("000000555009", "baseline_neutral"),
    ("000000555009", "order_disruption_stress"),
}


@dataclass
class CaseRecord:
    case_id: str
    metadata_path: Path
    prompt_dir: Path
    run_dir: Path
    metadata: dict[str, Any]
    words: list[dict[str, Any]]
    layers: list[int]
    warnings: list[str]
    is_official: bool
    used_fix256: bool


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("_") or "item"


def file_signature(path: Path) -> str:
    if not path.exists():
        return "missing"
    stat = path.stat()
    return f"{to_project_rel(path)}:{stat.st_size}:{int(stat.st_mtime)}"


def source_signature(paths: list[Path]) -> str:
    payload = "|".join(file_signature(path) for path in paths)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def discover_metadata(config: DashboardConfig, case_filter: str | None = None) -> list[Path]:
    if not config.source_root.exists():
        return []
    paths = sorted(config.source_root.rglob("metadata.json"))
    if case_filter:
        needle = case_filter.lower()
        paths = [path for path in paths if needle in path.as_posix().lower()]
    return paths


def infer_image_label(metadata_path: Path, metadata: dict[str, Any]) -> str:
    parent = metadata_path
    for part in parent.parts:
        if re.match(r"^\d{12}_.+", part):
            return part.split("_", 1)[1]
    image_stem = str(metadata.get("image_stem") or metadata.get("img_id") or "")
    return image_stem or "unknown_image"


def infer_run_name(metadata_path: Path) -> str:
    try:
        return metadata_path.parents[1].name
    except IndexError:
        return "unknown_run"


def case_id_from_metadata(metadata_path: Path, metadata: dict[str, Any]) -> str:
    image_stem = str(metadata.get("image_stem") or metadata.get("img_id") or "unknown")
    prompt_label = str(metadata.get("prompt_label") or metadata.get("prompt_id") or metadata_path.parent.name)
    run_name = infer_run_name(metadata_path)
    return safe_id(f"{image_stem}__{prompt_label}__{run_name}")


def is_official_case(metadata_path: Path, metadata: dict[str, Any]) -> bool:
    run_name = infer_run_name(metadata_path)
    prompt_label = str(metadata.get("prompt_label") or "")
    image_stem = str(metadata.get("image_stem") or "").zfill(12)
    if run_name.startswith("v3_wordlevel_gpu_alllayers_fix256_"):
        return (image_stem, prompt_label) in OFFICIAL_FIX256
    if not run_name.startswith("v3_wordlevel_gpu_alllayers_"):
        return False
    if "probe" in run_name:
        return False
    return (image_stem, prompt_label) not in OFFICIAL_FIX256


def map_paths_for_word_layer(metadata: dict[str, Any], word: dict[str, Any], layer: int, prompt_dir: Path) -> list[Path]:
    paths: list[Path] = []
    step_records = metadata.get("step_records") or []
    image_stem = str(metadata.get("image_stem") or Path(str(metadata.get("image_path", ""))).stem)
    for step_index in word.get("source_step_indices", []):
        step_idx = int(step_index)
        path_value = None
        if step_idx < len(step_records) and isinstance(step_records[step_idx], dict):
            layer_maps = step_records[step_idx].get("layer_raw_maps") or {}
            path_value = layer_maps.get(str(layer)) or layer_maps.get(layer)
        if path_value:
            paths.append(Path(str(path_value)))
            continue
        layer_dir = prompt_dir / "raw_maps" / image_stem / f"layer_{layer:03d}"
        matches = sorted(layer_dir.glob(f"step_{step_idx:04d}_*.npy"))
        if matches:
            paths.append(matches[0])
    return paths


def load_word_layer_map(paths: list[str | Path]) -> np.ndarray | None:
    arrays = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists():
            arrays.append(load_saliency_map(path))
    if not arrays:
        return None
    return np.maximum.reduce(arrays).astype(np.float32)


def build_case_record(metadata_path: Path, config: DashboardConfig) -> CaseRecord | None:
    try:
        metadata = load_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return None

    prompt_dir = metadata_path.parent
    run_dir = metadata_path.parents[1] if len(metadata_path.parents) > 1 else prompt_dir
    step_records = metadata.get("step_records") or []
    warnings: list[str] = []
    saved_words = metadata.get("word_records")
    if isinstance(saved_words, list) and saved_words:
        words = saved_words
    else:
        try:
            token_pieces = token_pieces_from_metadata(metadata)
            words = build_word_groups(step_records, token_pieces)
        except Exception as exc:
            warnings.append(f"word_group_failed:{exc}")
            words = []
    layers = [int(layer) for layer in metadata.get("layers") or []]
    if not layers:
        seen = set()
        for step in step_records:
            for key in (step.get("layer_raw_maps") or {}).keys():
                try:
                    seen.add(int(key))
                except (TypeError, ValueError):
                    pass
        layers = sorted(seen)
    if not layers:
        warnings.append("no_layers_found")
    case_id = case_id_from_metadata(metadata_path, metadata)
    used_fix256 = infer_run_name(metadata_path).startswith("v3_wordlevel_gpu_alllayers_fix256_")
    return CaseRecord(
        case_id=case_id,
        metadata_path=metadata_path,
        prompt_dir=prompt_dir,
        run_dir=run_dir,
        metadata=metadata,
        words=words,
        layers=layers,
        warnings=warnings,
        is_official=is_official_case(metadata_path, metadata),
        used_fix256=used_fix256,
    )


def insert_case(conn: sqlite3.Connection, config: DashboardConfig, case: CaseRecord) -> None:
    metadata = case.metadata
    image_id = metadata.get("img_id")
    image_stem = str(metadata.get("image_stem") or (str(image_id).zfill(12) if image_id is not None else ""))
    image_label = infer_image_label(case.metadata_path, metadata)
    prompt_label = str(metadata.get("prompt_label") or metadata.get("prompt_id") or case.prompt_dir.name)
    token_count = len(metadata.get("step_records") or metadata.get("generated_token_ids") or [])
    layer_count = len(case.layers)
    status = "ok" if case.words and case.layers else "incomplete"
    signature = source_signature([case.metadata_path])

    conn.execute(
        """
        INSERT OR REPLACE INTO cases VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )
        """,
        (
            case.case_id,
            int(image_id) if image_id is not None else None,
            image_stem,
            image_label,
            prompt_label,
            prompt_label,
            str(metadata.get("prompt_id") or ""),
            prompt_label,
            infer_run_name(case.metadata_path),
            to_project_rel(case.run_dir, config.project_root),
            to_project_rel(case.prompt_dir, config.project_root),
            to_project_rel(case.metadata_path, config.project_root),
            to_project_rel(metadata.get("image_path", ""), config.project_root),
            str(metadata.get("model_name") or ""),
            1 if metadata.get("all_layers") else 0,
            token_count,
            len(case.words),
            layer_count,
            status,
            1 if case.is_official else 0,
            1 if case.used_fix256 else 0,
            json.dumps(case.warnings),
            signature,
        ),
    )

    conn.execute("DELETE FROM words WHERE case_id=?", (case.case_id,))
    conn.execute("DELETE FROM maps WHERE case_id=?", (case.case_id,))
    for word in case.words:
        word_index = int(word.get("word_index", 0))
        conn.execute(
            """
            INSERT OR REPLACE INTO words VALUES (?,?,?,?,?,?)
            """,
            (
                case.case_id,
                word_index,
                str(word.get("word_label") or ""),
                str(word.get("canonical_word_label") or canonicalize_word_text(str(word.get("word_label") or ""))),
                json.dumps(word.get("source_step_indices") or []),
                json.dumps(word.get("source_token_pieces") or []),
            ),
        )
        for layer in case.layers:
            raw_paths = map_paths_for_word_layer(metadata, word, layer, case.prompt_dir)
            existing = [resolve_project_path(path, config.project_root) for path in raw_paths if resolve_project_path(path, config.project_root).exists()]
            if existing:
                shape_h = int((metadata.get("step_records") or [{}])[0].get("resized_raw_map_shape", [0, 0])[0] or 0)
                shape_w = int((metadata.get("step_records") or [{}])[0].get("resized_raw_map_shape", [0, 0])[1] or 0)
                dtype = "float32"
            else:
                shape_h = shape_w = None
                dtype = ""
            rel_paths = [to_project_rel(path, config.project_root) for path in raw_paths]
            conn.execute(
                """
                INSERT OR REPLACE INTO maps VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    case.case_id,
                    word_index,
                    int(layer),
                    json.dumps(rel_paths),
                    shape_h,
                    shape_w,
                    dtype,
                    1 if existing else 0,
                    source_signature(existing or [case.metadata_path]),
                ),
            )


def get_case(conn: sqlite3.Connection, case_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()


def get_case_words(conn: sqlite3.Connection, case_id: str) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM words WHERE case_id=? ORDER BY word_index", (case_id,)))


def get_map_row(conn: sqlite3.Connection, case_id: str, word_index: int, layer_index: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM maps WHERE case_id=? AND word_index=? AND layer_index=?",
        (case_id, word_index, layer_index),
    ).fetchone()


def row_paths(row: sqlite3.Row, config: DashboardConfig) -> list[Path]:
    values = json.loads(row["raw_map_paths_json"] or "[]")
    return [resolve_project_path(value, config.project_root) for value in values]
