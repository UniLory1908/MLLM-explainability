from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dashboard.config import DashboardConfig, resolve_project_path
from scripts.dashboard.data_access import load_json, load_word_layer_map, row_paths
from scripts.dashboard.db import connect
from scripts.dashboard.metrics import cosine_similarity, jsd, top_iou
from scripts.dashboard.pairwise import ssim_metric


BASELINE_LABEL = "baseline_neutral"
ALIGNMENT_METHOD = "positional_common_index"

DERIVED_SCHEMA = """
CREATE TABLE IF NOT EXISTS output_diagnostics (
    case_id TEXT PRIMARY KEY,
    image_id INTEGER,
    condition_id TEXT,
    condition_label TEXT,
    prompt_label TEXT,
    run_name TEXT,
    is_official INTEGER,
    metadata_path TEXT,
    image_path TEXT,
    prompt_text TEXT,
    response_text TEXT,
    generated_token_count INTEGER,
    generated_word_count INTEGER,
    response_char_length INTEGER,
    response_word_length INTEGER,
    response_token_length INTEGER,
    max_new_tokens INTEGER,
    special_token_count INTEGER,
    coordinate_token_count INTEGER,
    bbox_style_output_flag INTEGER,
    has_object_ref_tokens INTEGER,
    has_box_tokens INTEGER,
    exact_response_match_vs_baseline INTEGER,
    response_length_delta_vs_baseline INTEGER,
    response_length_ratio_vs_baseline REAL,
    token_overlap_vs_baseline REAL,
    content_jaccard_vs_baseline REAL,
    content_jaccard_distance_vs_baseline REAL,
    first_divergence_word_index INTEGER,
    first_divergence_ratio REAL,
    matched_word_count_vs_baseline INTEGER,
    matched_word_coverage_vs_baseline REAL
);

CREATE TABLE IF NOT EXISTS visual_sensitivity_vs_baseline (
    case_id TEXT PRIMARY KEY,
    image_id INTEGER,
    condition_id TEXT,
    condition_label TEXT,
    prompt_label TEXT,
    baseline_case_id TEXT,
    run_name TEXT,
    is_official INTEGER,
    alignment_method TEXT,
    common_word_count INTEGER,
    compared_map_count INTEGER,
    entropy_mean_delta_vs_baseline REAL,
    entropy_median_delta_vs_baseline REAL,
    top5_mass_mean_delta_vs_baseline REAL,
    effective_area_norm_mean_delta_vs_baseline REAL,
    spread_trace_mean_delta_vs_baseline REAL,
    peak_count_mean_delta_vs_baseline REAL,
    secondary_primary_ratio_mean_delta_vs_baseline REAL,
    multipeak_ratio_delta_vs_baseline REAL,
    mean_centroid_shift_vs_baseline REAL,
    median_centroid_shift_vs_baseline REAL,
    mean_top5_iou_vs_baseline REAL,
    median_top5_iou_vs_baseline REAL,
    mean_cosine_similarity_vs_baseline REAL,
    median_cosine_similarity_vs_baseline REAL,
    mean_jsd_vs_baseline REAL,
    median_jsd_vs_baseline REAL,
    mean_ssim_vs_baseline REAL,
    mean_emd_vs_baseline REAL,
    layer_path_length_mean_delta_vs_baseline REAL,
    layer_max_jump_mean_delta_vs_baseline REAL,
    layer_tortuosity_mean_delta_vs_baseline REAL,
    word_path_length_delta_vs_baseline REAL,
    word_max_jump_delta_vs_baseline REAL,
    word_tortuosity_delta_vs_baseline REAL
);

CREATE TABLE IF NOT EXISTS diagnostic_scores (
    case_id TEXT PRIMARY KEY,
    image_id INTEGER,
    condition_id TEXT,
    condition_label TEXT,
    prompt_label TEXT,
    run_name TEXT,
    is_official INTEGER,
    unstable_explanation_candidate_score REAL,
    prompt_dominated_candidate_score REAL,
    weak_grounding_candidate_score REAL,
    multipeak_ambiguity_score REAL,
    bbox_or_grounding_format_score REAL,
    formula_version TEXT
);

CREATE TABLE IF NOT EXISTS token_category_summary (
    case_id TEXT,
    token_category TEXT,
    token_count INTEGER,
    mean_entropy_norm REAL,
    mean_top5_mass REAL,
    mean_peak_count REAL,
    mean_secondary_primary_ratio REAL,
    mean_layer_path_length REAL,
    mean_layer_tortuosity REAL,
    mean_global_centroid_x_norm REAL,
    mean_global_centroid_y_norm REAL,
    PRIMARY KEY (case_id, token_category)
);

CREATE INDEX IF NOT EXISTS idx_output_diag_image ON output_diagnostics(image_id, condition_label);
CREATE INDEX IF NOT EXISTS idx_visual_sensitivity_image ON visual_sensitivity_vs_baseline(image_id, condition_label);
CREATE INDEX IF NOT EXISTS idx_diag_scores_image ON diagnostic_scores(image_id, condition_label);
CREATE INDEX IF NOT EXISTS idx_token_category_case ON token_category_summary(case_id);
"""


FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into",
    "is", "it", "its", "of", "on", "or", "that", "the", "their", "there", "this",
    "to", "with", "without", "what", "which", "who", "why", "how", "does", "do",
    "can", "could", "would", "should", "please", "describe", "tell", "me",
}
ATTRIBUTE_WORDS = {
    "black", "blue", "brown", "green", "gray", "grey", "red", "white", "yellow",
    "small", "large", "big", "little", "old", "new", "bright", "dark", "wooden",
    "metal", "plastic", "round", "square", "left", "right", "front", "back",
}
OBJECT_REF_WORDS = {"object", "objects", "item", "items", "thing", "things", "person", "people"}
BOX_WORDS = {"box", "bbox", "bounding", "coordinate", "coordinates", "region", "location"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute additive derived dashboard metrics.")
    parser.add_argument("--official-only", action="store_true", help="Use only cases marked is_official=1.")
    parser.add_argument("--case-filter", help="Substring matched against case_id.")
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument(
        "--compute-map-similarities",
        action="store_true",
        help="Compute positional raw-map cosine/top5/JSD aggregation. This can be expensive on large datasets.",
    )
    parser.add_argument("--skip-map-similarities", action="store_true", help="Compatibility flag; keep raw-map similarities skipped.")
    parser.add_argument("--no-ssim", action="store_true", help="Skip SSIM even when map similarities are enabled.")
    return parser


def initialize_derived(conn: sqlite3.Connection) -> None:
    conn.executescript(DERIVED_SCHEMA)
    conn.commit()


def delete_selected_cases(conn: sqlite3.Connection, table: str, case_ids: list[str]) -> None:
    if not case_ids:
        return
    placeholders = ",".join("?" for _ in case_ids)
    conn.execute(f"DELETE FROM {table} WHERE case_id IN ({placeholders})", case_ids)


def load_metadata_for_case(case: sqlite3.Row, config: DashboardConfig) -> dict[str, Any]:
    path = resolve_project_path(case["metadata_path"], config.project_root)
    return load_json(path)


def response_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def content_words(text: str) -> set[str]:
    return {word for word in response_words(text) if word not in FUNCTION_WORDS}


def token_pieces(metadata: dict[str, Any]) -> list[str]:
    pieces = metadata.get("generated_token_pieces")
    if isinstance(pieces, list) and pieces:
        return [str(piece) for piece in pieces]
    labels = metadata.get("generated_token_labels")
    if isinstance(labels, list) and labels:
        return [str(label) for label in labels]
    return [str(step.get("token_piece") or step.get("token_label") or "") for step in metadata.get("step_records") or []]


def output_flags(text: str, pieces: list[str]) -> dict[str, int]:
    combined = " ".join([text, *pieces])
    lowered = combined.lower()
    coord_patterns = [
        r"\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+",
        r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+",
        r"\b(?:x|y|x1|y1|x2|y2)\s*[:=]\s*-?\d",
        r"\b\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\b",
    ]
    coordinate_count = 0
    for piece in pieces:
        if any(re.search(pattern, piece) for pattern in coord_patterns) or re.fullmatch(r"[-+]?\d+(?:\.\d+)?", piece.strip()):
            coordinate_count += 1
    special_count = sum(1 for piece in pieces if re.search(r"<\|.*?\|>|<[^>]+>|\[[A-Z_]+\]", piece))
    has_object = int(any(word in lowered for word in OBJECT_REF_WORDS))
    has_box = int(any(word in lowered for word in BOX_WORDS) or bool(re.search(r"\bbox\b|\bbounding box\b|\[\s*\d", lowered)))
    return {
        "special_token_count": special_count,
        "coordinate_token_count": coordinate_count,
        "bbox_style_output_flag": int(has_box or coordinate_count >= 4),
        "has_object_ref_tokens": has_object,
        "has_box_tokens": has_box,
    }


def lexical_comparison(words: list[str], baseline: list[str]) -> dict[str, Any]:
    matched_prefix = 0
    for left, right in zip(words, baseline):
        if left != right:
            break
        matched_prefix += 1
    first_div = None if matched_prefix == len(words) == len(baseline) else matched_prefix
    word_counter = Counter(words)
    base_counter = Counter(baseline)
    overlap = sum((word_counter & base_counter).values())
    content = {w for w in words if w not in FUNCTION_WORDS}
    base_content = {w for w in baseline if w not in FUNCTION_WORDS}
    union = content | base_content
    jaccard = float(len(content & base_content) / len(union)) if union else 1.0
    return {
        "token_overlap_vs_baseline": float(overlap / max(len(words), len(baseline), 1)),
        "content_jaccard_vs_baseline": jaccard,
        "content_jaccard_distance_vs_baseline": 1.0 - jaccard,
        "first_divergence_word_index": first_div,
        "first_divergence_ratio": None if first_div is None else float(first_div / max(len(words), len(baseline), 1)),
        "matched_word_count_vs_baseline": overlap,
        "matched_word_coverage_vs_baseline": float(overlap / max(len(words), 1)),
    }


def metric_list(conn: sqlite3.Connection, table: str, case_id: str, column: str) -> list[float]:
    return [
        float(row[0])
        for row in conn.execute(f"SELECT {column} FROM {table} WHERE case_id=? AND {column} IS NOT NULL", (case_id,))
    ]


def mean_or_none(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def median_or_none(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def delta_mean(conn: sqlite3.Connection, table: str, case_id: str, baseline_case_id: str, column: str) -> float | None:
    left = mean_or_none(metric_list(conn, table, case_id, column))
    right = mean_or_none(metric_list(conn, table, baseline_case_id, column))
    if left is None or right is None:
        return None
    return float(left - right)


def delta_median(conn: sqlite3.Connection, table: str, case_id: str, baseline_case_id: str, column: str) -> float | None:
    left = median_or_none(metric_list(conn, table, case_id, column))
    right = median_or_none(metric_list(conn, table, baseline_case_id, column))
    if left is None or right is None:
        return None
    return float(left - right)


def multipeak_ratio(conn: sqlite3.Connection, case_id: str) -> float | None:
    row = conn.execute(
        "SELECT AVG(CAST(is_multipeak AS REAL)) FROM map_metrics WHERE case_id=? AND is_multipeak IS NOT NULL",
        (case_id,),
    ).fetchone()
    return None if row is None or row[0] is None else float(row[0])


def centroid_shifts(conn: sqlite3.Connection, case_id: str, baseline_case_id: str) -> list[float]:
    rows = conn.execute(
        """
        SELECT a.global_centroid_x_norm ax, a.global_centroid_y_norm ay,
               b.global_centroid_x_norm bx, b.global_centroid_y_norm by
        FROM map_metrics a
        JOIN map_metrics b ON b.word_index=a.word_index AND b.layer_index=a.layer_index
        WHERE a.case_id=? AND b.case_id=?
          AND a.global_centroid_x_norm IS NOT NULL AND a.global_centroid_y_norm IS NOT NULL
          AND b.global_centroid_x_norm IS NOT NULL AND b.global_centroid_y_norm IS NOT NULL
        """,
        (case_id, baseline_case_id),
    )
    return [float(math.hypot(row["ax"] - row["bx"], row["ay"] - row["by"])) for row in rows]


def map_similarity_summary(
    conn: sqlite3.Connection,
    config: DashboardConfig,
    case_id: str,
    baseline_case_id: str,
    include_ssim: bool,
) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT a.*, b.raw_map_paths_json AS baseline_raw_map_paths_json,
               b.shape_h AS baseline_shape_h, b.shape_w AS baseline_shape_w,
               b.dtype AS baseline_dtype, b.map_exists AS baseline_map_exists,
               b.source_signature AS baseline_source_signature
        FROM maps a
        JOIN maps b ON b.word_index=a.word_index AND b.layer_index=a.layer_index
        WHERE a.case_id=? AND b.case_id=? AND a.map_exists=1 AND b.map_exists=1
        ORDER BY a.word_index, a.layer_index
        """,
        (case_id, baseline_case_id),
    )
    cosines: list[float] = []
    top5s: list[float] = []
    jsds: list[float] = []
    ssims: list[float] = []
    count = 0
    for row in rows:
        baseline_row = {
            "case_id": baseline_case_id,
            "word_index": row["word_index"],
            "layer_index": row["layer_index"],
            "raw_map_paths_json": row["baseline_raw_map_paths_json"],
        }
        arr = load_word_layer_map(row_paths(row, config))
        base = load_word_layer_map(row_paths(baseline_row, config))
        if arr is None or base is None:
            continue
        count += 1
        for target, value in (
            (cosines, cosine_similarity(arr, base)),
            (top5s, top_iou(arr, base, 5.0)),
            (jsds, jsd(arr, base)),
        ):
            if value is not None:
                target.append(float(value))
        if include_ssim:
            value = ssim_metric(arr, base)
            if value is not None:
                ssims.append(float(value))
    return {
        "compared_map_count": count,
        "mean_top5_iou_vs_baseline": mean_or_none(top5s),
        "median_top5_iou_vs_baseline": median_or_none(top5s),
        "mean_cosine_similarity_vs_baseline": mean_or_none(cosines),
        "median_cosine_similarity_vs_baseline": median_or_none(cosines),
        "mean_jsd_vs_baseline": mean_or_none(jsds),
        "median_jsd_vs_baseline": median_or_none(jsds),
        "mean_ssim_vs_baseline": mean_or_none(ssims),
        "mean_emd_vs_baseline": None,
    }


def common_word_count(conn: sqlite3.Connection, case_id: str, baseline_case_id: str) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) FROM words a
        JOIN words b ON b.word_index=a.word_index
        WHERE a.case_id=? AND b.case_id=?
        """,
        (case_id, baseline_case_id),
    ).fetchone()
    return int(row[0] or 0)


def precompute_output_diagnostics(conn: sqlite3.Connection, config: DashboardConfig, cases: list[sqlite3.Row], baselines: dict[int, sqlite3.Row]) -> None:
    delete_selected_cases(conn, "output_diagnostics", [case["case_id"] for case in cases])
    metadata_by_case = {case["case_id"]: load_metadata_for_case(case, config) for case in cases}
    words_by_case = {case["case_id"]: response_words(str(metadata_by_case[case["case_id"]].get("response_text") or "")) for case in cases}
    for case in cases:
        metadata = metadata_by_case[case["case_id"]]
        text = str(metadata.get("response_text") or "")
        prompt = str(metadata.get("prompt_text") or "")
        pieces = token_pieces(metadata)
        flags = output_flags(text, pieces)
        words = words_by_case[case["case_id"]]
        baseline = baselines[int(case["image_id"])]
        baseline_metadata = metadata_by_case.get(baseline["case_id"]) or load_metadata_for_case(baseline, config)
        baseline_response = str(baseline_metadata.get("response_text") or "")
        base_words = words_by_case.get(baseline["case_id"]) or response_words(baseline_response)
        lexical = lexical_comparison(words, base_words)
        base_len = len(baseline_response)
        length_delta = len(text) - base_len
        conn.execute(
            """
            INSERT OR REPLACE INTO output_diagnostics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                case["case_id"], case["image_id"], case["condition_id"], case["condition_label"], case["prompt_label"],
                case["run_name"], case["is_official"], case["metadata_path"], case["image_path"], prompt, text,
                len(metadata.get("generated_token_ids") or pieces), case["word_count"], len(text), len(words),
                len(pieces) if pieces else None, metadata.get("max_new_tokens"), flags["special_token_count"],
                flags["coordinate_token_count"], flags["bbox_style_output_flag"], flags["has_object_ref_tokens"],
                flags["has_box_tokens"], int(text == baseline_response), length_delta,
                float(len(text) / base_len) if base_len else None, lexical["token_overlap_vs_baseline"],
                lexical["content_jaccard_vs_baseline"], lexical["content_jaccard_distance_vs_baseline"],
                lexical["first_divergence_word_index"], lexical["first_divergence_ratio"],
                lexical["matched_word_count_vs_baseline"], lexical["matched_word_coverage_vs_baseline"],
            ),
        )


def precompute_visual_sensitivity(
    conn: sqlite3.Connection,
    config: DashboardConfig,
    cases: list[sqlite3.Row],
    baselines: dict[int, sqlite3.Row],
    compute_map_similarities: bool,
    include_ssim: bool,
) -> None:
    delete_selected_cases(conn, "visual_sensitivity_vs_baseline", [case["case_id"] for case in cases])
    for idx, case in enumerate(cases, start=1):
        baseline = baselines[int(case["image_id"])]
        case_id = case["case_id"]
        base_id = baseline["case_id"]
        is_baseline = case_id == base_id
        shifts = centroid_shifts(conn, case_id, base_id)
        multipeak_delta = None
        left_multi = multipeak_ratio(conn, case_id)
        right_multi = multipeak_ratio(conn, base_id)
        if left_multi is not None and right_multi is not None:
            multipeak_delta = float(left_multi - right_multi)
        sim = {
            "compared_map_count": 0,
            "mean_top5_iou_vs_baseline": 1.0 if is_baseline else None,
            "median_top5_iou_vs_baseline": 1.0 if is_baseline else None,
            "mean_cosine_similarity_vs_baseline": 1.0 if is_baseline else None,
            "median_cosine_similarity_vs_baseline": 1.0 if is_baseline else None,
            "mean_jsd_vs_baseline": 0.0 if is_baseline else None,
            "median_jsd_vs_baseline": 0.0 if is_baseline else None,
            "mean_ssim_vs_baseline": 1.0 if is_baseline else None,
            "mean_emd_vs_baseline": None,
        }
        if compute_map_similarities and not is_baseline:
            print(f"[derived] map similarities {idx}/{len(cases)} {case_id} vs {base_id}", flush=True)
            sim = map_similarity_summary(conn, config, case_id, base_id, include_ssim=include_ssim)
        elif is_baseline:
            row = conn.execute("SELECT COUNT(*) FROM maps WHERE case_id=? AND map_exists=1", (case_id,)).fetchone()
            sim["compared_map_count"] = int(row[0] or 0)
        conn.execute(
            """
            INSERT OR REPLACE INTO visual_sensitivity_vs_baseline VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                case_id, case["image_id"], case["condition_id"], case["condition_label"], case["prompt_label"],
                base_id, case["run_name"], case["is_official"], ALIGNMENT_METHOD,
                common_word_count(conn, case_id, base_id), sim["compared_map_count"],
                0.0 if is_baseline else delta_mean(conn, "map_metrics", case_id, base_id, "entropy_norm"),
                0.0 if is_baseline else delta_median(conn, "map_metrics", case_id, base_id, "entropy_norm"),
                0.0 if is_baseline else delta_mean(conn, "map_metrics", case_id, base_id, "top_5_mass"),
                0.0 if is_baseline else delta_mean(conn, "map_metrics", case_id, base_id, "effective_area_norm"),
                0.0 if is_baseline else delta_mean(conn, "map_metrics", case_id, base_id, "spread_trace"),
                0.0 if is_baseline else delta_mean(conn, "map_metrics", case_id, base_id, "peak_count"),
                0.0 if is_baseline else delta_mean(conn, "map_metrics", case_id, base_id, "secondary_primary_ratio"),
                0.0 if is_baseline else multipeak_delta,
                0.0 if is_baseline else mean_or_none(shifts),
                0.0 if is_baseline else median_or_none(shifts),
                sim["mean_top5_iou_vs_baseline"], sim["median_top5_iou_vs_baseline"],
                sim["mean_cosine_similarity_vs_baseline"], sim["median_cosine_similarity_vs_baseline"],
                sim["mean_jsd_vs_baseline"], sim["median_jsd_vs_baseline"], sim["mean_ssim_vs_baseline"],
                sim["mean_emd_vs_baseline"],
                0.0 if is_baseline else delta_mean(conn, "layer_scanpaths", case_id, base_id, "layer_path_length"),
                0.0 if is_baseline else delta_mean(conn, "layer_scanpaths", case_id, base_id, "layer_max_jump"),
                0.0 if is_baseline else delta_mean(conn, "layer_scanpaths", case_id, base_id, "layer_tortuosity"),
                0.0 if is_baseline else delta_mean(conn, "word_scanpaths", case_id, base_id, "word_path_length"),
                0.0 if is_baseline else delta_mean(conn, "word_scanpaths", case_id, base_id, "word_max_jump"),
                0.0 if is_baseline else delta_mean(conn, "word_scanpaths", case_id, base_id, "word_tortuosity"),
            ),
        )


def clamp01(value: float | None) -> float:
    if value is None or math.isnan(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def scale(value: float | None, high: float, absolute: bool = True) -> float:
    if value is None or high <= 0:
        return 0.0
    v = abs(float(value)) if absolute else float(value)
    return clamp01(v / high)


def avg_score(values: list[float]) -> float:
    return clamp01(float(sum(values) / len(values))) if values else 0.0


def precompute_diagnostic_scores(conn: sqlite3.Connection, cases: list[sqlite3.Row]) -> None:
    delete_selected_cases(conn, "diagnostic_scores", [case["case_id"] for case in cases])
    for case in cases:
        case_id = case["case_id"]
        visual = conn.execute("SELECT * FROM visual_sensitivity_vs_baseline WHERE case_id=?", (case_id,)).fetchone()
        output = conn.execute("SELECT * FROM output_diagnostics WHERE case_id=?", (case_id,)).fetchone()
        entropy = mean_or_none(metric_list(conn, "map_metrics", case_id, "entropy_norm"))
        top5 = mean_or_none(metric_list(conn, "map_metrics", case_id, "top_5_mass"))
        eff_area = mean_or_none(metric_list(conn, "map_metrics", case_id, "effective_area_norm"))
        peak = mean_or_none(metric_list(conn, "map_metrics", case_id, "peak_count"))
        secondary = mean_or_none(metric_list(conn, "map_metrics", case_id, "secondary_primary_ratio"))
        layer_tort = mean_or_none(metric_list(conn, "layer_scanpaths", case_id, "layer_tortuosity"))
        # Formula version v1 keeps all heuristic candidate scores bounded in 0..1.
        # They are ranking proxies: visual drift, response divergence, diffuse maps and bbox-like output raise scores.
        unstable = avg_score([
            1.0 - clamp01(visual["mean_cosine_similarity_vs_baseline"] if visual else None),
            1.0 - clamp01(visual["mean_top5_iou_vs_baseline"] if visual else None),
            scale(visual["mean_centroid_shift_vs_baseline"] if visual else None, 0.5, absolute=False),
            scale(visual["entropy_mean_delta_vs_baseline"] if visual else None, 0.25),
            scale(visual["top5_mass_mean_delta_vs_baseline"] if visual else None, 0.25),
            scale(visual["spread_trace_mean_delta_vs_baseline"] if visual else None, 10000.0),
        ])
        response_divergence = 1.0 - clamp01(output["matched_word_coverage_vs_baseline"] if output else None)
        prompt_dom = avg_score([
            response_divergence,
            clamp01(output["content_jaccard_distance_vs_baseline"] if output else None),
            unstable,
            scale(visual["layer_path_length_mean_delta_vs_baseline"] if visual else None, 1.0),
            scale(visual["word_path_length_delta_vs_baseline"] if visual else None, 1.0),
        ])
        weak_grounding = avg_score([
            clamp01(entropy),
            1.0 - clamp01(top5),
            clamp01(eff_area),
            scale(peak, 5.0, absolute=False),
            clamp01(secondary),
            scale(layer_tort, 10.0, absolute=False),
        ])
        multipeak = avg_score([
            scale(peak, 5.0, absolute=False),
            clamp01(secondary),
            clamp01(multipeak_ratio(conn, case_id)),
        ])
        bbox = avg_score([
            float(output["bbox_style_output_flag"] or 0) if output else 0.0,
            scale(output["coordinate_token_count"] if output else None, 8.0, absolute=False),
            scale(output["special_token_count"] if output else None, 4.0, absolute=False),
            float(output["has_object_ref_tokens"] or 0) if output else 0.0,
            float(output["has_box_tokens"] or 0) if output else 0.0,
        ])
        conn.execute(
            "INSERT OR REPLACE INTO diagnostic_scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                case_id, case["image_id"], case["condition_id"], case["condition_label"], case["prompt_label"],
                case["run_name"], case["is_official"], unstable, prompt_dom, weak_grounding, multipeak, bbox, "v1_proxy_bounded_0_1",
            ),
        )


def categorize_token(token: str) -> str:
    stripped = token.strip()
    lower = stripped.lower()
    if re.search(r"<\|.*?\|>|<[^>]+>|\[[A-Z_]+\]", stripped):
        return "special_token_like"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", stripped) or re.search(r"\d+\s*,\s*\d+", stripped):
        return "coordinate_like" if "," in stripped else "number_like"
    if re.fullmatch(r"[\W_]+", stripped):
        return "punctuation_like"
    if lower in FUNCTION_WORDS:
        return "function_like"
    if lower in ATTRIBUTE_WORDS:
        return "attribute_like"
    return "content_like"


def precompute_token_category_summary(conn: sqlite3.Connection, cases: list[sqlite3.Row]) -> None:
    delete_selected_cases(conn, "token_category_summary", [case["case_id"] for case in cases])
    for case in cases:
        rows = list(conn.execute("SELECT * FROM words WHERE case_id=? ORDER BY word_index", (case["case_id"],)))
        categories: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            categories[categorize_token(row["word_label"])].append(int(row["word_index"]))
        for category, indices in sorted(categories.items()):
            placeholders = ",".join("?" for _ in indices)
            metric_rows = list(
                conn.execute(
                    f"""
                    SELECT entropy_norm, top_5_mass, peak_count, secondary_primary_ratio,
                           global_centroid_x_norm, global_centroid_y_norm
                    FROM map_metrics
                    WHERE case_id=? AND word_index IN ({placeholders})
                    """,
                    (case["case_id"], *indices),
                )
            )
            layer_rows = list(
                conn.execute(
                    f"""
                    SELECT layer_path_length, layer_tortuosity
                    FROM layer_scanpaths
                    WHERE case_id=? AND word_index IN ({placeholders})
                    """,
                    (case["case_id"], *indices),
                )
            )
            def col_mean(source: list[sqlite3.Row], name: str) -> float | None:
                vals = [float(row[name]) for row in source if row[name] is not None]
                return mean_or_none(vals)
            conn.execute(
                "INSERT OR REPLACE INTO token_category_summary VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    case["case_id"], category, len(indices),
                    col_mean(metric_rows, "entropy_norm"), col_mean(metric_rows, "top_5_mass"),
                    col_mean(metric_rows, "peak_count"), col_mean(metric_rows, "secondary_primary_ratio"),
                    col_mean(layer_rows, "layer_path_length"), col_mean(layer_rows, "layer_tortuosity"),
                    col_mean(metric_rows, "global_centroid_x_norm"), col_mean(metric_rows, "global_centroid_y_norm"),
                ),
            )


def selected_cases(conn: sqlite3.Connection, args: argparse.Namespace) -> list[sqlite3.Row]:
    query = "SELECT * FROM cases"
    clauses = []
    params: list[Any] = []
    if args.official_only:
        clauses.append("is_official=1")
    if args.case_filter:
        clauses.append("case_id LIKE ?")
        params.append(f"%{args.case_filter}%")
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY image_id, condition_label, case_id"
    rows = list(conn.execute(query, params))
    if args.case_limit > 0:
        rows = rows[: args.case_limit]
    return rows


def validate_baselines(cases: list[sqlite3.Row]) -> dict[int, sqlite3.Row]:
    by_image: dict[int, list[sqlite3.Row]] = defaultdict(list)
    selected_images = {int(case["image_id"]) for case in cases}
    for case in cases:
        if case["condition_label"] == BASELINE_LABEL:
            by_image[int(case["image_id"])].append(case)
    problems = []
    for image_id in sorted(selected_images):
        count = len(by_image.get(image_id, []))
        if count != 1:
            problems.append(f"image_id={image_id} baseline_count={count}")
    if problems:
        raise RuntimeError("Expected exactly one baseline_neutral per selected image: " + "; ".join(problems))
    return {image_id: rows[0] for image_id, rows in by_image.items()}


def main() -> int:
    args = build_parser().parse_args()
    config = DashboardConfig.default()
    conn = connect(config.db_path)
    initialize_derived(conn)
    cases = selected_cases(conn, args)
    baselines = validate_baselines(cases)
    print(f"[derived] db={config.db_path} cases={len(cases)} official_only={args.official_only}")
    precompute_output_diagnostics(conn, config, cases, baselines)
    conn.commit()
    precompute_visual_sensitivity(
        conn,
        config,
        cases,
        baselines,
        compute_map_similarities=args.compute_map_similarities and not args.skip_map_similarities,
        include_ssim=not args.no_ssim,
    )
    conn.commit()
    precompute_diagnostic_scores(conn, cases)
    precompute_token_category_summary(conn, cases)
    conn.commit()
    for table in ["output_diagnostics", "visual_sensitivity_vs_baseline", "diagnostic_scores", "token_category_summary"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"[derived] {table} rows={count}")
    print("[derived] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
