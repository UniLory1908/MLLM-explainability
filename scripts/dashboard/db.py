from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    image_id INTEGER,
    image_stem TEXT,
    image_label TEXT,
    condition_id TEXT,
    condition_label TEXT,
    prompt_id TEXT,
    prompt_label TEXT,
    run_name TEXT,
    run_dir TEXT,
    prompt_dir TEXT,
    metadata_path TEXT,
    image_path TEXT,
    model_name TEXT,
    all_layers INTEGER,
    token_count INTEGER,
    word_count INTEGER,
    layer_count INTEGER,
    status TEXT,
    is_official INTEGER,
    used_fix256 INTEGER,
    warnings_json TEXT,
    source_signature TEXT
);

CREATE TABLE IF NOT EXISTS words (
    case_id TEXT,
    word_index INTEGER,
    word_label TEXT,
    canonical_word_label TEXT,
    source_step_indices_json TEXT,
    source_token_pieces_json TEXT,
    PRIMARY KEY (case_id, word_index)
);

CREATE TABLE IF NOT EXISTS maps (
    case_id TEXT,
    word_index INTEGER,
    layer_index INTEGER,
    raw_map_paths_json TEXT,
    shape_h INTEGER,
    shape_w INTEGER,
    dtype TEXT,
    map_exists INTEGER,
    source_signature TEXT,
    PRIMARY KEY (case_id, word_index, layer_index)
);

CREATE TABLE IF NOT EXISTS map_metrics (
    case_id TEXT,
    word_index INTEGER,
    layer_index INTEGER,
    energy_sum REAL,
    energy_mean REAL,
    min_value REAL,
    max_value REAL,
    std_value REAL,
    nonzero_ratio REAL,
    entropy_norm REAL,
    top_1_mass REAL,
    top_5_mass REAL,
    top_10_mass REAL,
    hhi REAL,
    effective_area REAL,
    effective_area_norm REAL,
    hoyer_sparsity REAL,
    global_centroid_x_px REAL,
    global_centroid_y_px REAL,
    global_centroid_x_norm REAL,
    global_centroid_y_norm REAL,
    spread_trace REAL,
    spread_x REAL,
    spread_y REAL,
    covariance_xy REAL,
    anisotropy REAL,
    peak_count INTEGER,
    primary_region_mass REAL,
    secondary_region_mass REAL,
    secondary_primary_ratio REAL,
    primary_region_centroid_x_px REAL,
    primary_region_centroid_y_px REAL,
    primary_region_centroid_x_norm REAL,
    primary_region_centroid_y_norm REAL,
    secondary_region_centroid_x_norm REAL,
    secondary_region_centroid_y_norm REAL,
    is_multipeak INTEGER,
    PRIMARY KEY (case_id, word_index, layer_index)
);

CREATE TABLE IF NOT EXISTS map_metrics_normalization_variants (
    case_id TEXT,
    word_index INTEGER,
    layer_index INTEGER,
    normalization_mode TEXT,
    energy_sum REAL,
    energy_mean REAL,
    min_value REAL,
    max_value REAL,
    std_value REAL,
    nonzero_ratio REAL,
    entropy_norm REAL,
    top_1_mass REAL,
    top_5_mass REAL,
    top_10_mass REAL,
    hhi REAL,
    effective_area REAL,
    effective_area_norm REAL,
    hoyer_sparsity REAL,
    global_centroid_x_px REAL,
    global_centroid_y_px REAL,
    global_centroid_x_norm REAL,
    global_centroid_y_norm REAL,
    spread_trace REAL,
    spread_x REAL,
    spread_y REAL,
    covariance_xy REAL,
    anisotropy REAL,
    peak_count INTEGER,
    primary_region_mass REAL,
    secondary_region_mass REAL,
    secondary_primary_ratio REAL,
    primary_region_centroid_x_px REAL,
    primary_region_centroid_y_px REAL,
    primary_region_centroid_x_norm REAL,
    primary_region_centroid_y_norm REAL,
    secondary_region_centroid_x_norm REAL,
    secondary_region_centroid_y_norm REAL,
    is_multipeak INTEGER,
    source_signature TEXT,
    PRIMARY KEY (case_id, word_index, layer_index, normalization_mode)
);

CREATE TABLE IF NOT EXISTS regions (
    case_id TEXT,
    word_index INTEGER,
    layer_index INTEGER,
    threshold REAL,
    rank INTEGER,
    mass REAL,
    ratio_to_primary REAL,
    centroid_x_px REAL,
    centroid_y_px REAL,
    centroid_x_norm REAL,
    centroid_y_norm REAL,
    bbox_x0 INTEGER,
    bbox_y0 INTEGER,
    bbox_x1 INTEGER,
    bbox_y1 INTEGER,
    bbox_x0_norm REAL,
    bbox_y0_norm REAL,
    bbox_x1_norm REAL,
    bbox_y1_norm REAL,
    area INTEGER,
    peak_value REAL,
    PRIMARY KEY (case_id, word_index, layer_index, threshold, rank)
);

CREATE TABLE IF NOT EXISTS map_pairs (
    pair_id TEXT PRIMARY KEY,
    case_id_a TEXT,
    word_index_a INTEGER,
    layer_index_a INTEGER,
    case_id_b TEXT,
    word_index_b INTEGER,
    layer_index_b INTEGER,
    comparison_profile TEXT,
    cosine_similarity REAL,
    pearson_correlation REAL,
    ssim REAL,
    jsd REAL,
    emd_2d REAL,
    l1_distance REAL,
    l2_distance REAL,
    top_1_iou REAL,
    top_5_iou REAL,
    top_10_iou REAL,
    hotspot_iou_percentile_90 REAL,
    hotspot_iou_percentile_95 REAL,
    hausdorff_peak_distance REAL,
    argmax_distance REAL,
    global_centroid_shift REAL,
    primary_centroid_shift REAL,
    spread_delta REAL,
    anisotropy_delta REAL,
    radial_profile_distance REAL,
    source_signature TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS layer_scanpaths (
    case_id TEXT,
    word_index INTEGER,
    layer_path_length REAL,
    layer_mean_step REAL,
    layer_max_jump REAL,
    layer_net_displacement REAL,
    layer_tortuosity REAL,
    layer_large_jump_count INTEGER,
    layer_bbox_area REAL,
    adjacent_layer_cosine_mean REAL,
    adjacent_layer_cosine_min REAL,
    adjacent_layer_ssim_mean REAL,
    adjacent_layer_jsd_mean REAL,
    adjacent_layer_emd_mean REAL,
    adjacent_layer_top5_iou_mean REAL,
    early_late_cosine REAL,
    early_late_jsd REAL,
    early_late_centroid_shift REAL,
    early_late_spread_delta REAL,
    peak_count_mean REAL,
    peak_count_max REAL,
    secondary_primary_ratio_mean REAL,
    secondary_primary_ratio_max REAL,
    multipeak_layer_count INTEGER,
    multipeak_layer_ratio REAL,
    PRIMARY KEY (case_id, word_index)
);

CREATE TABLE IF NOT EXISTS word_scanpaths (
    case_id TEXT,
    layer_index INTEGER,
    word_path_length REAL,
    word_mean_step REAL,
    word_max_jump REAL,
    word_net_displacement REAL,
    word_tortuosity REAL,
    word_large_jump_count INTEGER,
    word_bbox_area REAL,
    word_revisit_count INTEGER,
    PRIMARY KEY (case_id, layer_index)
);

CREATE TABLE IF NOT EXISTS condition_pairs (
    image_id INTEGER,
    condition_a TEXT,
    condition_b TEXT,
    aggregation_profile TEXT,
    metrics_json TEXT,
    source_signature TEXT,
    created_at TEXT,
    PRIMARY KEY (image_id, condition_a, condition_b, aggregation_profile)
);

CREATE TABLE IF NOT EXISTS cache_manifest (
    cache_key TEXT PRIMARY KEY,
    artifact_path TEXT,
    artifact_type TEXT,
    params_json TEXT,
    source_paths_json TEXT,
    source_signature TEXT,
    created_at TEXT,
    size_bytes INTEGER,
    last_accessed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_cases_image_condition ON cases(image_id, condition_label);
CREATE INDEX IF NOT EXISTS idx_cases_prompt ON cases(prompt_label);
CREATE INDEX IF NOT EXISTS idx_maps_case_layer ON maps(case_id, layer_index);
CREATE INDEX IF NOT EXISTS idx_metrics_case_layer ON map_metrics(case_id, layer_index);
CREATE INDEX IF NOT EXISTS idx_metric_variants_case_layer ON map_metrics_normalization_variants(case_id, normalization_mode, layer_index);
CREATE INDEX IF NOT EXISTS idx_regions_lookup ON regions(case_id, word_index, layer_index, threshold);
CREATE INDEX IF NOT EXISTS idx_pairs_cases ON map_pairs(case_id_a, case_id_b);
CREATE INDEX IF NOT EXISTS idx_condition_pairs ON condition_pairs(image_id, condition_a, condition_b);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize(conn: sqlite3.Connection, rebuild: bool = False) -> None:
    if rebuild:
        tables = [
            "cache_manifest",
            "condition_pairs",
            "word_scanpaths",
            "layer_scanpaths",
            "map_pairs",
            "regions",
            "map_metrics_normalization_variants",
            "map_metrics",
            "maps",
            "words",
            "cases",
        ]
        for table in tables:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.executescript(SCHEMA)
    conn.commit()
