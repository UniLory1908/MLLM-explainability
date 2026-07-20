from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dashboard.config import DashboardConfig
from scripts.dashboard.data_access import load_word_layer_map, row_paths
from scripts.dashboard.db import connect, initialize
from scripts.dashboard.metrics import cosine_similarity, extract_regions, jsd, path_metrics, per_map_metrics, top_iou


REGION_THRESHOLDS = [0.70, 0.80, 0.90, 0.95, 0.98]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute lightweight dashboard metrics.")
    parser.add_argument("--case-filter", help="Substring matched against case_id.")
    parser.add_argument("--official-only", action="store_true")
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--init-db", action="store_true", help="Create schema if needed.")
    return parser


def insert_map_metrics(conn, case_id: str, word_index: int, layer_index: int, metrics: dict) -> None:
    columns = [
        "case_id",
        "word_index",
        "layer_index",
        *metrics.keys(),
    ]
    placeholders = ",".join("?" for _ in columns)
    conn.execute(
        f"INSERT OR REPLACE INTO map_metrics ({','.join(columns)}) VALUES ({placeholders})",
        (case_id, word_index, layer_index, *metrics.values()),
    )


def insert_regions(conn, case_id: str, word_index: int, layer_index: int, regions) -> None:
    conn.execute(
        "DELETE FROM regions WHERE case_id=? AND word_index=? AND layer_index=?",
        (case_id, word_index, layer_index),
    )
    for region in regions:
        conn.execute(
            """
            INSERT OR REPLACE INTO regions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                case_id,
                word_index,
                layer_index,
                region.threshold,
                region.rank,
                region.mass,
                region.ratio_to_primary,
                region.centroid_x_px,
                region.centroid_y_px,
                region.centroid_x_norm,
                region.centroid_y_norm,
                region.bbox_x0,
                region.bbox_y0,
                region.bbox_x1,
                region.bbox_y1,
                region.bbox_x0_norm,
                region.bbox_y0_norm,
                region.bbox_x1_norm,
                region.bbox_y1_norm,
                region.area,
                region.peak_value,
            ),
        )


def precompute_case(conn, config: DashboardConfig, case_id: str) -> None:
    maps = list(conn.execute("SELECT * FROM maps WHERE case_id=? AND map_exists=1 ORDER BY word_index, layer_index", (case_id,)))
    arrays: dict[tuple[int, int], np.ndarray] = {}
    skipped_maps = 0
    for row in maps:
        word_index = int(row["word_index"])
        layer_index = int(row["layer_index"])
        try:
            arr = load_word_layer_map(row_paths(row, config))
        except (OSError, ValueError) as exc:
            skipped_maps += 1
            print(
                f"[warn] skip unreadable map case={case_id} word_index={word_index} "
                f"layer_index={layer_index}: {exc}",
                flush=True,
            )
            continue
        if arr is None:
            continue
        arrays[(word_index, layer_index)] = arr
        all_regions = []
        regions_90 = []
        for threshold in REGION_THRESHOLDS:
            regions = extract_regions(arr, threshold=threshold)
            all_regions.extend(regions)
            if abs(threshold - 0.90) < 1e-9:
                regions_90 = regions
        insert_regions(conn, case_id, word_index, layer_index, all_regions)
        insert_map_metrics(conn, case_id, word_index, layer_index, per_map_metrics(arr, regions_90))
    if skipped_maps:
        print(f"[warn] case={case_id} skipped_unreadable_maps={skipped_maps}", flush=True)

    layers = [int(row["layer_index"]) for row in conn.execute("SELECT DISTINCT layer_index FROM maps WHERE case_id=? ORDER BY layer_index", (case_id,))]
    words = [int(row["word_index"]) for row in conn.execute("SELECT DISTINCT word_index FROM maps WHERE case_id=? ORDER BY word_index", (case_id,))]

    for word_index in words:
        points = []
        peak_counts = []
        secondary_ratios = []
        adjacent_cosines = []
        adjacent_jsd = []
        adjacent_top5 = []
        previous = None
        for layer_index in layers:
            metric = conn.execute(
                "SELECT * FROM map_metrics WHERE case_id=? AND word_index=? AND layer_index=?",
                (case_id, word_index, layer_index),
            ).fetchone()
            if metric:
                points.append((metric["global_centroid_x_norm"], metric["global_centroid_y_norm"]))
                if metric["peak_count"] is not None:
                    peak_counts.append(metric["peak_count"])
                if metric["secondary_primary_ratio"] is not None:
                    secondary_ratios.append(metric["secondary_primary_ratio"])
            current = arrays.get((word_index, layer_index))
            if previous is not None and current is not None:
                adjacent_cosines.append(cosine_similarity(previous, current))
                adjacent_jsd.append(jsd(previous, current))
                adjacent_top5.append(top_iou(previous, current, 5.0))
            if current is not None:
                previous = current
        pm = path_metrics(points)
        first = arrays.get((word_index, layers[0])) if layers else None
        last = arrays.get((word_index, layers[-1])) if layers else None
        early_late_cosine = cosine_similarity(first, last) if first is not None and last is not None else None
        early_late_jsd = jsd(first, last) if first is not None and last is not None else None
        conn.execute(
            """
            INSERT OR REPLACE INTO layer_scanpaths VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                case_id,
                word_index,
                pm["path_length"],
                pm["mean_step"],
                pm["max_jump"],
                pm["net_displacement"],
                pm["tortuosity"],
                pm["large_jump_count"],
                pm["bbox_area"],
                mean_nonnull(adjacent_cosines),
                min_nonnull(adjacent_cosines),
                None,
                mean_nonnull(adjacent_jsd),
                None,
                mean_nonnull(adjacent_top5),
                early_late_cosine,
                early_late_jsd,
                None,
                None,
                mean_nonnull(peak_counts),
                max(peak_counts) if peak_counts else None,
                mean_nonnull(secondary_ratios),
                max(secondary_ratios) if secondary_ratios else None,
                int(sum(1 for count in peak_counts if count and count > 1)),
                float(sum(1 for count in peak_counts if count and count > 1) / len(peak_counts)) if peak_counts else None,
            ),
        )

    for layer_index in layers:
        points = []
        for word_index in words:
            metric = conn.execute(
                "SELECT global_centroid_x_norm, global_centroid_y_norm FROM map_metrics WHERE case_id=? AND word_index=? AND layer_index=?",
                (case_id, word_index, layer_index),
            ).fetchone()
            if metric:
                points.append((metric["global_centroid_x_norm"], metric["global_centroid_y_norm"]))
        pm = path_metrics(points)
        conn.execute(
            """
            INSERT OR REPLACE INTO word_scanpaths VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                case_id,
                layer_index,
                pm["path_length"],
                pm["mean_step"],
                pm["max_jump"],
                pm["net_displacement"],
                pm["tortuosity"],
                pm["large_jump_count"],
                pm["bbox_area"],
                pm["revisit_count"],
            ),
        )


def mean_nonnull(values) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return float(np.mean(clean)) if clean else None


def min_nonnull(values) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return float(min(clean)) if clean else None


def main() -> int:
    args = build_parser().parse_args()
    config = DashboardConfig.default()
    config.ensure_dirs()
    conn = connect(config.db_path)
    if args.init_db:
        initialize(conn, rebuild=False)
    query = "SELECT case_id FROM cases"
    clauses = []
    params = []
    if args.official_only:
        clauses.append("is_official=1")
    if args.case_filter:
        clauses.append("case_id LIKE ?")
        params.append(f"%{args.case_filter}%")
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY case_id"
    cases = [row["case_id"] for row in conn.execute(query, params)]
    if args.case_limit > 0:
        cases = cases[: args.case_limit]
    print(f"[precompute] cases={len(cases)} db={config.db_path}")
    for idx, case_id in enumerate(cases, start=1):
        print(f"[precompute] {idx}/{len(cases)} {case_id}")
        precompute_case(conn, config, case_id)
        conn.commit()
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
