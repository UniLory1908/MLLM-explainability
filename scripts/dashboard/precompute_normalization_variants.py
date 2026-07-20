from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dashboard.config import DashboardConfig
from scripts.dashboard.data_access import load_word_layer_map, row_paths
from scripts.dashboard.db import connect, initialize
from scripts.dashboard.metrics import extract_regions, minmax, per_map_metrics, prob


DEFAULT_MODES = ("tam_uint8_native",)
ALL_MODES = ("tam_uint8_native", "local_minmax_0_1", "probability_sum_1")
REGION_THRESHOLD = 0.90


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute additive TAM normalization-variant map metrics.")
    parser.add_argument("--case-filter", help="Substring matched against case_id.")
    parser.add_argument("--official-only", action="store_true")
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--mode", action="append", choices=ALL_MODES, help="Normalization mode to compute. Repeatable.")
    parser.add_argument("--init-db", action="store_true", help="Create schema if needed.")
    parser.add_argument("--replace", action="store_true", help="Replace existing rows for selected modes.")
    parser.add_argument("--include-regions", action="store_true", help="Also compute 0.90 region-derived fields for each variant.")
    return parser


def normalize_for_mode(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "tam_uint8_native":
        return np.asarray(arr, dtype=np.float64)
    if mode == "local_minmax_0_1":
        return minmax(arr)
    if mode == "probability_sum_1":
        return prob(arr)
    raise KeyError(mode)


def insert_variant_metrics(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    mode: str,
    metrics: dict[str, float | int | None],
) -> None:
    columns = [
        "case_id",
        "word_index",
        "layer_index",
        "normalization_mode",
        *metrics.keys(),
        "source_signature",
    ]
    placeholders = ",".join("?" for _ in columns)
    conn.execute(
        f"INSERT OR REPLACE INTO map_metrics_normalization_variants ({','.join(columns)}) VALUES ({placeholders})",
        (
            row["case_id"],
            int(row["word_index"]),
            int(row["layer_index"]),
            mode,
            *metrics.values(),
            row["source_signature"],
        ),
    )


def ensure_table(conn: sqlite3.Connection) -> None:
    initialize(conn, rebuild=False)


def selected_cases(conn: sqlite3.Connection, args: argparse.Namespace) -> list[str]:
    query = "SELECT case_id FROM cases"
    clauses: list[str] = []
    params: list[str] = []
    if args.official_only:
        clauses.append("is_official=1")
    if args.case_filter:
        clauses.append("case_id LIKE ?")
        params.append(f"%{args.case_filter}%")
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY image_id, condition_label, case_id"
    cases = [row["case_id"] for row in conn.execute(query, params)]
    if args.case_limit > 0:
        cases = cases[: args.case_limit]
    return cases


def fast_copy_tam_native(conn: sqlite3.Connection, cases: list[str], replace: bool) -> int:
    if replace:
        if cases:
            placeholders = ",".join("?" for _ in cases)
            conn.execute(
                f"DELETE FROM map_metrics_normalization_variants WHERE normalization_mode='tam_uint8_native' AND case_id IN ({placeholders})",
                cases,
            )
        else:
            conn.execute("DELETE FROM map_metrics_normalization_variants WHERE normalization_mode='tam_uint8_native'")
    if not cases:
        return 0
    placeholders = ",".join("?" for _ in cases)
    before = conn.total_changes
    conn.execute(
        f"""
        INSERT OR IGNORE INTO map_metrics_normalization_variants (
            case_id, word_index, layer_index, normalization_mode,
            energy_sum, energy_mean, min_value, max_value, std_value, nonzero_ratio,
            entropy_norm, top_1_mass, top_5_mass, top_10_mass, hhi,
            effective_area, effective_area_norm, hoyer_sparsity,
            global_centroid_x_px, global_centroid_y_px, global_centroid_x_norm, global_centroid_y_norm,
            spread_trace, spread_x, spread_y, covariance_xy, anisotropy,
            peak_count, primary_region_mass, secondary_region_mass, secondary_primary_ratio,
            primary_region_centroid_x_px, primary_region_centroid_y_px,
            primary_region_centroid_x_norm, primary_region_centroid_y_norm,
            secondary_region_centroid_x_norm, secondary_region_centroid_y_norm,
            is_multipeak, source_signature
        )
        SELECT
            mm.case_id, mm.word_index, mm.layer_index, 'tam_uint8_native',
            mm.energy_sum, mm.energy_mean, mm.min_value, mm.max_value, mm.std_value, mm.nonzero_ratio,
            mm.entropy_norm, mm.top_1_mass, mm.top_5_mass, mm.top_10_mass, mm.hhi,
            mm.effective_area, mm.effective_area_norm, mm.hoyer_sparsity,
            mm.global_centroid_x_px, mm.global_centroid_y_px, mm.global_centroid_x_norm, mm.global_centroid_y_norm,
            mm.spread_trace, mm.spread_x, mm.spread_y, mm.covariance_xy, mm.anisotropy,
            mm.peak_count, mm.primary_region_mass, mm.secondary_region_mass, mm.secondary_primary_ratio,
            mm.primary_region_centroid_x_px, mm.primary_region_centroid_y_px,
            mm.primary_region_centroid_x_norm, mm.primary_region_centroid_y_norm,
            mm.secondary_region_centroid_x_norm, mm.secondary_region_centroid_y_norm,
            mm.is_multipeak, maps.source_signature
        FROM map_metrics mm
        JOIN maps
          ON maps.case_id=mm.case_id
         AND maps.word_index=mm.word_index
         AND maps.layer_index=mm.layer_index
        WHERE mm.case_id IN ({placeholders})
        """,
        cases,
    )
    conn.commit()
    return int(conn.total_changes - before)


def existing_modes_for_case(conn: sqlite3.Connection, case_id: str) -> dict[tuple[int, int], set[str]]:
    existing: dict[tuple[int, int], set[str]] = {}
    for value in conn.execute(
        """
        SELECT word_index, layer_index, normalization_mode
        FROM map_metrics_normalization_variants
        WHERE case_id=?
        """,
        (case_id,),
    ):
        key = (int(value["word_index"]), int(value["layer_index"]))
        existing.setdefault(key, set()).add(str(value["normalization_mode"]))
    return existing


def precompute_cases(
    conn: sqlite3.Connection,
    config: DashboardConfig,
    cases: list[str],
    modes: tuple[str, ...],
    replace: bool,
    batch_size: int,
    include_regions: bool,
) -> dict[str, int]:
    counts = {"maps_seen": 0, "maps_loaded": 0, "rows_written": 0, "rows_skipped": 0, "unreadable_maps": 0}
    batch_size = max(1, int(batch_size))
    for case_idx, case_id in enumerate(cases, start=1):
        maps = list(
            conn.execute(
                """
                SELECT maps.*
                FROM maps
                JOIN map_metrics mm
                  ON mm.case_id=maps.case_id
                 AND mm.word_index=maps.word_index
                 AND mm.layer_index=maps.layer_index
                WHERE maps.case_id=? AND maps.map_exists=1
                ORDER BY maps.word_index, maps.layer_index
                """,
                (case_id,),
            )
        )
        existing_by_key = {} if replace else existing_modes_for_case(conn, case_id)
        print(f"[variants] case {case_idx}/{len(cases)} {case_id} maps={len(maps)}", flush=True)
        for row in maps:
            counts["maps_seen"] += 1
            wanted_modes = set(modes)
            if not replace:
                wanted_modes -= existing_by_key.get((int(row["word_index"]), int(row["layer_index"])), set())
            if not wanted_modes:
                counts["rows_skipped"] += len(modes)
                continue
            try:
                arr = load_word_layer_map(row_paths(row, config))
            except (OSError, ValueError) as exc:
                counts["unreadable_maps"] += 1
                print(
                    f"[warn] skip unreadable map case={row['case_id']} word_index={row['word_index']} "
                    f"layer_index={row['layer_index']}: {exc}",
                    flush=True,
                )
                continue
            if arr is None:
                counts["unreadable_maps"] += 1
                continue
            counts["maps_loaded"] += 1
            for mode in modes:
                if mode not in wanted_modes:
                    continue
                normalized = normalize_for_mode(arr, mode)
                regions_90 = extract_regions(normalized, threshold=REGION_THRESHOLD) if include_regions else []
                insert_variant_metrics(conn, row, mode, per_map_metrics(normalized, regions_90))
                counts["rows_written"] += 1
            if counts["maps_seen"] % batch_size == 0:
                conn.commit()
                print(
                    f"[variants] progress maps_seen={counts['maps_seen']} rows_written={counts['rows_written']} "
                    f"rows_skipped={counts['rows_skipped']} unreadable={counts['unreadable_maps']}",
                    flush=True,
                )
        conn.commit()
    return counts


def main() -> int:
    args = build_parser().parse_args()
    config = DashboardConfig.default()
    config.ensure_dirs()
    conn = connect(config.db_path)
    if args.init_db:
        ensure_table(conn)
    modes = tuple(args.mode or DEFAULT_MODES)
    cases = selected_cases(conn, args)
    print(f"[variants] cases={len(cases)} modes={','.join(modes)} db={config.db_path}", flush=True)
    if "tam_uint8_native" in modes and not args.include_regions:
        copied = fast_copy_tam_native(conn, cases, args.replace)
        print(f"[variants] fast_copied_tam_uint8_native_rows={copied}", flush=True)
        modes = tuple(mode for mode in modes if mode != "tam_uint8_native")
    counts = precompute_cases(conn, config, cases, modes, args.replace, args.batch_size, args.include_regions) if modes else {
        "maps_seen": 0,
        "maps_loaded": 0,
        "rows_written": 0,
        "rows_skipped": 0,
        "unreadable_maps": 0,
    }
    conn.commit()
    print("[variants] done " + " ".join(f"{key}={value}" for key, value in counts.items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
