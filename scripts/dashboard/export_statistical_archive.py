from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dashboard.config import DashboardConfig
from scripts.dashboard.db import connect


REQUIRED_TABLES = [
    "cases",
    "map_metrics_core",
    "map_metrics_normalization_variants",
    "layer_scanpaths",
    "word_scanpaths",
    "region_summary",
    "output_diagnostics",
    "visual_sensitivity_vs_baseline",
    "diagnostic_scores",
    "token_category_summary",
    "dashboard_links",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export dashboard DB and derived metrics as a statistical archive.")
    parser.add_argument("--official-only", action="store_true", help="Export only cases marked is_official=1.")
    parser.add_argument("--archive-name", required=True, help="Archive folder name under outputs/statistical_archive.")
    parser.add_argument("--dashboard-base-url", default="http://127.0.0.1:5050")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing archive folder.")
    parser.add_argument("--skip-parquet", action="store_true")
    parser.add_argument("--skip-xlsx", action="store_true")
    return parser


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def case_filter_sql(official_only: bool, alias: str = "c") -> tuple[str, list[Any]]:
    if not official_only:
        return "", []
    return f" WHERE {alias}.is_official=1", []


def selected_case_ids(conn: sqlite3.Connection, official_only: bool) -> list[str]:
    where, params = case_filter_sql(official_only, "cases")
    return [row[0] for row in conn.execute(f"SELECT case_id FROM cases{where} ORDER BY image_id, condition_label, case_id", params)]


def sql_for_table(table: str, official_only: bool) -> str:
    if table == "cases":
        where, _ = case_filter_sql(official_only, "cases")
        return f"SELECT * FROM cases{where} ORDER BY image_id, condition_label, case_id"
    if table == "map_metrics_core":
        where = "WHERE c.is_official=1" if official_only else ""
        return f"""
            SELECT m.*, c.image_id, c.condition_id, c.condition_label, c.prompt_label, c.run_name, c.is_official
            FROM map_metrics m
            JOIN cases c ON c.case_id=m.case_id
            {where}
            ORDER BY c.image_id, c.condition_label, m.word_index, m.layer_index
        """
    if table == "map_metrics_normalization_variants":
        where = "WHERE c.is_official=1" if official_only else ""
        return f"""
            SELECT m.*, c.image_id, c.condition_id, c.condition_label, c.prompt_label, c.run_name, c.is_official
            FROM map_metrics_normalization_variants m
            JOIN cases c ON c.case_id=m.case_id
            {where}
            ORDER BY c.image_id, c.condition_label, m.word_index, m.layer_index, m.normalization_mode
        """
    if table in {"layer_scanpaths", "word_scanpaths", "output_diagnostics", "visual_sensitivity_vs_baseline", "diagnostic_scores", "token_category_summary"}:
        where = "WHERE c.is_official=1" if official_only else ""
        return f"""
            SELECT t.*
            FROM {table} t
            JOIN cases c ON c.case_id=t.case_id
            {where}
            ORDER BY c.image_id, c.condition_label, t.case_id
        """
    if table == "region_summary":
        where = "WHERE c.is_official=1" if official_only else ""
        return f"""
            SELECT c.case_id, c.image_id, c.condition_id, c.condition_label, c.prompt_label,
                   r.threshold, COUNT(*) AS region_count,
                   AVG(r.mass) AS mean_region_mass,
                   AVG(r.area) AS mean_region_area,
                   AVG(r.ratio_to_primary) AS mean_ratio_to_primary,
                   AVG(r.centroid_x_norm) AS mean_centroid_x_norm,
                   AVG(r.centroid_y_norm) AS mean_centroid_y_norm
            FROM regions r
            JOIN cases c ON c.case_id=r.case_id
            {where}
            GROUP BY c.case_id, r.threshold
            ORDER BY c.image_id, c.condition_label, r.threshold
        """
    if table == "dashboard_links":
        where, _ = case_filter_sql(official_only, "c")
        return f"""
            SELECT c.case_id, c.image_id, c.condition_id, c.condition_label, c.prompt_label, c.run_name, c.is_official,
                   '' AS dashboard_case_url,
                   '' AS dashboard_matrix_url,
                   '' AS dashboard_compare_url,
                   '' AS dashboard_word_url
            FROM cases c
            {where}
            ORDER BY c.image_id, c.condition_label, c.case_id
        """
    raise KeyError(table)


def add_dashboard_links(df: pd.DataFrame, base_url: str) -> pd.DataFrame:
    base = base_url.rstrip("/")
    if "case_id" in df.columns:
        df["dashboard_case_url"] = df["case_id"].map(lambda value: f"{base}/case/{value}")
        df["dashboard_matrix_url"] = df["case_id"].map(lambda value: f"{base}/case/{value}/matrix")
    if "image_id" in df.columns:
        df["dashboard_compare_url"] = df["image_id"].map(lambda value: f"{base}/compare?image_id={value}")
    if "case_id" in df.columns and "word_index" in df.columns:
        df["dashboard_word_url"] = df.apply(lambda row: f"{base}/case/{row['case_id']}/word/{int(row['word_index'])}" if pd.notna(row["word_index"]) else "", axis=1)
    return df


def read_export_frame(conn: sqlite3.Connection, table: str, official_only: bool, dashboard_base_url: str) -> pd.DataFrame:
    df = pd.read_sql_query(sql_for_table(table, official_only), conn)
    if table == "dashboard_links":
        df = add_dashboard_links(df, dashboard_base_url)
    elif table in {"cases", "output_diagnostics", "visual_sensitivity_vs_baseline", "diagnostic_scores", "token_category_summary"}:
        df = add_dashboard_links(df, dashboard_base_url)
    return df


def prepare_archive(root: Path, overwrite: bool) -> None:
    archive_parent = DashboardConfig.default().project_root / "outputs" / "statistical_archive"
    resolved_root = root.resolve()
    resolved_parent = archive_parent.resolve()
    if resolved_root == resolved_parent or resolved_parent not in resolved_root.parents:
        raise ValueError(f"Refusing to prepare archive outside {archive_parent}: {root}")
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"Archive already exists: {root}. Use --overwrite to replace it.")
        import shutil

        shutil.rmtree(root)
    for child in ["csv", "parquet", "preview", "manifest", "logs"]:
        (root / child).mkdir(parents=True, exist_ok=True)


def write_dataset_readme(root: Path, args: argparse.Namespace, counts: dict[str, int], notes: list[str]) -> None:
    lines = [
        "# TAM Statistical Archive",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Archive name: `{args.archive_name}`",
        f"Official only: `{args.official_only}`",
        f"Dashboard base URL: `{args.dashboard_base_url}`",
        "",
        "## Contents",
        "",
        "- `csv/`: required CSV tables.",
        "- `parquet/`: optional Parquet tables when an engine is installed.",
        "- `preview/`: small XLSX/index preview when Excel support is installed.",
        "- `manifest/`: JSON manifest with counts and export notes.",
        "",
        "## Tables",
        "",
    ]
    for table, count in counts.items():
        lines.append(f"- `{table}`: {count} rows")
    lines.extend(
        [
            "",
            "## Normalization Tables",
            "",
            "`map_metrics_core` preserves the existing dashboard metric definitions. "
            "`map_metrics_normalization_variants` is additive; the full archive currently "
            "includes `normalization_mode=tam_uint8_native`, an explicit row set for metrics "
            "computed on the saved TAM postprocessed 0-255-like maps.",
            "",
            "## Limitations",
            "",
            "- Baseline comparisons use `baseline_neutral` only.",
            "- Visual baseline alignment is positional/common-index only.",
            "- Three indexed raw maps are not covered by `map_metrics_core` because their `.npy` files are unreadable or filesystem-stalling.",
            "- No semantic word alignment, GT metric, causal faithfulness metric, or bulk EMD is included.",
            "- Candidate diagnostic scores are bounded ranking proxies, not proof of hallucination or causal grounding.",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in notes)
    (root / "DATASET_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_archive_summary(root: Path, args: argparse.Namespace, counts: dict[str, int], notes: list[str], image_count: int) -> None:
    lines = [
        "# TAM Statistical Archive - Archive Summary",
        "",
        f"Archive: `{args.archive_name}`",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Dataset Scope",
        "",
        f"- Cases: {counts.get('cases', 0)}",
        f"- Images: {image_count}",
        "- Prompt conditions per image: 8",
        "- Model/source: Qwen2-VL TAM prompt-sensitivity runs indexed from `outputs/prompt_sensitivity/`.",
        "- Excluded image `224724` is not part of this archive.",
        "",
        "## Main Tables",
        "",
    ]
    for table, count in counts.items():
        lines.append(f"- `{table}`: {count} rows")
    lines.extend(
        [
            "",
            "## Normalization Notes",
            "",
            "- `map_metrics_core` preserves the existing dashboard metric definitions.",
            "- `map_metrics_normalization_variants` currently includes `tam_uint8_native`, an explicit copy of metrics computed on the saved TAM postprocessed uint8-like raw maps.",
            "- Dashboard shape/comparison metrics use cleaned raw values, probability normalization, local minmax normalization, and word-level pixel-wise max pooling over subtoken maps as appropriate.",
            "- TAM original processing normalizes image/text scores jointly, applies the rank-Gaussian filter to image token scores, and scales maps to 0-255 uint8-like values before saving in this pipeline.",
            "",
            "## Known Limitations",
            "",
            "- Three raw map files remain unreadable or filesystem-stalling, so `map_metrics_core` and `tam_uint8_native` cover 995683 of 995686 indexed maps.",
            "- No broad Qwen/TAM inference, no new images, no bulk EMD, and no raw-map pairwise similarity bulk export are included.",
            "- No semantic word alignment, ground-truth metric, or causal faithfulness metric is included.",
            "- Candidate diagnostic scores are exploratory proxies for ranking and inspection, not proof of hallucination or causal grounding.",
            "",
            "## Export Formats",
            "",
            "- CSV tables are written under `csv/`.",
            "- Parquet tables are written under `parquet/` when the local engine is available.",
            "- XLSX is a small preview/index workbook only.",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in notes)
    (root / "RUN_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_preview_workbook(root: Path, frames: dict[str, pd.DataFrame]) -> str | None:
    preview_path = root / "preview" / "statistical_archive_preview.xlsx"
    try:
        with pd.ExcelWriter(preview_path) as writer:
            pd.DataFrame(
                {
                    "field": ["purpose", "scope"],
                    "value": [
                        "Preview workbook for the TAM statistical archive",
                        "Small index-style sheets only; full tables are in CSV/Parquet.",
                    ],
                }
            ).to_excel(writer, sheet_name="README", index=False)
            for sheet, table in [
                ("cases", "cases"),
                ("case_summary", "output_diagnostics"),
                ("condition_summary", "visual_sensitivity_vs_baseline"),
                ("visual_sensitivity", "visual_sensitivity_vs_baseline"),
                ("diagnostic_scores", "diagnostic_scores"),
                ("dashboard_links", "dashboard_links"),
            ]:
                if table in frames:
                    frames[table].head(200).to_excel(writer, sheet_name=sheet[:31], index=False)
        return str(preview_path)
    except Exception as exc:
        if preview_path.exists():
            preview_path.unlink()
        return f"skipped:{exc}"


def main() -> int:
    args = build_parser().parse_args()
    config = DashboardConfig.default()
    conn = connect(config.db_path)
    missing = [table for table in ["output_diagnostics", "visual_sensitivity_vs_baseline", "diagnostic_scores", "token_category_summary"] if not table_exists(conn, table)]
    if missing:
        raise RuntimeError("Derived tables are missing. Run scripts.dashboard.precompute_derived_metrics first: " + ", ".join(missing))

    archive_root = config.project_root / "outputs" / "statistical_archive" / args.archive_name
    prepare_archive(archive_root, args.overwrite)

    frames: dict[str, pd.DataFrame] = {}
    counts: dict[str, int] = {}
    notes: list[str] = []
    parquet_ok = not args.skip_parquet
    for table in REQUIRED_TABLES:
        df = read_export_frame(conn, table, args.official_only, args.dashboard_base_url)
        frames[table] = df
        counts[table] = int(len(df))
        df.to_csv(archive_root / "csv" / f"{table}.csv", index=False)
        if parquet_ok:
            try:
                df.to_parquet(archive_root / "parquet" / f"{table}.parquet", index=False)
            except Exception as exc:
                parquet_ok = False
                notes.append(f"Parquet export skipped after `{table}` because no working parquet engine was available: {exc}")

    xlsx_status: str | None = None
    if args.skip_xlsx:
        notes.append("XLSX preview skipped by --skip-xlsx.")
    else:
        xlsx_status = write_preview_workbook(archive_root, frames)
        if xlsx_status and xlsx_status.startswith("skipped:"):
            notes.append(f"XLSX preview skipped because Excel writer support was unavailable: {xlsx_status.removeprefix('skipped:')}")
        elif xlsx_status:
            notes.append(f"XLSX preview written: {xlsx_status}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive_name": args.archive_name,
        "db_path": str(config.db_path),
        "source_root": str(config.source_root),
        "official_only": args.official_only,
        "dashboard_base_url": args.dashboard_base_url,
        "tables": counts,
        "csv_written": True,
        "parquet_written": parquet_ok and not args.skip_parquet,
        "xlsx_status": xlsx_status or "skipped_by_option",
        "selected_case_count": len(selected_case_ids(conn, args.official_only)),
        "selected_image_count": int(frames["cases"]["image_id"].nunique()) if "cases" in frames and "image_id" in frames["cases"].columns else None,
        "notes": notes,
    }
    (archive_root / "manifest" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_dataset_readme(archive_root, args, counts, notes)
    write_archive_summary(archive_root, args, counts, notes, int(manifest["selected_image_count"] or 0))
    print(f"[export] archive={archive_root}")
    for table, count in counts.items():
        print(f"[export] {table} rows={count}")
    for note in notes:
        print(f"[export] note: {note}")
    print("[export] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
