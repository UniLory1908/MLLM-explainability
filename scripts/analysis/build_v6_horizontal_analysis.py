from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = PROJECT_ROOT / "outputs" / "statistical_archive" / "stat_timebox_20260523_progress"
DB_PATH = PROJECT_ROOT / "outputs" / "dashboard_index" / "tam_index.sqlite"
V1_DIR = PROJECT_ROOT / "outputs" / "analysis" / "v1_case_level"
OUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "v6_horizontal_analysis"
PROMPT_ORDER = [
    "baseline_neutral",
    "image_grounded_visible_only",
    "ambiguous_open",
    "misleading_wrong_subject",
    "extra_knowledge_context",
    "reasoning_controlled_brief",
    "order_disruption_stress",
    "colleague_obj_detection_hard",
]
PROMPT_CATEGORY = {
    "baseline_neutral": "baseline",
    "image_grounded_visible_only": "image_grounded",
    "ambiguous_open": "ambiguous",
    "misleading_wrong_subject": "misleading",
    "extra_knowledge_context": "extra_knowledge",
    "reasoning_controlled_brief": "reasoning",
    "order_disruption_stress": "grounding_format_stress",
    "colleague_obj_detection_hard": "object_detection",
}
EXPECTED_BBOX_PROMPTS = {"order_disruption_stress", "colleague_obj_detection_hard"}
RANDOM_SEED = 20260619


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v6 horizontal TAM analysis artifacts.")
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--v1-dir", type=Path, default=V1_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser


def read_table(root: Path, name: str) -> pd.DataFrame:
    parquet_path = root / "parquet" / f"{name}.parquet"
    csv_path = root / "csv" / f"{name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing archive table {name}")


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = ["_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col) for col in out.columns]
    return out


def iqr(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    return float(s.quantile(0.75) - s.quantile(0.25))


def summarize_numeric(df: pd.DataFrame, group_col: str, specs: dict[str, list[str]]) -> pd.DataFrame:
    agg = {}
    for col, funcs in specs.items():
        if col in df.columns:
            agg[col] = funcs
    if not agg:
        return pd.DataFrame({group_col: sorted(df[group_col].dropna().unique())})
    return flatten_columns(df.groupby(group_col, dropna=False).agg(agg).reset_index())


def robust_scale(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    cols = []
    params: dict[str, dict[str, float]] = {}
    for col in frame.columns:
        s = pd.to_numeric(frame[col], errors="coerce")
        median = float(s.median()) if s.notna().any() else 0.0
        q1 = float(s.quantile(0.25)) if s.notna().any() else 0.0
        q3 = float(s.quantile(0.75)) if s.notna().any() else 1.0
        denom = q3 - q1
        if not np.isfinite(denom) or denom == 0:
            denom = float(s.std(ddof=0)) if s.notna().any() else 1.0
        if not np.isfinite(denom) or denom == 0:
            denom = 1.0
        cols.append(((s.fillna(median) - median) / denom).clip(-8, 8).to_numpy(dtype=float))
        params[col] = {"median": median, "iqr_or_std": denom, "missing": int(s.isna().sum())}
    return np.column_stack(cols), params


def pca_scores(x: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    centered = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vt[:n_components].T
    denom = max(x.shape[0] - 1, 1)
    explained = (s[:n_components] ** 2) / denom
    total = float(np.sum((s**2) / denom))
    ratio = explained / total if total else np.zeros_like(explained)
    return scores, ratio


def bh_fdr(p_values: Iterable[float]) -> list[float]:
    vals = np.asarray([np.nan if p is None else p for p in p_values], dtype=float)
    out = np.full(vals.shape, np.nan)
    ok = np.isfinite(vals)
    if not ok.any():
        return out.tolist()
    idx = np.where(ok)[0]
    order = idx[np.argsort(vals[ok])]
    ranked = vals[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[order] = np.clip(adjusted, 0, 1)
    return out.tolist()


def short_preview(text: object, n: int = 180) -> str:
    if not isinstance(text, str):
        return ""
    clean = re.sub(r"\s+", " ", text).strip()
    return clean[: n - 1] + "..." if len(clean) > n else clean


STRICT_COORD_RE = re.compile(
    r"(?i)(?:bbox|bounding\s*box|box|coordinates?|location)\D{0,60}"
    r"(?:\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]"
    r"|\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)\s*(?:,|to|-|and)\s*\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\))"
)
COORD_LIKE_RE = re.compile(
    r"(?:\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]"
    r"|\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\))"
)
NUMERIC_SUSPECT_RE = re.compile(
    r"(?i)\b(?:19|20)\d{2}\b|\b\d{1,4}\s?(?:am|pm|°|degrees?|percent|%|kg|mph|km/h)\b|"
    r"\b(?:train|bus|route|jersey|number|id|model)\s*#?\s*\d+\b|\b\d+\s*[:/]\s*\d+\b"
)


def add_bbox_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text = out["response_text"].fillna("").astype(str)
    strict_regex = text.str.contains(STRICT_COORD_RE, regex=True)
    coord_like = text.str.contains(COORD_LIKE_RE, regex=True)
    stored = pd.to_numeric(out.get("bbox_style_output_flag", 0), errors="coerce").fillna(0).astype(int).gt(0)
    coord_tokens = pd.to_numeric(out.get("coordinate_token_count", 0), errors="coerce").fillna(0).gt(0)
    box_tokens = pd.to_numeric(out.get("has_box_tokens", 0), errors="coerce").fillna(0).astype(int).gt(0)
    out["bbox_strict"] = (stored | strict_regex).astype(int)
    out["bbox_broad"] = (out["bbox_strict"].astype(bool) | coord_tokens | box_tokens | coord_like).astype(int)
    numeric_suspect = text.str.contains(NUMERIC_SUSPECT_RE, regex=True)
    out["bbox_numeric_only_suspect"] = (out["bbox_broad"].astype(bool) & ~out["bbox_strict"].astype(bool) & numeric_suspect).astype(int)
    out["bbox_expected_prompt"] = out["prompt_label"].isin(EXPECTED_BBOX_PROMPTS).astype(int)
    out["bbox_unexpected"] = (out["bbox_strict"].astype(bool) & ~out["bbox_expected_prompt"].astype(bool)).astype(int)
    return out


def db_case_counts(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame(columns=["case_id", "db_map_count", "db_map_metrics_count"])
    with sqlite3.connect(db_path) as conn:
        maps = pd.read_sql_query("select case_id, count(*) as db_map_count from maps group by case_id", conn)
        metrics = pd.read_sql_query("select case_id, count(*) as db_map_metrics_count from map_metrics group by case_id", conn)
    out = maps.merge(metrics, on="case_id", how="outer").fillna(0)
    out["db_map_count"] = out["db_map_count"].astype(int)
    out["db_map_metrics_count"] = out["db_map_metrics_count"].astype(int)
    out["missing_map_metrics_count"] = out["db_map_count"] - out["db_map_metrics_count"]
    return out


def build_case_features(tables: dict[str, pd.DataFrame], db_path: Path, v1_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    cases = tables["cases"].copy()
    cases["prompt_category"] = cases["prompt_label"].map(PROMPT_CATEGORY).fillna("unknown")
    cases["analysis_include_full800"] = cases["status"].eq("ok")
    cases["strict_homogeneous_subset_flag"] = (cases["used_fix256"].fillna(0).astype(int).eq(0) & cases["run_name"].notna()).astype(int)

    df = cases.copy()
    for name in ["output_diagnostics", "visual_sensitivity_vs_baseline", "diagnostic_scores", "dashboard_links"]:
        right = tables[name].copy()
        keep = ["case_id"] + [c for c in right.columns if c != "case_id" and c not in df.columns]
        df = df.merge(right[keep], on="case_id", how="left", validate="one_to_one")

    df["response_preview"] = df["response_text"].map(short_preview)
    df = add_bbox_flags(df)

    baseline = cases.loc[cases["prompt_label"].eq("baseline_neutral"), ["image_id", "case_id"]].rename(
        columns={"case_id": "baseline_case_id_from_cases"}
    )
    df = df.merge(baseline, on="image_id", how="left", validate="many_to_one")
    if "baseline_case_id" not in df.columns:
        df["baseline_case_id"] = df["baseline_case_id_from_cases"]
    else:
        df["baseline_case_id"] = df["baseline_case_id"].fillna(df["baseline_case_id_from_cases"])
    df = df.drop(columns=["baseline_case_id_from_cases"])

    counts = db_case_counts(db_path)
    if counts.empty:
        mm_counts = tables["map_metrics_core"].groupby("case_id").size().rename("map_metrics_count").reset_index()
        df = df.merge(mm_counts, on="case_id", how="left")
        df["map_count"] = df["map_metrics_count"]
        df["missing_map_metrics_count"] = 0
    else:
        df = df.merge(counts, on="case_id", how="left")
        df["map_count"] = df["db_map_count"].fillna(0).astype(int)
        df["map_metrics_count"] = df["db_map_metrics_count"].fillna(0).astype(int)
        df["missing_map_metrics_count"] = df["missing_map_metrics_count"].fillna(0).astype(int)

    map_specs = {
        "entropy_norm": ["mean", "median", iqr],
        "top_5_mass": ["mean", "median", iqr],
        "effective_area_norm": ["mean", "median", iqr],
        "hhi": ["mean", "median", iqr],
        "hoyer_sparsity": ["mean", "median", iqr],
        "spread_trace": ["mean", "median"],
        "global_centroid_x_norm": ["mean", "std"],
        "global_centroid_y_norm": ["mean", "std"],
        "anisotropy": ["mean"],
        "peak_count": ["mean"],
        "primary_region_mass": ["mean", "median"],
        "secondary_primary_ratio": ["mean", "median"],
        "is_multipeak": ["mean"],
    }
    map_agg = summarize_numeric(tables["map_metrics_core"], "case_id", map_specs)
    map_agg = map_agg.rename(
        columns={
            "entropy_norm_iqr": "entropy_norm_iqr",
            "top_5_mass_iqr": "top_5_mass_iqr",
            "effective_area_norm_iqr": "effective_area_norm_iqr",
            "hhi_iqr": "hhi_iqr",
            "hoyer_sparsity_iqr": "hoyer_sparsity_iqr",
            "is_multipeak_mean": "multipeak_proportion",
        }
    )
    centroid_range = tables["map_metrics_core"].groupby("case_id").agg(
        global_centroid_x_norm_range=("global_centroid_x_norm", lambda s: float(pd.to_numeric(s, errors="coerce").max() - pd.to_numeric(s, errors="coerce").min())),
        global_centroid_y_norm_range=("global_centroid_y_norm", lambda s: float(pd.to_numeric(s, errors="coerce").max() - pd.to_numeric(s, errors="coerce").min())),
    ).reset_index()
    df = df.merge(map_agg, on="case_id", how="left", validate="one_to_one")
    df = df.merge(centroid_range, on="case_id", how="left", validate="one_to_one")

    regions = tables["region_summary"].copy()
    regions["threshold_suffix"] = regions["threshold"].map(lambda x: f"thr{int(round(float(x) * 100)):03d}")
    region_cols = ["region_count", "mean_region_mass", "mean_region_area", "mean_ratio_to_primary"]
    region_pivots = []
    for col in region_cols:
        if col in regions.columns:
            p = regions.pivot_table(index="case_id", columns="threshold_suffix", values=col, aggfunc="mean")
            p.columns = [f"{col}_{c}" for c in p.columns]
            region_pivots.append(p)
    if region_pivots:
        region_agg = pd.concat(region_pivots, axis=1).reset_index()
        df = df.merge(region_agg, on="case_id", how="left", validate="one_to_one")

    layer_specs = {
        "layer_path_length": ["mean", "median"],
        "layer_max_jump": ["mean", "median"],
        "layer_net_displacement": ["mean", "median"],
        "layer_tortuosity": ["mean", "median"],
        "layer_large_jump_count": ["mean"],
        "multipeak_layer_ratio": ["mean"],
    }
    layer_agg = summarize_numeric(tables["layer_scanpaths"], "case_id", layer_specs)
    df = df.merge(layer_agg, on="case_id", how="left", validate="one_to_one")

    word_specs = {
        "word_path_length": ["mean", "median"],
        "word_max_jump": ["mean", "median"],
        "word_net_displacement": ["mean", "median"],
        "word_tortuosity": ["mean", "median"],
        "word_large_jump_count": ["mean"],
        "word_revisit_count": ["mean"],
    }
    word_agg = summarize_numeric(tables["word_scanpaths"], "case_id", word_specs)
    df = df.merge(word_agg, on="case_id", how="left", validate="one_to_one")

    token = tables["token_category_summary"].copy()
    if not token.empty:
        token["token_category_clean"] = token["token_category"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
        selected = ["token_count", "mean_entropy_norm", "mean_top5_mass", "mean_layer_path_length", "mean_layer_tortuosity"]
        pivots = []
        for col in selected:
            if col in token.columns:
                p = token.pivot_table(index="case_id", columns="token_category_clean", values=col, aggfunc="mean")
                p.columns = [f"token_{cat}_{col}" for cat in p.columns]
                pivots.append(p)
        if pivots:
            token_agg = pd.concat(pivots, axis=1).reset_index()
            df = df.merge(token_agg, on="case_id", how="left", validate="one_to_one")

    v1_path = v1_dir / "analysis_case_level_v1.parquet"
    if v1_path.exists():
        v1 = pd.read_parquet(v1_path)
        v1_keep = [
            "case_id",
            "textual_change",
            "visual_change",
            "quadrant",
            "text_changed_visual_stable_score",
            "visual_changed_text_stable_score",
        ]
        v1_keep = [c for c in v1_keep if c in v1.columns and c not in df.columns]
        if v1_keep:
            df = df.merge(v1[["case_id"] + v1_keep], on="case_id", how="left", validate="one_to_one")

    # Prefer stable, analysis-facing column names.
    rename_map = {
        "global_centroid_x_norm_mean": "centroid_x_mean",
        "global_centroid_y_norm_mean": "centroid_y_mean",
        "global_centroid_x_norm_std": "centroid_x_std",
        "global_centroid_y_norm_std": "centroid_y_std",
        "peak_count_mean": "peak_count_mean_map",
        "secondary_primary_ratio_mean": "secondary_primary_ratio_mean_map",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["prompt_label"] = pd.Categorical(df["prompt_label"], categories=PROMPT_ORDER, ordered=True)
    df = df.sort_values(["image_id", "prompt_label"]).reset_index(drop=True)

    metadata = {
        "case_rows": int(len(df)),
        "valid_case_rows": int(df["analysis_include_full800"].sum()),
        "map_count_total": int(df["map_count"].sum()),
        "map_metrics_count_total": int(df["map_metrics_count"].sum()),
        "missing_map_metrics_total": int(df["missing_map_metrics_count"].sum()),
        "bbox_style_output_flag_count": int(pd.to_numeric(df["bbox_style_output_flag"], errors="coerce").fillna(0).sum()),
        "bbox_strict_count": int(df["bbox_strict"].sum()),
        "bbox_broad_count": int(df["bbox_broad"].sum()),
        "bbox_numeric_only_suspect_count": int(df["bbox_numeric_only_suspect"].sum()),
    }
    return df, metadata


def write_dataframe(df: pd.DataFrame, path_base: Path) -> None:
    df.to_parquet(path_base.with_suffix(".parquet"), index=False)
    df.to_csv(path_base.with_suffix(".csv"), index=False)


def summary_stats(df: pd.DataFrame, group_col: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(group_col, observed=False):
        row: dict[str, object] = {group_col: key, "n_cases": int(len(group))}
        for col in metrics:
            if col not in group.columns:
                continue
            s = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(s.mean()) if s.notna().any() else np.nan
            row[f"{col}_median"] = float(s.median()) if s.notna().any() else np.nan
            row[f"{col}_std"] = float(s.std()) if s.notna().sum() > 1 else np.nan
            row[f"{col}_iqr"] = iqr(s) if s.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_tables(df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    core = [
        "visual_change",
        "entropy_norm_mean",
        "entropy_mean_delta_vs_baseline",
        "top_5_mass_mean",
        "top5_mass_mean_delta_vs_baseline",
        "effective_area_norm_mean",
        "effective_area_norm_mean_delta_vs_baseline",
        "mean_centroid_shift_vs_baseline",
        "weak_grounding_candidate_score",
        "multipeak_ambiguity_score",
        "unstable_explanation_candidate_score",
    ]
    prompt_summary = summary_stats(df, "prompt_label", [c for c in core if c in df.columns])
    bbox_counts = df.groupby("prompt_label", observed=False).agg(
        bbox_style_output_flag_count=("bbox_style_output_flag", "sum"),
        bbox_strict_count=("bbox_strict", "sum"),
        bbox_broad_count=("bbox_broad", "sum"),
        bbox_numeric_only_suspect_count=("bbox_numeric_only_suspect", "sum"),
    ).reset_index()
    prompt_summary = prompt_summary.merge(bbox_counts, on="prompt_label", how="left")
    for col in ["bbox_style_output_flag", "bbox_strict", "bbox_broad", "bbox_numeric_only_suspect"]:
        count_col = f"{col}_count"
        if count_col in prompt_summary.columns:
            prompt_summary[f"{col}_rate"] = prompt_summary[count_col] / prompt_summary["n_cases"]
    prompt_summary.to_csv(table_dir / "prompt_summary.csv", index=False)

    image_rank = df.groupby("image_id").agg(
        n_cases=("case_id", "count"),
        mean_abs_visual_delta=("visual_change", "mean"),
        max_visual_change=("visual_change", "max"),
        max_centroid_shift=("mean_centroid_shift_vs_baseline", "max"),
        entropy_range=("entropy_norm_mean", lambda s: float(pd.to_numeric(s, errors="coerce").max() - pd.to_numeric(s, errors="coerce").min())),
        top5_mass_range=("top_5_mass_mean", lambda s: float(pd.to_numeric(s, errors="coerce").max() - pd.to_numeric(s, errors="coerce").min())),
        bbox_count=("bbox_strict", "sum"),
        weak_grounding_mean=("weak_grounding_candidate_score", "mean"),
        weak_grounding_max=("weak_grounding_candidate_score", "max"),
        compare_link=("dashboard_compare_url", "first"),
    ).reset_index()
    image_rank["sensitivity_rank"] = image_rank["mean_abs_visual_delta"].rank(ascending=False, method="min").astype(int)
    image_rank["stability_rank"] = image_rank["mean_abs_visual_delta"].rank(ascending=True, method="min").astype(int)
    image_rank.sort_values("sensitivity_rank").to_csv(table_dir / "image_sensitivity_ranking.csv", index=False)

    bbox_by_prompt = df.groupby("prompt_label", observed=False).agg(
        n_cases=("case_id", "count"),
        stored_bbox_count=("bbox_style_output_flag", "sum"),
        bbox_strict_count=("bbox_strict", "sum"),
        bbox_broad_count=("bbox_broad", "sum"),
        numeric_only_suspect_count=("bbox_numeric_only_suspect", "sum"),
        coordinate_token_mean=("coordinate_token_count", "mean"),
        coordinate_token_median=("coordinate_token_count", "median"),
        coordinate_token_max=("coordinate_token_count", "max"),
    ).reset_index()
    for col in ["stored_bbox", "bbox_strict", "bbox_broad", "numeric_only_suspect"]:
        bbox_by_prompt[f"{col}_rate"] = bbox_by_prompt[f"{col}_count"] / bbox_by_prompt["n_cases"]
    bbox_by_prompt.to_csv(table_dir / "bbox_by_prompt.csv", index=False)

    comparison_metrics = [
        "entropy_norm_mean",
        "top_5_mass_mean",
        "effective_area_norm_mean",
        "hhi_mean",
        "region_count_thr090",
        "primary_region_mass_mean",
        "secondary_primary_ratio_mean_map",
        "mean_centroid_shift_vs_baseline",
        "layer_path_length_mean",
        "layer_tortuosity_mean",
        "word_path_length_mean",
        "word_tortuosity_mean",
    ]
    rows = []
    groups = {
        "bbox_strict_vs_nonbbox": df["bbox_strict"].astype(bool),
        "bbox_broad_vs_nonbbox": df["bbox_broad"].astype(bool),
        "expected_localization_prompt_vs_other": df["bbox_expected_prompt"].astype(bool),
    }
    for label, mask in groups.items():
        for metric in [m for m in comparison_metrics if m in df.columns]:
            a = pd.to_numeric(df.loc[mask, metric], errors="coerce")
            b = pd.to_numeric(df.loc[~mask, metric], errors="coerce")
            rows.append(
                {
                    "comparison": label,
                    "metric": metric,
                    "group_true_n": int(a.notna().sum()),
                    "group_false_n": int(b.notna().sum()),
                    "group_true_mean": float(a.mean()) if a.notna().any() else np.nan,
                    "group_false_mean": float(b.mean()) if b.notna().any() else np.nan,
                    "mean_difference_true_minus_false": float(a.mean() - b.mean()) if a.notna().any() and b.notna().any() else np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(table_dir / "bbox_metric_comparison.csv", index=False)

    stat_tests = run_stat_tests(df)
    stat_tests.to_csv(table_dir / "stat_tests_core.csv", index=False)

    sep = run_prompt_separability(df, table_dir)
    cluster_summary = run_clustering(df, table_dir)
    representatives = representative_cases(df)
    representatives.to_csv(table_dir / "representative_cases.csv", index=False)

    return {
        "prompt_summary_rows": int(len(prompt_summary)),
        "image_sensitivity_rows": int(len(image_rank)),
        "bbox_by_prompt_rows": int(len(bbox_by_prompt)),
        "stat_tests_rows": int(len(stat_tests)),
        "separability": sep,
        "clustering": cluster_summary,
        "representative_rows": int(len(representatives)),
    }


def run_stat_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "visual_change",
        "entropy_mean_delta_vs_baseline",
        "top5_mass_mean_delta_vs_baseline",
        "effective_area_norm_mean_delta_vs_baseline",
        "mean_centroid_shift_vs_baseline",
        "peak_count_mean_delta_vs_baseline",
        "weak_grounding_candidate_score",
        "multipeak_ambiguity_score",
    ]
    try:
        from scipy import stats
    except Exception as exc:
        return pd.DataFrame([{"test": "scipy_import", "status": "failed", "error": str(exc)}])

    try:
        import statsmodels.formula.api as smf
    except Exception:
        smf = None

    for metric in [m for m in metrics if m in df.columns]:
        wide = df.pivot(index="image_id", columns="prompt_label", values=metric)
        wide = wide[[p for p in PROMPT_ORDER if p in wide.columns]].dropna()
        if len(wide) > 0 and wide.shape[1] >= 3:
            stat, p = stats.friedmanchisquare(*[wide[col].to_numpy() for col in wide.columns])
            rows.append({"metric": metric, "test": "friedman_repeated_measures", "n_images": int(len(wide)), "statistic": float(stat), "p_value": float(p), "status": "ok"})
        baseline = "baseline_neutral"
        if baseline in wide.columns:
            for prompt in [c for c in wide.columns if c != baseline]:
                diff = wide[prompt] - wide[baseline]
                if diff.notna().sum() > 1:
                    stat, p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                    rows.append(
                        {
                            "metric": metric,
                            "test": "paired_wilcoxon_vs_baseline",
                            "contrast": f"{prompt} - {baseline}",
                            "n_images": int(diff.notna().sum()),
                            "statistic": float(stat),
                            "p_value": float(p),
                            "mean_difference": float(diff.mean()),
                            "median_difference": float(diff.median()),
                            "status": "ok",
                        }
                    )
        if smf is not None:
            try:
                model_df = df[["image_id", "prompt_label", metric]].dropna().copy()
                model_df[metric] = pd.to_numeric(model_df[metric], errors="coerce")
                model_df = model_df.dropna()
                if model_df["image_id"].nunique() > 10 and model_df["prompt_label"].nunique() > 2:
                    model = smf.mixedlm(f"{metric} ~ C(prompt_label)", model_df, groups=model_df["image_id"]).fit(reml=False, method="lbfgs", disp=False)
                    rows.append(
                        {
                            "metric": metric,
                            "test": "mixedlm_prompt_random_intercept_image",
                            "n_cases": int(len(model_df)),
                            "n_images": int(model_df["image_id"].nunique()),
                            "aic": float(model.aic) if np.isfinite(model.aic) else np.nan,
                            "llf": float(model.llf),
                            "status": "ok",
                        }
                    )
            except Exception as exc:
                rows.append({"metric": metric, "test": "mixedlm_prompt_random_intercept_image", "status": "failed", "error": str(exc)[:240]})

    if smf is not None:
        try:
            import statsmodels.api as sm

            gee_df = df[["image_id", "prompt_label", "bbox_strict"]].dropna().copy()
            gee_df["bbox_strict"] = gee_df["bbox_strict"].astype(int)
            model = smf.gee("bbox_strict ~ C(prompt_label)", groups="image_id", data=gee_df, family=sm.families.Binomial()).fit()
            rows.append(
                {
                    "metric": "bbox_strict",
                    "test": "gee_binomial_prompt_clustered_by_image",
                    "n_cases": int(len(gee_df)),
                    "n_images": int(gee_df["image_id"].nunique()),
                    "qic": float(model.qic()[0]) if hasattr(model, "qic") else np.nan,
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append({"metric": "bbox_strict", "test": "gee_binomial_prompt_clustered_by_image", "status": "failed", "error": str(exc)[:240]})

    out = pd.DataFrame(rows)
    if "p_value" in out.columns:
        out["p_value_fdr_bh"] = bh_fdr(out["p_value"].tolist())
    return out


def separability_features(df: pd.DataFrame, include_text: bool) -> list[str]:
    tam = [
        "visual_change",
        "entropy_norm_mean",
        "entropy_norm_iqr",
        "top_5_mass_mean",
        "top_5_mass_iqr",
        "effective_area_norm_mean",
        "effective_area_norm_iqr",
        "hhi_mean",
        "hoyer_sparsity_mean",
        "spread_trace_mean",
        "centroid_x_std",
        "centroid_y_std",
        "global_centroid_x_norm_range",
        "global_centroid_y_norm_range",
        "anisotropy_mean",
        "peak_count_mean_map",
        "secondary_primary_ratio_mean_map",
        "multipeak_proportion",
        "mean_centroid_shift_vs_baseline",
        "layer_path_length_mean",
        "layer_max_jump_mean",
        "layer_tortuosity_mean",
        "word_path_length_mean",
        "word_max_jump_mean",
        "word_tortuosity_mean",
        "weak_grounding_candidate_score",
        "multipeak_ambiguity_score",
    ]
    text = [
        "response_char_length",
        "response_word_length",
        "coordinate_token_count",
        "bbox_strict",
        "bbox_broad",
        "has_box_tokens",
        "has_object_ref_tokens",
        "content_jaccard_distance_vs_baseline",
        "first_divergence_ratio",
    ]
    cols = tam + (text if include_text else [])
    return [c for c in cols if c in df.columns and pd.to_numeric(df[c], errors="coerce").nunique(dropna=True) > 1]


def run_prompt_separability(df: pd.DataFrame, table_dir: Path) -> dict[str, object]:
    rows = []
    confusion_written = False
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return run_prompt_separability_numpy(df, table_dir, f"sklearn unavailable: {exc}")

    for variant, include_text in [("tam_only", False), ("tam_plus_text", True)]:
        features = separability_features(df, include_text)
        model_df = df.dropna(subset=["prompt_label", "image_id"]).copy()
        x = model_df[features].apply(pd.to_numeric, errors="coerce")
        med = x.median(numeric_only=True)
        x = x.fillna(med).fillna(0.0)
        y = model_df["prompt_label"].astype(str)
        groups = model_df["image_id"].astype(str)
        n_splits = min(5, groups.nunique())
        clf = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=250, min_samples_leaf=3, random_state=RANDOM_SEED, class_weight="balanced_subsample"),
        )
        cv = GroupKFold(n_splits=n_splits)
        pred = cross_val_predict(clf, x, y, groups=groups, cv=cv)
        bal = balanced_accuracy_score(y, pred)
        f1 = f1_score(y, pred, average="macro")
        rows.append(
            {
                "variant": variant,
                "n_cases": int(len(model_df)),
                "n_images": int(groups.nunique()),
                "n_features": int(len(features)),
                "cv": f"GroupKFold({n_splits}) by image_id",
                "chance_balanced_accuracy": 0.125,
                "balanced_accuracy": float(bal),
                "macro_f1": float(f1),
                "features": ";".join(features),
                "status": "ok",
            }
        )
        if not confusion_written:
            labels = [p for p in PROMPT_ORDER if p in set(y)]
            cm = confusion_matrix(y, pred, labels=labels)
            pd.DataFrame(cm, index=labels, columns=labels).to_csv(table_dir / "prompt_confusion_matrix.csv")
            confusion_written = True
    out = pd.DataFrame(rows)
    out.to_csv(table_dir / "prompt_separability_scores.csv", index=False)
    best = out.sort_values("balanced_accuracy", ascending=False).iloc[0].to_dict() if not out.empty else {}
    return {"status": "ok", "rows": rows, "best": best}


def make_group_folds(groups: pd.Series, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.array(sorted(groups.astype(str).unique()))
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(unique)
    folds = np.array_split(unique, min(n_splits, len(unique)))
    out = []
    g = groups.astype(str).to_numpy()
    for fold in folds:
        test_mask = np.isin(g, fold)
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) and len(test_idx):
            out.append((train_idx, test_idx))
    return out


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> tuple[float, float, np.ndarray]:
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_i = {label: i for i, label in enumerate(labels)}
    for truth, pred in zip(y_true, y_pred):
        if truth in label_to_i and pred in label_to_i:
            cm[label_to_i[truth], label_to_i[pred]] += 1
    recalls = []
    f1s = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return float(np.mean(recalls)), float(np.mean(f1s)), cm


def nearest_centroid_group_predict(x: pd.DataFrame, y: pd.Series, groups: pd.Series) -> np.ndarray:
    labels = [p for p in PROMPT_ORDER if p in set(y.astype(str))]
    x_np = x.to_numpy(dtype=float)
    y_np = y.astype(str).to_numpy()
    pred = np.empty(len(y_np), dtype=object)
    for train_idx, test_idx in make_group_folds(groups, n_splits=5):
        train = x_np[train_idx]
        test = x_np[test_idx]
        med = np.nanmedian(train, axis=0)
        train = np.where(np.isfinite(train), train, med)
        test = np.where(np.isfinite(test), test, med)
        mean = train.mean(axis=0)
        std = train.std(axis=0)
        std[~np.isfinite(std) | (std == 0)] = 1.0
        train_z = (train - mean) / std
        test_z = (test - mean) / std
        centroids = []
        centroid_labels = []
        for label in labels:
            mask = y_np[train_idx] == label
            if mask.any():
                centroids.append(train_z[mask].mean(axis=0))
                centroid_labels.append(label)
        c = np.vstack(centroids)
        dists = ((test_z[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
        pred[test_idx] = np.array(centroid_labels, dtype=object)[np.argmin(dists, axis=1)]
    return pred


def run_prompt_separability_numpy(df: pd.DataFrame, table_dir: Path, note: str) -> dict[str, object]:
    rows = []
    best_cm = None
    best_labels = [p for p in PROMPT_ORDER if p in set(df["prompt_label"].astype(str))]
    for variant, include_text in [("tam_only_nearest_centroid", False), ("tam_plus_text_nearest_centroid", True)]:
        features = separability_features(df, include_text)
        model_df = df.dropna(subset=["prompt_label", "image_id"]).copy()
        x = model_df[features].apply(pd.to_numeric, errors="coerce")
        y = model_df["prompt_label"].astype(str)
        groups = model_df["image_id"].astype(str)
        pred = nearest_centroid_group_predict(x, y, groups)
        bal, f1, cm = classification_scores(y.to_numpy(), pred, best_labels)
        row = {
            "variant": variant,
            "n_cases": int(len(model_df)),
            "n_images": int(groups.nunique()),
            "n_features": int(len(features)),
            "cv": "manual GroupKFold(5) by image_id",
            "chance_balanced_accuracy": 0.125,
            "balanced_accuracy": bal,
            "macro_f1": f1,
            "features": ";".join(features),
            "status": "ok",
            "note": note,
        }
        rows.append(row)
        if best_cm is None or bal > max(r["balanced_accuracy"] for r in rows[:-1]):
            best_cm = cm
    if best_cm is not None:
        pd.DataFrame(best_cm, index=best_labels, columns=best_labels).to_csv(table_dir / "prompt_confusion_matrix.csv")
    pd.DataFrame(rows).to_csv(table_dir / "prompt_separability_scores.csv", index=False)
    best = max(rows, key=lambda r: r["balanced_accuracy"]) if rows else {}
    return {"status": "ok", "fallback": "numpy_nearest_centroid", "rows": rows, "best": best}


def run_clustering(df: pd.DataFrame, table_dir: Path) -> dict[str, object]:
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except Exception as exc:
        return run_clustering_numpy(df, table_dir, f"sklearn unavailable: {exc}")

    features = separability_features(df, include_text=False)[:25]
    x, params = robust_scale(df[features].apply(pd.to_numeric, errors="coerce"))
    rows = []
    for k in [3, 4, 5, 6]:
        km = KMeans(n_clusters=k, n_init=25, random_state=RANDOM_SEED)
        labels = km.fit_predict(x)
        sil = silhouette_score(x, labels) if len(set(labels)) > 1 else np.nan
        rows.append({"clustering": "case_kmeans", "k": k, "silhouette": float(sil), "inertia": float(km.inertia_), "features": ";".join(features), "status": "ok"})
    best_k = int(max(rows, key=lambda r: r["silhouette"] if np.isfinite(r["silhouette"]) else -999)["k"])
    km = KMeans(n_clusters=best_k, n_init=25, random_state=RANDOM_SEED)
    labels = km.fit_predict(x)
    cluster_df = df[["case_id", "image_id", "prompt_label", "dashboard_case_url"]].copy()
    cluster_df["case_cluster"] = labels
    cluster_df.to_csv(table_dir / "case_clusters.csv", index=False)
    comp = cluster_df.groupby(["case_cluster", "prompt_label"], observed=False).size().rename("n").reset_index()
    comp.to_csv(table_dir / "case_cluster_prompt_composition.csv", index=False)
    pd.DataFrame(rows).to_csv(table_dir / "clustering_summary.csv", index=False)
    (table_dir / "clustering_feature_scaling.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    return {"status": "ok", "best_k": best_k, "features": features, "summary_rows": rows}


def simple_kmeans(x: np.ndarray, k: int, n_iter: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(RANDOM_SEED + k)
    centers = x[rng.choice(np.arange(len(x)), size=k, replace=False)].copy()
    labels = np.zeros(len(x), dtype=int)
    for _ in range(n_iter):
        dists = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            if np.any(labels == i):
                centers[i] = x[labels == i].mean(axis=0)
    inertia = float(((x - centers[labels]) ** 2).sum())
    return labels, centers, inertia


def sampled_silhouette(x: np.ndarray, labels: np.ndarray, max_n: int = 400) -> float:
    if len(set(labels)) < 2:
        return np.nan
    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.arange(len(x))
    if len(idx) > max_n:
        idx = rng.choice(idx, size=max_n, replace=False)
    xs = x[idx]
    labs = labels[idx]
    d = np.sqrt(((xs[:, None, :] - xs[None, :, :]) ** 2).sum(axis=2))
    vals = []
    for i, lab in enumerate(labs):
        same = labs == lab
        other_labels = [l for l in set(labs) if l != lab]
        a = float(d[i, same].sum() / max(same.sum() - 1, 1))
        b = min(float(d[i, labs == other].mean()) for other in other_labels)
        vals.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return float(np.mean(vals))


def run_clustering_numpy(df: pd.DataFrame, table_dir: Path, note: str) -> dict[str, object]:
    features = separability_features(df, include_text=False)[:25]
    x, params = robust_scale(df[features].apply(pd.to_numeric, errors="coerce"))
    rows = []
    best = None
    for k in [3, 4, 5, 6]:
        labels, _, inertia = simple_kmeans(x, k)
        sil = sampled_silhouette(x, labels)
        row = {"clustering": "case_kmeans_numpy", "k": k, "silhouette": sil, "inertia": inertia, "features": ";".join(features), "status": "ok", "note": note}
        rows.append(row)
        if best is None or sil > best["silhouette"]:
            best = row
    best_k = int(best["k"]) if best else 4
    labels, _, _ = simple_kmeans(x, best_k)
    cluster_df = df[["case_id", "image_id", "prompt_label", "dashboard_case_url"]].copy()
    cluster_df["case_cluster"] = labels
    cluster_df.to_csv(table_dir / "case_clusters.csv", index=False)
    comp = cluster_df.groupby(["case_cluster", "prompt_label"], observed=False).size().rename("n").reset_index()
    comp.to_csv(table_dir / "case_cluster_prompt_composition.csv", index=False)
    pd.DataFrame(rows).to_csv(table_dir / "clustering_summary.csv", index=False)
    (table_dir / "clustering_feature_scaling.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    return {"status": "ok", "fallback": "numpy_kmeans", "best_k": best_k, "features": features, "summary_rows": rows}


def representative_cases(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("highest_visual_change", "visual_change", False),
        ("highest_centroid_shift", "mean_centroid_shift_vs_baseline", False),
        ("strongest_entropy_increase", "entropy_mean_delta_vs_baseline", False),
        ("strongest_concentration_increase", "top5_mass_mean_delta_vs_baseline", False),
        ("high_weak_grounding_score", "weak_grounding_candidate_score", False),
        ("high_multipeak_ambiguity", "multipeak_ambiguity_score", False),
        ("stable_cases", "visual_change", True),
    ]
    parts = []
    for label, metric, asc in specs:
        if metric not in df.columns:
            continue
        sub = df.copy()
        if label == "stable_cases":
            sub = sub.loc[~sub["prompt_label"].astype(str).eq("baseline_neutral")]
        sub = sub.sort_values(metric, ascending=asc).head(15)
        sub = sub.assign(selection_reason=label, selection_metric=metric)
        parts.append(sub)
    if "bbox_strict" in df.columns:
        for label, metric, asc in [
            ("bbox_like_concentrated_cases", "top_5_mass_mean", False),
            ("bbox_like_diffuse_cases", "entropy_norm_mean", False),
        ]:
            sub = df.loc[df["bbox_strict"].astype(bool)].sort_values(metric, ascending=asc).head(15)
            sub = sub.assign(selection_reason=label, selection_metric=metric)
            parts.append(sub)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    cols = [
        "selection_reason",
        "selection_metric",
        "case_id",
        "image_id",
        "prompt_label",
        "response_preview",
        "visual_change",
        "mean_centroid_shift_vs_baseline",
        "entropy_mean_delta_vs_baseline",
        "top5_mass_mean_delta_vs_baseline",
        "weak_grounding_candidate_score",
        "multipeak_ambiguity_score",
        "bbox_strict",
        "bbox_broad",
        "dashboard_case_url",
        "dashboard_matrix_url",
        "dashboard_compare_url",
    ]
    return out[[c for c in cols if c in out.columns]].drop_duplicates(["selection_reason", "case_id"])


def plot_bar(df: pd.DataFrame, x: str, ys: list[str], path: Path, title: str, ylabel: str = "count") -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    xpos = np.arange(len(df))
    width = 0.8 / max(len(ys), 1)
    for i, y in enumerate(ys):
        ax.bar(xpos + i * width - (len(ys) - 1) * width / 2, df[y], width=width, label=y)
    ax.set_xticks(xpos)
    ax.set_xticklabels(df[x].astype(str), rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_box(df: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    data = [pd.to_numeric(df.loc[df["prompt_label"].astype(str).eq(prompt), metric], errors="coerce").dropna().to_numpy() for prompt in PROMPT_ORDER]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(data, tick_labels=PROMPT_ORDER, showfliers=False)
    ax.set_xticklabels(PROMPT_ORDER, rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_box_by_group(df: pd.DataFrame, metric: str, group_col: str, path: Path, title: str) -> None:
    labels = [str(v) for v in sorted(df[group_col].dropna().unique())]
    data = [pd.to_numeric(df.loc[df[group_col].astype(str).eq(label), metric], errors="coerce").dropna().to_numpy() for label in labels]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xlabel(group_col)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    if metric not in df.columns:
        return
    mat = df.pivot(index="image_id", columns="prompt_label", values=metric)
    mat = mat[[p for p in PROMPT_ORDER if p in mat.columns]]
    order_metric = mat.abs().mean(axis=1).sort_values(ascending=False).index
    mat = mat.loc[order_metric]
    fig_h = max(8, min(22, 0.12 * len(mat) + 4))
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns.astype(str), rotation=35, ha="right")
    step = max(1, len(mat) // 25)
    ax.set_yticks(np.arange(0, len(mat), step))
    ax.set_yticklabels(mat.index.astype(str)[::step])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_plots(df: pd.DataFrame, out_dir: Path) -> list[str]:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    cases_by_prompt = df.groupby("prompt_label", observed=False).size().rename("cases").reset_index()
    plot_bar(cases_by_prompt, "prompt_label", ["cases"], plot_dir / "dataset_cases_by_prompt.png", "Cases by prompt")
    written.append("dataset_cases_by_prompt.png")

    quality = pd.DataFrame(
        {
            "flag": ["used_fix256", "missing_map_metrics_cases", "non_ok_status", "old_is_official_subset"],
            "count": [
                int(pd.to_numeric(df["used_fix256"], errors="coerce").fillna(0).sum()),
                int(df["missing_map_metrics_count"].gt(0).sum()),
                int(~df["status"].eq("ok").sum()) if False else int((~df["status"].eq("ok")).sum()),
                int(pd.to_numeric(df["is_official"], errors="coerce").fillna(0).sum()),
            ],
        }
    )
    plot_bar(quality, "flag", ["count"], plot_dir / "dataset_quality_flags.png", "Quality flags and known caveats")
    written.append("dataset_quality_flags.png")

    bbox_by_prompt = df.groupby("prompt_label", observed=False).agg(
        stored_bbox=("bbox_style_output_flag", "sum"),
        bbox_strict=("bbox_strict", "sum"),
        bbox_broad=("bbox_broad", "sum"),
        numeric_suspect=("bbox_numeric_only_suspect", "sum"),
    ).reset_index()
    plot_bar(bbox_by_prompt, "prompt_label", ["stored_bbox", "bbox_strict", "bbox_broad"], plot_dir / "bbox_count_by_prompt.png", "BBox/location-style count by prompt")
    written.append("bbox_count_by_prompt.png")
    plot_bar(bbox_by_prompt, "prompt_label", ["bbox_strict", "bbox_broad", "numeric_suspect"], plot_dir / "bbox_strict_vs_broad_by_prompt.png", "Strict vs broad bbox/location flags")
    written.append("bbox_strict_vs_broad_by_prompt.png")

    for metric in ["entropy_norm_mean", "top_5_mass_mean", "effective_area_norm_mean", "mean_centroid_shift_vs_baseline", "layer_path_length_mean"]:
        if metric in df.columns:
            plot_box_by_group(
                df.assign(bbox_group=np.where(df["bbox_strict"].astype(bool), "bbox_strict", "non_bbox")),
                metric,
                "bbox_group",
                plot_dir / f"bbox_vs_nonbbox_{metric}.png",
                f"{metric}: bbox-style vs non-bbox distribution",
            )
            written.append(f"bbox_vs_nonbbox_{metric}.png")

    heatmaps = [
        ("visual_change", "Image x prompt heatmap: visual_change"),
        ("mean_centroid_shift_vs_baseline", "Image x prompt heatmap: centroid shift vs baseline"),
        ("entropy_mean_delta_vs_baseline", "Image x prompt heatmap: entropy delta vs baseline"),
        ("top5_mass_mean_delta_vs_baseline", "Image x prompt heatmap: top-5 mass delta vs baseline"),
        ("bbox_strict", "Image x prompt heatmap: bbox strict flag"),
        ("weak_grounding_candidate_score", "Image x prompt heatmap: weak grounding proxy score"),
    ]
    for metric, title in heatmaps:
        path = plot_dir / f"heatmap_{metric}.png"
        plot_heatmap(df, metric, path, title)
        if path.exists():
            written.append(path.name)

    for metric in [
        "visual_change",
        "entropy_mean_delta_vs_baseline",
        "top5_mass_mean_delta_vs_baseline",
        "mean_centroid_shift_vs_baseline",
        "weak_grounding_candidate_score",
    ]:
        if metric in df.columns:
            path = plot_dir / f"boxplot_{metric}_by_prompt.png"
            plot_box(df, metric, path, f"{metric} by prompt")
            written.append(path.name)

    ranking = pd.read_csv(out_dir / "tables" / "image_sensitivity_ranking.csv")
    top = ranking.sort_values("sensitivity_rank").head(20).sort_values("mean_abs_visual_delta")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["image_id"].astype(str), top["mean_abs_visual_delta"])
    ax.set_title("Top 20 prompt-sensitive images")
    ax.set_xlabel("mean visual_change across prompts")
    fig.tight_layout()
    fig.savefig(plot_dir / "top20_prompt_sensitive_images.png", dpi=160)
    plt.close(fig)
    written.append("top20_prompt_sensitive_images.png")

    stable = ranking.sort_values("stability_rank").head(20).sort_values("mean_abs_visual_delta", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(stable["image_id"].astype(str), stable["mean_abs_visual_delta"])
    ax.set_title("Top 20 stable images")
    ax.set_xlabel("mean visual_change across prompts")
    fig.tight_layout()
    fig.savefig(plot_dir / "top20_stable_images.png", dpi=160)
    plt.close(fig)
    written.append("top20_stable_images.png")

    features = separability_features(df, include_text=False)
    x, _ = robust_scale(df[features].apply(pd.to_numeric, errors="coerce"))
    scores, ratio = pca_scores(x, 2)
    pca_df = pd.DataFrame({"pc1": scores[:, 0], "pc2": scores[:, 1], "prompt_label": df["prompt_label"].astype(str), "bbox_strict": df["bbox_strict"].astype(int)})
    fig, ax = plt.subplots(figsize=(8, 6))
    for prompt in PROMPT_ORDER:
        sub = pca_df.loc[pca_df["prompt_label"].eq(prompt)]
        ax.scatter(sub["pc1"], sub["pc2"], s=16, alpha=0.75, label=prompt)
    ax.set_title("Exploratory PCA of TAM metric features by prompt")
    ax.set_xlabel(f"PC1 ({ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ratio[1]*100:.1f}%)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "pca_cases_by_prompt.png", dpi=160)
    plt.close(fig)
    written.append("pca_cases_by_prompt.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    for flag, label in [(0, "non-bbox"), (1, "bbox_strict")]:
        sub = pca_df.loc[pca_df["bbox_strict"].eq(flag)]
        ax.scatter(sub["pc1"], sub["pc2"], s=18, alpha=0.7, label=label)
    ax.set_title("Exploratory PCA of TAM metric features by bbox flag")
    ax.set_xlabel(f"PC1 ({ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ratio[1]*100:.1f}%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "pca_cases_by_bbox.png", dpi=160)
    plt.close(fig)
    written.append("pca_cases_by_bbox.png")

    cm_path = out_dir / "tables" / "prompt_confusion_matrix.csv"
    if cm_path.exists():
        cm = pd.read_csv(cm_path, index_col=0)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm.to_numpy(), cmap="Blues")
        ax.set_xticks(np.arange(len(cm.columns)))
        ax.set_yticks(np.arange(len(cm.index)))
        ax.set_xticklabels(cm.columns, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(cm.index, fontsize=8)
        ax.set_title("Prompt classifier confusion matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(int(cm.iloc[i, j])), ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(plot_dir / "prompt_classifier_confusion_matrix.png", dpi=160)
        plt.close(fig)
        written.append("prompt_classifier_confusion_matrix.png")

    return written


def write_representative_markdown(table_dir: Path) -> None:
    csv_path = table_dir / "representative_cases.csv"
    if not csv_path.exists():
        return
    reps = pd.read_csv(csv_path)
    lines = ["# Representative Case Gallery", "", "Dashboard links point to existing inspection pages; selections are metric-based examples, not ground truth labels.", ""]
    for reason, group in reps.groupby("selection_reason"):
        lines.extend([f"## {reason}", ""])
        lines.append("| case_id | image_id | prompt | metric | visual_change | link |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for _, row in group.head(10).iterrows():
            link = row.get("dashboard_case_url", "")
            lines.append(
                f"| {row.get('case_id', '')} | {row.get('image_id', '')} | `{row.get('prompt_label', '')}` | "
                f"`{row.get('selection_metric', '')}` | {float(row.get('visual_change', np.nan)):.4f} | [case]({link}) |"
            )
        lines.append("")
    (table_dir / "representative_case_gallery.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(out_dir: Path, meta: dict[str, object], table_meta: dict[str, object], plots: list[str]) -> None:
    sep = table_meta.get("separability", {})
    best = sep.get("best", {}) if isinstance(sep, dict) else {}
    bbox = meta
    lines = [
        "# Horizontal Analysis Report v1",
        "",
        "Scientific framing: repeated-measures analysis of TAM-derived attribution behavior across prompt conditions, using precomputed saliency-map descriptors and baseline-relative visual sensitivity metrics.",
        "",
        "## 1. Dataset completeness and caveats",
        "",
        f"- Cases: {meta['case_rows']} rows; valid full-800 include flag: {meta['valid_case_rows']}.",
        f"- Maps: {meta['map_count_total']}; map metrics: {meta['map_metrics_count_total']}; missing map metrics: {meta['missing_map_metrics_total']}.",
        "- `is_official` is retained only as a historical subset flag and is not used as the full dataset filter.",
        "- TAM scanpaths are attribution-derived trajectories, not human eye-tracking.",
        "- Diagnostic scores are heuristic proxy rankings, not causal proof.",
        "- No COCO ground-truth localization, deletion/insertion faithfulness, EMD, or full raw-map pairwise similarity is computed here.",
        "",
        "## 2. Construction of `case_features_800_v2`",
        "",
        "The feature table has one row per image x prompt case. It joins case metadata, output diagnostics, baseline-relative visual sensitivity, proxy diagnostic scores, dashboard links, DB map coverage counts, case-level aggregates from map metrics, compact region summaries, layer scanpaths, word scanpaths, and selected token-category summaries.",
        "",
        "BBox/location fields are descriptive text-output flags. `bbox_strict` keeps the stored bbox-style flag and clear coordinate/box-like regex hits. `bbox_broad` adds coordinate tokens, box/location tokens, and generic coordinate-like patterns. `bbox_numeric_only_suspect` marks broad hits likely caused by ordinary numbers such as years, IDs, route/train numbers, times, or display values.",
        "",
        "## 3. Prompt effects on attribution metrics",
        "",
        "See `tables/prompt_summary.csv` and prompt-level boxplots. These summarize prompt-associated changes in concentration, diffusion, centroid shift, hotspot structure, and proxy scores. They should be interpreted as attribution-behavior differences, not proof of better grounding.",
        "",
        "## 4. Image-level prompt sensitivity",
        "",
        "See `tables/image_sensitivity_ranking.csv` and top-20 plots. The ranking identifies images with larger or smaller prompt-associated attribution changes across the eight prompt conditions.",
        "",
        "## 5. Prompt separability from TAM metrics",
        "",
        f"- Best computed variant: `{best.get('variant', 'n/a')}`.",
        f"- Balanced accuracy: {best.get('balanced_accuracy', np.nan):.4f} against chance 0.125." if best else "- Prompt separability did not run.",
        f"- Macro-F1: {best.get('macro_f1', np.nan):.4f}." if best else "",
        "- Cross-validation uses GroupKFold by image_id so the same image does not appear in both train and test.",
        "- This measures prompt separability in metric space only.",
        "",
        "## 6. BBox/location-style output analysis",
        "",
        f"- Stored bbox-style count: {bbox['bbox_style_output_flag_count']}.",
        f"- Strict bbox/location count: {bbox['bbox_strict_count']}.",
        f"- Broad bbox/location count: {bbox['bbox_broad_count']}.",
        f"- Numeric-only suspect count: {bbox['bbox_numeric_only_suspect_count']}.",
        "See `tables/bbox_by_prompt.csv` and `tables/bbox_metric_comparison.csv`. These compare output style against TAM concentration, diffusion, centroid shift, region, and scanpath descriptors without claiming localization correctness.",
        "",
        "## 7. Clustering and representative cases",
        "",
        "Clustering is exploratory and uses robust-standardized TAM metric features. See `tables/clustering_summary.csv`, `tables/case_clusters.csv`, and `tables/representative_cases.csv`.",
        "",
        "## 8. Limitations",
        "",
        "- Visual inspection and metric clustering do not establish causal faithfulness.",
        "- PCA/clustering are exploratory summaries and can be unstable under feature choices.",
        "- BBox/location-style text does not imply correct localization.",
        "- The three known missing map metrics remain visible as coverage caveats if not repaired upstream.",
        "- Some cases used fix256 preprocessing; the flag is retained for sensitivity filtering.",
        "",
        "## 9. Next steps",
        "",
        "- Use `case_features_800_v2` as the stable table for Final report plots and discussion.",
        "- Add a small dashboard or static HTML index for v6 only after the report figures are reviewed.",
        "- Future heavier work: COCO bbox/segmentation evaluation, deletion/insertion perturbation tests, EMD/raw-map similarity, and semantic image grouping with COCO labels or CLIP embeddings.",
        "",
        "## Generated plots",
        "",
    ]
    lines.extend(f"- `plots/{p}`" for p in plots)
    (out_dir / "horizontal_analysis_report_v1.md").write_text("\n".join([line for line in lines if line != ""]), encoding="utf-8")


def git_status_summary() -> str:
    try:
        result = subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
        return result.stdout.strip()
    except Exception as exc:
        return f"git status unavailable: {exc}"


def write_manifest(out_dir: Path, tables: dict[str, pd.DataFrame], case_features: pd.DataFrame, meta: dict[str, object], table_meta: dict[str, object], plots: list[str], archive_root: Path, db_path: Path) -> None:
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_name": "v6_horizontal_analysis",
        "scientific_framing": "Repeated-measures analysis of TAM-derived attribution behavior across prompt conditions using precomputed descriptors.",
        "input_archive_root": str(archive_root),
        "input_db_path": str(db_path),
        "input_tables": {name: {"rows": int(len(df)), "columns": list(df.columns)} for name, df in tables.items()},
        "outputs": {
            "case_features_parquet": "case_features_800_v2.parquet",
            "case_features_csv": "case_features_800_v2.csv",
            "tables_dir": "tables/",
            "plots_dir": "plots/",
            "report": "horizontal_analysis_report_v1.md",
        },
        "row_counts": {
            "expected_cases": 800,
            "actual_cases": int(len(case_features)),
            "distinct_images": int(case_features["image_id"].nunique()),
            "distinct_prompts": int(case_features["prompt_label"].nunique()),
            **meta,
        },
        "selected_metrics": {
            "case_level": ["visual_change", "textual_change", "entropy/top5/effective_area aggregates", "centroid/spread", "regions", "scanpaths", "diagnostic proxy scores"],
            "statistical_tests": ["Friedman repeated-measures", "paired Wilcoxon vs baseline", "MixedLM where converged", "GEE binomial for bbox where feasible"],
            "separability": "RandomForest classifier with GroupKFold by image_id; TAM-only and TAM+text variants",
            "clustering": "KMeans on robust-standardized TAM metrics",
        },
        "skipped_metrics": {
            "raw_map_recomputation": "not requested; existing exported tables only",
            "emd": "explicitly skipped except pre-existing archive columns, no bulk computation",
            "full_pairwise_similarity": "explicitly skipped",
            "coco_ground_truth_bbox_segmentation": "not implemented in v6",
            "causal_deletion_insertion": "not implemented in v6",
        },
        "quality_caveats": [
            "Three known map rows lack map_metrics due unreadable/truncated raw maps unless repaired upstream.",
            "`is_official=1` marks an old subset only and is not a full-dataset filter.",
            "`used_fix256` is retained as a quality/preprocessing flag.",
            "BBox/location-style output flags are regex/token based and do not establish localization correctness.",
            "TAM scanpaths are attribution-derived trajectories, not human gaze.",
            "Diagnostic scores are proxy rankings, not causal proof.",
        ],
        "bbox_flag_definitions": {
            "bbox_style_output_flag": "stored archive flag from output diagnostics",
            "bbox_strict": "stored flag OR explicit box/location/coordinate term with coordinate-like format",
            "bbox_broad": "strict OR coordinate token count OR box/location token OR generic coordinate-like regex",
            "bbox_numeric_only_suspect": "broad but not strict and text matches ordinary-number patterns such as years, IDs, routes, times, or units",
            "bbox_expected_prompt": f"prompt in {sorted(EXPECTED_BBOX_PROMPTS)}",
        },
        "table_generation": table_meta,
        "plots_generated": plots,
        "git_status_short": git_status_summary(),
        "reproducibility_notes": [
            "Regenerate with `.\\.venv\\Scripts\\python.exe -m scripts.analysis.build_v6_horizontal_analysis`.",
            "Uses archive Parquet files when available and the dashboard DB only for per-case map/map_metric counts.",
            "Does not modify DB/cache/raw maps/notebooks.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plots").mkdir(parents=True, exist_ok=True)

    names = [
        "cases",
        "dashboard_links",
        "diagnostic_scores",
        "layer_scanpaths",
        "map_metrics_core",
        "output_diagnostics",
        "region_summary",
        "token_category_summary",
        "visual_sensitivity_vs_baseline",
        "word_scanpaths",
    ]
    tables = {name: read_table(args.archive_root, name) for name in names}
    case_features, meta = build_case_features(tables, args.db_path, args.v1_dir)
    write_dataframe(case_features, args.output_dir / "case_features_800_v2")
    table_meta = build_tables(case_features, args.output_dir)
    write_representative_markdown(args.output_dir / "tables")
    plots = build_plots(case_features, args.output_dir)
    write_report(args.output_dir, meta, table_meta, plots)
    write_manifest(args.output_dir, tables, case_features, meta, table_meta, plots, args.archive_root, args.db_path)
    print(json.dumps({"case_features_rows": int(len(case_features)), "metadata": meta, "tables": table_meta, "plots": len(plots)}, indent=2, default=str))


if __name__ == "__main__":
    main()
