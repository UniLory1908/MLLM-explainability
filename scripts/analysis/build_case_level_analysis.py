from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE_ROOT = PROJECT_ROOT / "outputs" / "statistical_archive" / "stat_timebox_20260523_progress"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "v1_case_level"

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

EXPECTED_PROMPTS = list(PROMPT_CATEGORY)
KEYS = ["case_id"]
ID_COLS = {"image_id", "condition_id", "prompt_id", "is_official", "token_count", "word_count", "layer_count"}
TEXT_COLS = {
    "case_id",
    "condition_id",
    "condition_label",
    "prompt_label",
    "prompt_category",
    "run_name",
    "prompt_text",
    "response_text",
    "metadata_path",
    "image_path",
    "dashboard_case_url",
    "dashboard_matrix_url",
    "dashboard_compare_url",
    "dashboard_word_url",
    "formula_version",
    "alignment_method",
    "notes",
    "limitations",
}
BOOL_OR_FLAG_COLS = {
    "is_baseline",
    "is_official",
    "all_layers",
    "used_fix256",
    "bbox_style_output_flag",
    "has_object_ref_tokens",
    "has_box_tokens",
    "exact_response_match_vs_baseline",
    "has_all_required_fields",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the v1 case-level TAM analysis dataset and first exploratory reports.",
    )
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=50)
    return parser


def read_table(parquet_dir: Path, name: str) -> pd.DataFrame:
    path = parquet_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def schema_info(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    return {
        name: {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
        for name, df in tables.items()
    }


def write_schema_snapshot(schema: dict[str, object], out_dir: Path) -> None:
    (out_dir / "schema_snapshot.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    lines = ["# Schema Snapshot", ""]
    for name, info in schema.items():
        lines.extend([f"## {name}", "", f"- rows: {info['rows']}", ""])
        lines.append("| column | dtype |")
        lines.append("| --- | --- |")
        for col in info["columns"]:
            lines.append(f"| `{col}` | `{info['dtypes'][col]}` |")
        lines.append("")
    (out_dir / "schema_snapshot.md").write_text("\n".join(lines), encoding="utf-8")


def merge_unique(left: pd.DataFrame, right: pd.DataFrame, suffix: str) -> pd.DataFrame:
    keep = KEYS + [col for col in right.columns if col not in left.columns and col not in KEYS]
    return left.merge(right[keep], on=KEYS, how="left", validate="one_to_one", indicator=suffix)


def robust_rank(series: pd.Series, *, invert: bool = False, absolute: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if absolute:
        s = s.abs()
    if invert:
        s = -s
    out = s.rank(pct=True, method="average")
    return out.fillna(0.0).clip(0.0, 1.0)


def mean_available(df: pd.DataFrame, specs: Iterable[tuple[str, bool, bool]]) -> pd.Series:
    parts = []
    for col, invert, absolute in specs:
        if col in df.columns:
            parts.append(robust_rank(df[col], invert=invert, absolute=absolute))
    if not parts:
        return pd.Series(0.0, index=df.index)
    return pd.concat(parts, axis=1).mean(axis=1).fillna(0.0).clip(0.0, 1.0)


def build_case_level(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, int]]:
    cases = tables["cases"].copy()
    output = tables["output_diagnostics"].copy()
    visual = tables["visual_sensitivity_vs_baseline"].copy()
    scores = tables["diagnostic_scores"].copy()
    links = tables["dashboard_links"].copy()

    row_presence = {
        "case_rows": int(len(cases)),
        "output_diagnostics_rows": int(len(output)),
        "visual_sensitivity_rows": int(len(visual)),
        "diagnostic_scores_rows": int(len(scores)),
        "dashboard_links_rows": int(len(links)),
    }

    df = cases.copy()
    df = merge_unique(df, output, "_output")
    df = merge_unique(df, visual, "_visual")
    df = merge_unique(df, scores, "_scores")
    df = merge_unique(df, links, "_links")

    df["prompt_category"] = df["prompt_label"].map(PROMPT_CATEGORY)
    df["is_baseline"] = df["prompt_label"].eq("baseline_neutral")
    df["has_all_required_fields"] = (
        df["_output"].eq("both")
        & df["_visual"].eq("both")
        & df["_scores"].eq("both")
        & df["_links"].eq("both")
        & df["prompt_category"].notna()
    )
    df["notes"] = ""
    df["limitations"] = (
        "Exploratory proxy metrics; baseline comparisons use baseline_neutral with positional/common-index alignment; "
        "no GT metric, causal faithfulness metric, semantic word alignment, bulk EMD, or raw all-vs-all pairwise comparison."
    )
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    return df, row_presence


def add_text_visual_axes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    non_base = ~out["is_baseline"].astype(bool)

    text_specs = [
        ("content_jaccard_distance_vs_baseline", False, False),
        ("response_length_delta_vs_baseline", False, True),
        ("response_length_ratio_vs_baseline", False, False),
        ("first_divergence_ratio", True, False),
        ("matched_word_coverage_vs_baseline", True, False),
    ]
    visual_specs = [
        ("mean_centroid_shift_vs_baseline", False, False),
        ("entropy_mean_delta_vs_baseline", False, True),
        ("top5_mass_mean_delta_vs_baseline", False, True),
        ("effective_area_norm_mean_delta_vs_baseline", False, True),
        ("spread_trace_mean_delta_vs_baseline", False, True),
        ("peak_count_mean_delta_vs_baseline", False, True),
        ("secondary_primary_ratio_mean_delta_vs_baseline", False, True),
        ("multipeak_ratio_delta_vs_baseline", False, True),
        ("layer_path_length_mean_delta_vs_baseline", False, True),
        ("layer_max_jump_mean_delta_vs_baseline", False, True),
        ("layer_tortuosity_mean_delta_vs_baseline", False, True),
        ("word_path_length_delta_vs_baseline", False, True),
        ("word_max_jump_delta_vs_baseline", False, True),
        ("word_tortuosity_delta_vs_baseline", False, True),
        ("mean_top5_iou_vs_baseline", True, False),
        ("mean_cosine_similarity_vs_baseline", True, False),
        ("mean_jsd_vs_baseline", False, False),
        ("unstable_explanation_candidate_score", False, False),
    ]

    out["textual_change"] = 0.0
    out["visual_change"] = 0.0
    out.loc[non_base, "textual_change"] = mean_available(out.loc[non_base], text_specs)
    out.loc[non_base, "visual_change"] = mean_available(out.loc[non_base], visual_specs)

    text_threshold = float(out.loc[non_base, "textual_change"].median()) if non_base.any() else 0.5
    visual_threshold = float(out.loc[non_base, "visual_change"].median()) if non_base.any() else 0.5
    out["text_visual_threshold_text"] = text_threshold
    out["text_visual_threshold_visual"] = visual_threshold

    text_changed = out["textual_change"].ge(text_threshold)
    visual_changed = out["visual_change"].ge(visual_threshold)
    out["quadrant"] = "baseline"
    out.loc[non_base & ~text_changed & ~visual_changed, "quadrant"] = "Q1_text_stable_visual_stable"
    out.loc[non_base & text_changed & visual_changed, "quadrant"] = "Q2_text_changed_visual_changed"
    out.loc[non_base & text_changed & ~visual_changed, "quadrant"] = "Q3_text_changed_visual_stable"
    out.loc[non_base & ~text_changed & visual_changed, "quadrant"] = "Q4_text_stable_visual_changed"
    return out


def write_missing_and_invalid(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    missing = (
        pd.DataFrame(
            {
                "column": df.columns,
                "missing_count": [int(df[c].isna().sum()) for c in df.columns],
                "missing_rate": [float(df[c].isna().mean()) for c in df.columns],
            }
        )
        .sort_values(["missing_count", "column"], ascending=[False, True])
        .reset_index(drop=True)
    )
    missing.to_csv(out_dir / "missing_values_by_column.csv", index=False)

    invalid_rows: list[dict[str, object]] = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        inf_count = int(np.isinf(values).sum())
        if inf_count:
            invalid_rows.append({"check": "infinite_numeric", "column": col, "count": inf_count, "detail": ""})
    score_cols = [c for c in df.columns if c.endswith("_score")]
    for col in score_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        bad = values.notna() & ((values < 0) | (values > 1))
        if bad.any():
            invalid_rows.append(
                {
                    "check": "score_outside_0_1",
                    "column": col,
                    "count": int(bad.sum()),
                    "detail": f"min={values.min()}, max={values.max()}",
                }
            )
    unknown_prompt = df["prompt_category"].isna() if "prompt_category" in df else pd.Series(False, index=df.index)
    if unknown_prompt.any():
        invalid_rows.append(
            {
                "check": "unmapped_prompt_label",
                "column": "prompt_label",
                "count": int(unknown_prompt.sum()),
                "detail": ",".join(sorted(df.loc[unknown_prompt, "prompt_label"].dropna().unique())),
            }
        )
    invalid = pd.DataFrame(invalid_rows, columns=["check", "column", "count", "detail"])
    invalid.to_csv(out_dir / "invalid_values.csv", index=False)
    return missing


def write_coverage(df: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_prompt = (
        df.groupby(["prompt_label", "prompt_category"], dropna=False)
        .agg(
            cases=("case_id", "count"),
            images=("image_id", "nunique"),
            has_all_required_fields_rate=("has_all_required_fields", "mean"),
            dashboard_case_url_present=("dashboard_case_url", lambda s: s.notna().mean()),
            dashboard_compare_url_present=("dashboard_compare_url", lambda s: s.notna().mean()),
        )
        .reset_index()
        .sort_values("prompt_label")
    )
    by_image = (
        df.groupby("image_id")
        .agg(
            cases=("case_id", "count"),
            prompt_count=("prompt_label", "nunique"),
            has_baseline=("prompt_label", lambda s: bool((s == "baseline_neutral").any())),
            has_all_expected_prompts=("prompt_label", lambda s: set(EXPECTED_PROMPTS).issubset(set(s))),
            required_fields_complete=("has_all_required_fields", "all"),
        )
        .reset_index()
        .sort_values("image_id")
    )
    by_prompt.to_csv(out_dir / "coverage_by_prompt.csv", index=False)
    by_image.to_csv(out_dir / "coverage_by_image.csv", index=False)
    return by_prompt, by_image


def numeric_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    preferred = [
        "prompt_dominated_candidate_score",
        "weak_grounding_candidate_score",
        "unstable_explanation_candidate_score",
        "multipeak_ambiguity_score",
        "bbox_or_grounding_format_score",
        "mean_centroid_shift_vs_baseline",
        "entropy_mean_delta_vs_baseline",
        "top5_mass_mean_delta_vs_baseline",
        "effective_area_norm_mean_delta_vs_baseline",
        "content_jaccard_distance_vs_baseline",
        "textual_change",
        "visual_change",
    ]
    cols = [c for c in preferred if c in df.columns]
    grouped = df.groupby(group_col, dropna=False)[cols].agg(["count", "mean", "median", "std"])
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
    return grouped.reset_index()


def write_prompt_report(label_summary: pd.DataFrame, category_summary: pd.DataFrame, out_dir: Path) -> None:
    score_col = "unstable_explanation_candidate_score_mean"
    lines = [
        "# Prompt Effects Report v1",
        "",
        "This report summarizes exploratory proxy scores and baseline-relative visual sensitivity metrics. "
        "These diagnostic scores are candidate rankings only; they are not causal grounding evidence.",
        "",
        "## Prompt labels ranked by unstable explanation candidate score",
        "",
    ]
    if score_col in label_summary.columns:
        top = label_summary.sort_values(score_col, ascending=False)
        lines.extend(["| prompt_label | mean unstable score | mean weak grounding | mean prompt dominated |", "| --- | ---: | ---: | ---: |"])
        for _, row in top.iterrows():
            lines.append(
                f"| `{row['prompt_label']}` | {row.get(score_col, math.nan):.4f} | "
                f"{row.get('weak_grounding_candidate_score_mean', math.nan):.4f} | "
                f"{row.get('prompt_dominated_candidate_score_mean', math.nan):.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Higher values indicate stronger diagnostic signals for visual sensitivity, prompt dominance, weak-grounding candidates, or multipeak ambiguity under the current proxy formulas.",
            "- `order_disruption_stress` is interpreted as a grounding-format / bbox stress condition, not automatically as an error.",
            "- Baseline rows have zero or near-zero baseline-relative deltas by construction and should mainly be used as anchors.",
            "",
            "## Limitations",
            "",
            "- No semantic word alignment, GT metric, causal faithfulness metric, or bulk EMD is included.",
            "- Candidate scores should be inspected qualitatively in the dashboard before scientific interpretation.",
        ]
    )
    (out_dir / "report_prompt_effects_v1.md").write_text("\n".join(lines), encoding="utf-8")


def write_quadrants(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    cols = [
        "image_id",
        "case_id",
        "prompt_label",
        "prompt_category",
        "textual_change",
        "visual_change",
        "quadrant",
        "response_text",
        "dashboard_case_url",
        "dashboard_compare_url",
    ]
    quad = df.loc[~df["is_baseline"].astype(bool), [c for c in cols if c in df.columns]].copy()
    quad.to_parquet(out_dir / "text_visual_quadrants.parquet", index=False)

    summary = (
        quad.groupby(["quadrant", "prompt_label"], dropna=False)
        .agg(
            cases=("case_id", "count"),
            textual_change_mean=("textual_change", "mean"),
            visual_change_mean=("visual_change", "mean"),
        )
        .reset_index()
        .sort_values(["quadrant", "cases"], ascending=[True, False])
    )
    rep_rows = []
    for quadrant, group in quad.groupby("quadrant"):
        center_text = group["textual_change"].median()
        center_visual = group["visual_change"].median()
        ranked = group.assign(_dist=(group["textual_change"] - center_text).abs() + (group["visual_change"] - center_visual).abs())
        rep_rows.append(ranked.sort_values("_dist").head(5).drop(columns=["_dist"]))
    reps = pd.concat(rep_rows, ignore_index=True) if rep_rows else quad.head(0)
    reps.to_csv(out_dir / "representative_cases_by_quadrant.csv", index=False)

    lines = [
        "# Text vs Vision Quadrants",
        "",
        "Axes are exploratory percentile-normalized aggregates over non-baseline rows.",
        f"Text threshold: `{df['text_visual_threshold_text'].iloc[0]:.4f}`.",
        f"Visual threshold: `{df['text_visual_threshold_visual'].iloc[0]:.4f}`.",
        "",
        "| quadrant | prompt_label | cases | mean textual_change | mean visual_change |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row['quadrant']}` | `{row['prompt_label']}` | {int(row['cases'])} | "
            f"{row['textual_change_mean']:.4f} | {row['visual_change_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Quadrant definitions:",
            "",
            "- Q1: text stable + visual stable.",
            "- Q2: text changes + visual changes.",
            "- Q3: text changes + visual stable.",
            "- Q4: text stable + visual changes.",
        ]
    )
    (out_dir / "text_visual_quadrants_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return quad


def ranking_columns(df: pd.DataFrame, metric_col: str) -> list[str]:
    base = [
        "image_id",
        "case_id",
        "prompt_label",
        "prompt_category",
        metric_col,
        "response_text",
        "dashboard_case_url",
        "dashboard_matrix_url",
        "dashboard_compare_url",
    ]
    return [c for c in base if c in df.columns]


def write_rankings(df: pd.DataFrame, out_dir: Path, top_n: int) -> dict[str, str]:
    rank_dir = out_dir / "rankings_v1"
    rank_dir.mkdir(parents=True, exist_ok=True)
    non_base = df.loc[~df["is_baseline"].astype(bool)].copy()
    specs = {
        "top_prompt_dominated.csv": "prompt_dominated_candidate_score",
        "top_weak_grounding.csv": "weak_grounding_candidate_score",
        "top_unstable_explanation.csv": "unstable_explanation_candidate_score",
        "top_multipeak_ambiguity.csv": "multipeak_ambiguity_score",
        "top_bbox_grounding_format.csv": "bbox_or_grounding_format_score",
        "top_centroid_shift.csv": "mean_centroid_shift_vs_baseline",
        "top_text_changed_visual_stable.csv": "text_changed_visual_stable_score",
        "top_visual_changed_text_stable.csv": "visual_changed_text_stable_score",
    }
    non_base["text_changed_visual_stable_score"] = non_base["textual_change"] * (1.0 - non_base["visual_change"])
    non_base["visual_changed_text_stable_score"] = non_base["visual_change"] * (1.0 - non_base["textual_change"])

    written: dict[str, str] = {}
    for filename, metric in specs.items():
        if metric not in non_base.columns:
            continue
        cols = ranking_columns(non_base, metric)
        ranked = non_base.sort_values(metric, ascending=False)[cols].head(top_n)
        ranked.to_csv(rank_dir / filename, index=False)
        written[filename] = metric

    lines = [
        "# Rankings v1",
        "",
        "Rankings are exploratory candidate lists for dashboard inspection. They are not causal or ground-truth claims.",
        "",
        "| file | metric |",
        "| --- | --- |",
    ]
    for filename, metric in written.items():
        lines.append(f"| `{filename}` | `{metric}` |")
    (rank_dir / "rankings_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return written


def correlation_columns(df: pd.DataFrame) -> list[str]:
    exclude_fragments = ("_id", "count")
    preferred = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in ID_COLS or col in BOOL_OR_FLAG_COLS:
            continue
        if any(col.endswith(fragment) for fragment in exclude_fragments):
            continue
        if col.endswith("_index"):
            continue
        preferred.append(col)
    return preferred


def write_correlations(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    cols = correlation_columns(df)
    corr = df.loc[~df["is_baseline"].astype(bool), cols].corr(method="spearman", numeric_only=True)
    corr.to_csv(out_dir / "correlation_matrix_spearman.csv")

    pairs = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1 :]:
            val = corr.loc[a, b]
            if pd.notna(val):
                pairs.append({"metric_a": a, "metric_b": b, "spearman": float(val), "abs_spearman": abs(float(val))})
    pair_df = pd.DataFrame(pairs).sort_values("abs_spearman", ascending=False)
    lines = [
        "# Metric Redundancy Report",
        "",
        "Spearman correlations are computed on non-baseline case-level rows only.",
        "",
        "## Strongest absolute correlations",
        "",
        "| metric_a | metric_b | spearman |",
        "| --- | --- | ---: |",
    ]
    for _, row in pair_df.head(20).iterrows():
        lines.append(f"| `{row['metric_a']}` | `{row['metric_b']}` | {row['spearman']:.4f} |")
    lines.extend(
        [
            "",
            "High correlations indicate possible redundancy for exploratory ranking, not metric invalidity.",
        ]
    )
    (out_dir / "metric_redundancy_report.md").write_text("\n".join(lines), encoding="utf-8")
    return corr


def write_figures(df: pd.DataFrame, corr: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    score = "unstable_explanation_candidate_score"
    fig, ax = plt.subplots(figsize=(11, 5.5))
    order = sorted(df["prompt_category"].dropna().unique())
    data = [pd.to_numeric(df.loc[df["prompt_category"].eq(cat), score], errors="coerce").dropna() for cat in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_title("Unstable Explanation Candidate Score by Prompt Category")
    ax.set_ylabel("score")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(fig_dir / "boxplot_scores_by_prompt_category.png", dpi=180)
    plt.close(fig)

    non_base = df.loc[~df["is_baseline"].astype(bool)].copy()
    fig, ax = plt.subplots(figsize=(7, 6))
    categories = sorted(non_base["prompt_category"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    for idx, cat in enumerate(categories):
        g = non_base.loc[non_base["prompt_category"].eq(cat)]
        ax.scatter(g["textual_change"], g["visual_change"], s=18, alpha=0.75, label=cat, color=cmap(idx % 10))
    ax.axvline(non_base["textual_change"].median(), color="0.35", linestyle="--", linewidth=1)
    ax.axhline(non_base["visual_change"].median(), color="0.35", linestyle="--", linewidth=1)
    ax.set_xlabel("textual_change")
    ax.set_ylabel("visual_change")
    ax.set_title("Textual vs Visual Change")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "scatter_text_vs_visual_change.png", dpi=180)
    plt.close(fig)

    fig_size = max(7, min(18, 0.35 * len(corr.columns)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(corr.fillna(0.0), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.columns, fontsize=6)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Spearman Correlation Matrix")
    fig.tight_layout()
    fig.savefig(fig_dir / "correlation_matrix_spearman.png", dpi=180)
    plt.close(fig)


def write_quality_report(
    df: pd.DataFrame,
    row_presence: dict[str, int],
    coverage_by_prompt: pd.DataFrame,
    coverage_by_image: pd.DataFrame,
    missing: pd.DataFrame,
    out_dir: Path,
) -> None:
    invalid = pd.read_csv(out_dir / "invalid_values.csv")
    prompt_counts = df.groupby("prompt_label")["case_id"].count().to_dict()
    lines = [
        "# Quality Report v1",
        "",
        "## Dataset counts",
        "",
        f"- case rows: {len(df)}",
        f"- image count: {df['image_id'].nunique()}",
        f"- prompt labels: {df['prompt_label'].nunique()}",
        f"- minimum prompts per image: {int(coverage_by_image['prompt_count'].min())}",
        f"- maximum prompts per image: {int(coverage_by_image['prompt_count'].max())}",
        f"- images with baseline_neutral: {int(coverage_by_image['has_baseline'].sum())}",
        f"- images with all expected prompts: {int(coverage_by_image['has_all_expected_prompts'].sum())}",
        "",
        "## Source row presence",
        "",
    ]
    lines.extend([f"- {k}: {v}" for k, v in row_presence.items()])
    lines.extend(
        [
            "",
            "## Prompt coverage",
            "",
            "```json",
            json.dumps(prompt_counts, indent=2),
            "```",
            "",
            "## Required field checks",
            "",
            f"- cases with all required joined rows and prompt mapping: {int(df['has_all_required_fields'].sum())}",
            f"- missing dashboard_case_url: {int(df['dashboard_case_url'].isna().sum()) if 'dashboard_case_url' in df else 'n/a'}",
            f"- missing dashboard_compare_url: {int(df['dashboard_compare_url'].isna().sum()) if 'dashboard_compare_url' in df else 'n/a'}",
            f"- invalid value checks triggered: {len(invalid)}",
            "",
            "## Largest missing-value columns",
            "",
            "| column | missing_count | missing_rate |",
            "| --- | ---: | ---: |",
        ]
    )
    for _, row in missing.loc[missing["missing_count"] > 0].head(20).iterrows():
        lines.append(f"| `{row['column']}` | {int(row['missing_count'])} | {row['missing_rate']:.4f} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `order_disruption_stress` may contain bbox/object-reference style output and is treated as a grounding-format stress condition.",
            "- Diagnostic scores are exploratory proxies and do not establish causal faithfulness.",
        ]
    )
    (out_dir / "quality_report_v1.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    out_dir: Path,
    archive_root: Path,
    schema: dict[str, object],
    df: pd.DataFrame,
    rankings: dict[str, str],
    used_columns: list[str],
) -> None:
    files = sorted(str(p.relative_to(out_dir)).replace("\\", "/") for p in out_dir.rglob("*") if p.is_file())
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_root": str(archive_root),
        "output_dir": str(out_dir),
        "case_rows": int(len(df)),
        "image_count": int(df["image_id"].nunique()),
        "prompt_count": int(df["prompt_label"].nunique()),
        "used_columns": used_columns,
        "rankings": rankings,
        "schema_tables": {name: {"rows": info["rows"], "columns": info["columns"]} for name, info in schema.items()},
        "methodological_note": "Exploratory proxy analysis only; no causal faithfulness, GT grounding, new inference, or raw all-vs-all map analysis.",
        "created_files": files,
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    archive_root = args.archive_root.resolve()
    parquet_dir = archive_root / "parquet"
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    table_names = [
        "cases",
        "output_diagnostics",
        "visual_sensitivity_vs_baseline",
        "diagnostic_scores",
        "dashboard_links",
        "token_category_summary",
        "layer_scanpaths",
        "word_scanpaths",
    ]
    tables = {name: read_table(parquet_dir, name) for name in table_names}
    schema = schema_info(tables)
    write_schema_snapshot(schema, out_dir)

    df, row_presence = build_case_level(tables)
    df = add_text_visual_axes(df)
    df = df.sort_values(["image_id", "condition_label", "case_id"]).reset_index(drop=True)

    df.to_parquet(out_dir / "analysis_case_level_v1.parquet", index=False)
    df.to_csv(out_dir / "analysis_case_level_v1.csv", index=False)

    missing = write_missing_and_invalid(df, out_dir)
    coverage_by_prompt, coverage_by_image = write_coverage(df, out_dir)
    label_summary = numeric_summary(df, "prompt_label").sort_values("prompt_label")
    category_summary = numeric_summary(df, "prompt_category").sort_values("prompt_category")
    label_summary.to_csv(out_dir / "summary_by_prompt_label.csv", index=False)
    category_summary.to_csv(out_dir / "summary_by_prompt_category.csv", index=False)
    write_prompt_report(label_summary, category_summary, out_dir)
    write_quadrants(df, out_dir)
    rankings = write_rankings(df, out_dir, args.top_n)
    corr = write_correlations(df, out_dir)
    write_figures(df, corr, out_dir)
    write_quality_report(df, row_presence, coverage_by_prompt, coverage_by_image, missing, out_dir)
    write_manifest(out_dir, archive_root, schema, df, rankings, list(df.columns))

    print(f"Wrote {len(df)} case rows to {out_dir}")
    print(f"Images: {df['image_id'].nunique()}, prompts: {df['prompt_label'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
