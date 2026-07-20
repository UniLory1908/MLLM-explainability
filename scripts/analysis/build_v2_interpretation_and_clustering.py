from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_DIR = PROJECT_ROOT / "outputs" / "analysis" / "v1_case_level"
INTERPRETATION_DIR = PROJECT_ROOT / "outputs" / "analysis" / "v2_interpretation"
CLUSTERING_DIR = PROJECT_ROOT / "outputs" / "analysis" / "v2_clustering"
RANDOM_SEED = 20260528

FEATURE_CANDIDATES = [
    "textual_change",
    "visual_change",
    "content_jaccard_distance_vs_baseline",
    "first_divergence_ratio",
    "matched_word_coverage_vs_baseline",
    "mean_centroid_shift_vs_baseline",
    "entropy_mean_delta_vs_baseline",
    "top5_mass_mean_delta_vs_baseline",
    "effective_area_norm_mean_delta_vs_baseline",
    "spread_trace_mean_delta_vs_baseline",
    "peak_count_mean_delta_vs_baseline",
    "secondary_primary_ratio_mean_delta_vs_baseline",
    "multipeak_ratio_delta_vs_baseline",
    "layer_path_length_mean_delta_vs_baseline",
    "layer_max_jump_mean_delta_vs_baseline",
    "layer_tortuosity_mean_delta_vs_baseline",
    "word_path_length_delta_vs_baseline",
    "word_max_jump_delta_vs_baseline",
    "word_tortuosity_delta_vs_baseline",
    "unstable_explanation_candidate_score",
    "prompt_dominated_candidate_score",
    "weak_grounding_candidate_score",
    "multipeak_ambiguity_score",
    "bbox_or_grounding_format_score",
]

MAIN_METRICS = [
    "textual_change",
    "visual_change",
    "content_jaccard_distance_vs_baseline",
    "mean_centroid_shift_vs_baseline",
    "unstable_explanation_candidate_score",
    "prompt_dominated_candidate_score",
    "weak_grounding_candidate_score",
    "multipeak_ambiguity_score",
    "bbox_or_grounding_format_score",
]

RANKING_SPECS = {
    "top_prompt_dominated.csv": ("prompt_dominated", "prompt_dominated_candidate_score"),
    "top_weak_grounding.csv": ("weak_grounding", "weak_grounding_candidate_score"),
    "top_unstable_explanation.csv": ("unstable_explanation", "unstable_explanation_candidate_score"),
    "top_multipeak_ambiguity.csv": ("multipeak_ambiguity", "multipeak_ambiguity_score"),
    "top_centroid_shift.csv": ("centroid_shift", "mean_centroid_shift_vs_baseline"),
    "top_text_changed_visual_stable.csv": ("Q3_text_changed_visual_stable", "text_changed_visual_stable_score"),
    "top_visual_changed_text_stable.csv": ("Q4_text_stable_visual_changed", "visual_changed_text_stable_score"),
}


@dataclass
class ClusterResult:
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    silhouette: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v2 interpretation, case-study selection and light clustering.")
    parser.add_argument("--v1-dir", type=Path, default=V1_DIR)
    parser.add_argument("--interpretation-dir", type=Path, default=INTERPRETATION_DIR)
    parser.add_argument("--clustering-dir", type=Path, default=CLUSTERING_DIR)
    parser.add_argument("--top-outlier-rate", type=float, default=0.05)
    return parser


def read_inputs(v1_dir: Path) -> dict[str, object]:
    rank_dir = v1_dir / "rankings_v1"
    rankings = {path.name: pd.read_csv(path) for path in sorted(rank_dir.glob("top_*.csv"))}
    return {
        "case_level": pd.read_parquet(v1_dir / "analysis_case_level_v1.parquet"),
        "prompt_report": (v1_dir / "report_prompt_effects_v1.md").read_text(encoding="utf-8"),
        "quadrants": pd.read_parquet(v1_dir / "text_visual_quadrants.parquet"),
        "quadrant_report": (v1_dir / "text_visual_quadrants_summary.md").read_text(encoding="utf-8"),
        "representatives": pd.read_csv(v1_dir / "representative_cases_by_quadrant.csv"),
        "summary_label": pd.read_csv(v1_dir / "summary_by_prompt_label.csv"),
        "summary_category": pd.read_csv(v1_dir / "summary_by_prompt_category.csv"),
        "redundancy_report": (v1_dir / "metric_redundancy_report.md").read_text(encoding="utf-8"),
        "correlation": pd.read_csv(v1_dir / "correlation_matrix_spearman.csv", index_col=0),
        "rankings": rankings,
    }


def markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> list[str]:
    use = df[cols].head(max_rows) if max_rows else df[cols]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in use.iterrows():
        values = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(f"`{val}`" if col.endswith("label") or col.endswith("category") or col == "quadrant" else str(val))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def dataframe_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    use = df.head(max_rows).reset_index()
    cols = list(use.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in use.iterrows():
        values = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def choose_feature_columns(df: pd.DataFrame) -> list[str]:
    non_base = df.loc[~df["is_baseline"].astype(bool)]
    selected = []
    for col in FEATURE_CANDIDATES:
        if col not in non_base.columns:
            continue
        s = pd.to_numeric(non_base[col], errors="coerce")
        if s.notna().mean() < 0.95:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        selected.append(col)
    return selected


def robust_scale(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    params: dict[str, dict[str, float]] = {}
    cols = []
    for col in frame.columns:
        s = pd.to_numeric(frame[col], errors="coerce")
        median = float(s.median())
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            iqr = float(s.std(ddof=0))
        if not np.isfinite(iqr) or iqr == 0:
            iqr = 1.0
        filled = s.fillna(median)
        cols.append(((filled - median) / iqr).clip(-8, 8).to_numpy(dtype=float))
        params[col] = {"median": median, "iqr_or_std": iqr, "missing_count": int(s.isna().sum())}
    return np.column_stack(cols), params


def pca_scores(x: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    centered = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vt[:n_components].T
    denom = max(x.shape[0] - 1, 1)
    explained = (s[:n_components] ** 2) / denom
    total = float(np.sum((s**2) / denom))
    ratio = explained / total if total else np.zeros_like(explained)
    return scores, ratio


def kmeans_once(x: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 200) -> ClusterResult:
    centroids = x[rng.choice(x.shape[0], size=k, replace=False)].copy()
    labels = np.zeros(x.shape[0], dtype=int)
    for _ in range(max_iter):
        distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        new_centroids = centroids.copy()
        for cluster_id in range(k):
            members = x[new_labels == cluster_id]
            if len(members):
                new_centroids[cluster_id] = members.mean(axis=0)
            else:
                new_centroids[cluster_id] = x[rng.integers(0, x.shape[0])]
        if np.array_equal(labels, new_labels) and np.allclose(centroids, new_centroids):
            centroids = new_centroids
            break
        labels = new_labels
        centroids = new_centroids
    inertia = float(((x - centroids[labels]) ** 2).sum())
    return ClusterResult(labels=labels, centroids=centroids, inertia=inertia, silhouette=float("nan"))


def silhouette_score(x: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) <= 1 or len(unique) >= len(labels):
        return float("nan")
    distances = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for idx in range(len(labels)):
        same = labels == labels[idx]
        same[idx] = False
        a = distances[idx, same].mean() if same.any() else 0.0
        b = min(distances[idx, labels == other].mean() for other in unique if other != labels[idx])
        denom = max(a, b)
        scores.append((b - a) / denom if denom else 0.0)
    return float(np.mean(scores))


def run_kmeans_grid(x: np.ndarray) -> tuple[int, dict[int, ClusterResult]]:
    rng = np.random.default_rng(RANDOM_SEED)
    results: dict[int, ClusterResult] = {}
    for k in [3, 4, 5, 6]:
        best: ClusterResult | None = None
        for _ in range(25):
            result = kmeans_once(x, k, rng)
            if best is None or result.inertia < best.inertia:
                best = result
        assert best is not None
        best.silhouette = silhouette_score(x, best.labels)
        results[k] = best
    best_k = max(results, key=lambda k: (results[k].silhouette, -k))
    return best_k, results


def average_path_length(n: int) -> float:
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)


def isolation_tree_path_lengths(
    x: np.ndarray,
    indices: np.ndarray,
    all_x: np.ndarray,
    rng: random.Random,
    current_depth: int,
    max_depth: int,
    path_lengths: np.ndarray,
) -> None:
    if current_depth >= max_depth or len(indices) <= 1:
        path_lengths[indices] += current_depth + average_path_length(len(indices))
        return
    feature = rng.randrange(x.shape[1])
    values = all_x[indices, feature]
    min_v = float(values.min())
    max_v = float(values.max())
    if min_v == max_v:
        path_lengths[indices] += current_depth + average_path_length(len(indices))
        return
    split = rng.uniform(min_v, max_v)
    left = indices[values < split]
    right = indices[values >= split]
    isolation_tree_path_lengths(x, left, all_x, rng, current_depth + 1, max_depth, path_lengths)
    isolation_tree_path_lengths(x, right, all_x, rng, current_depth + 1, max_depth, path_lengths)


def isolation_forest_scores(x: np.ndarray, n_trees: int = 100, sample_size: int = 256) -> np.ndarray:
    rng_np = np.random.default_rng(RANDOM_SEED)
    path_lengths = np.zeros(x.shape[0], dtype=float)
    sample_size = min(sample_size, x.shape[0])
    max_depth = int(math.ceil(math.log2(sample_size)))
    for tree_idx in range(n_trees):
        rng = random.Random(RANDOM_SEED + tree_idx)
        sample = rng_np.choice(x.shape[0], size=sample_size, replace=False)
        tree_lengths = np.zeros(x.shape[0], dtype=float)
        isolation_tree_path_lengths(x, sample, x, rng, 0, max_depth, tree_lengths)
        unsampled = tree_lengths == 0
        tree_lengths[unsampled] = max_depth + average_path_length(sample_size)
        path_lengths += tree_lengths
    mean_path = path_lengths / n_trees
    c_n = average_path_length(sample_size) or 1.0
    return np.power(2.0, -mean_path / c_n)


def build_clustering(df: pd.DataFrame, out_dir: Path, top_outlier_rate: float) -> tuple[pd.DataFrame, dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    non_base = df.loc[~df["is_baseline"].astype(bool)].copy().reset_index(drop=True)
    features = choose_feature_columns(non_base)
    x, scale_params = robust_scale(non_base[features])
    pca, explained = pca_scores(x, 3)
    best_k, k_results = run_kmeans_grid(x)
    labels = k_results[best_k].labels
    outlier_score = isolation_forest_scores(x)
    threshold = float(np.quantile(outlier_score, 1.0 - top_outlier_rate))
    is_outlier = outlier_score >= threshold

    cluster_df = non_base[
        [
            "case_id",
            "image_id",
            "prompt_label",
            "prompt_category",
            "quadrant",
            "textual_change",
            "visual_change",
            "content_jaccard_distance_vs_baseline",
            "mean_centroid_shift_vs_baseline",
            "unstable_explanation_candidate_score",
            "prompt_dominated_candidate_score",
            "weak_grounding_candidate_score",
            "multipeak_ambiguity_score",
            "bbox_or_grounding_format_score",
            "dashboard_case_url",
            "dashboard_matrix_url",
            "dashboard_compare_url",
            "response_text",
        ]
    ].copy()
    cluster_df["cluster_id"] = labels
    cluster_df["pca_x"] = pca[:, 0]
    cluster_df["pca_y"] = pca[:, 1]
    cluster_df["pca_z"] = pca[:, 2]
    cluster_df["outlier_score"] = outlier_score
    cluster_df["is_outlier"] = is_outlier
    ordered_cols = [
        "case_id",
        "image_id",
        "prompt_label",
        "prompt_category",
        "quadrant",
        "cluster_id",
        "pca_x",
        "pca_y",
        "pca_z",
        "outlier_score",
        "is_outlier",
        "textual_change",
        "visual_change",
        "content_jaccard_distance_vs_baseline",
        "mean_centroid_shift_vs_baseline",
        "unstable_explanation_candidate_score",
        "prompt_dominated_candidate_score",
        "weak_grounding_candidate_score",
        "multipeak_ambiguity_score",
        "bbox_or_grounding_format_score",
        "dashboard_case_url",
        "dashboard_matrix_url",
        "dashboard_compare_url",
        "response_text",
    ]
    cluster_df = cluster_df[ordered_cols]
    cluster_df.to_parquet(out_dir / "clusters_v1.parquet", index=False)

    outliers = cluster_df.sort_values("outlier_score", ascending=False).head(max(20, int(is_outlier.sum())))
    outliers.to_csv(out_dir / "outliers_v1.csv", index=False)
    reps = representative_clusters(cluster_df)
    reps.to_csv(out_dir / "cluster_representative_cases.csv", index=False)

    write_cluster_figures(cluster_df, out_dir / "figures")
    write_cluster_summary(cluster_df, reps, features, scale_params, explained, best_k, k_results, out_dir)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(V1_DIR / "analysis_case_level_v1.parquet"),
        "rows_clustered": int(len(cluster_df)),
        "baseline_excluded": True,
        "features_used": features,
        "preprocessing": "median imputation plus robust median/IQR scaling clipped to [-8, 8]",
        "pca_explained_variance_ratio_3d": [float(v) for v in explained],
        "kmeans_scores": {str(k): {"silhouette": r.silhouette, "inertia": r.inertia} for k, r in k_results.items()},
        "selected_k": int(best_k),
        "selected_k_reason": "highest silhouette among k=3..6, then interpreted through prompt/quadrant composition",
        "outlier_method": "lightweight Isolation Forest implemented locally with random split trees; no sklearn dependency",
        "outlier_threshold": threshold,
        "outlier_count": int(is_outlier.sum()),
    }
    (out_dir / "clustering_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return cluster_df, manifest


def representative_clusters(cluster_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_id, group in cluster_df.groupby("cluster_id"):
        center_x = group["pca_x"].median()
        center_y = group["pca_y"].median()
        ranked = group.assign(_dist=(group["pca_x"] - center_x).abs() + (group["pca_y"] - center_y).abs())
        rows.append(ranked.sort_values("_dist").head(5).drop(columns="_dist"))
    return pd.concat(rows, ignore_index=True)


def write_cluster_figures(cluster_df: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    def scatter_by(column: str, path: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        values = list(cluster_df[column].dropna().unique())
        cmap = plt.get_cmap("tab10")
        for idx, value in enumerate(values):
            group = cluster_df.loc[cluster_df[column].eq(value)]
            ax.scatter(group["pca_x"], group["pca_y"], s=18, alpha=0.75, label=str(value), color=cmap(idx % 10))
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(fig_dir / path, dpi=180)
        plt.close(fig)

    scatter_by("cluster_id", "pca_clusters.png", "PCA projection by KMeans cluster")
    scatter_by("prompt_category", "pca_prompt_category.png", "PCA projection by prompt category")
    scatter_by("quadrant", "pca_quadrants.png", "PCA projection by text-vision quadrant")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(cluster_df["outlier_score"], bins=30, color="#4c78a8", edgecolor="white")
    ax.axvline(cluster_df.loc[cluster_df["is_outlier"], "outlier_score"].min(), color="0.25", linestyle="--")
    ax.set_xlabel("outlier_score")
    ax.set_ylabel("cases")
    ax.set_title("Isolation-style outlier score distribution")
    fig.tight_layout()
    fig.savefig(fig_dir / "outlier_score_distribution.png", dpi=180)
    plt.close(fig)


def write_cluster_summary(
    cluster_df: pd.DataFrame,
    reps: pd.DataFrame,
    features: list[str],
    scale_params: dict[str, dict[str, float]],
    explained: np.ndarray,
    best_k: int,
    k_results: dict[int, ClusterResult],
    out_dir: Path,
) -> None:
    size = cluster_df.groupby("cluster_id").size().reset_index(name="cases")
    category = pd.crosstab(cluster_df["cluster_id"], cluster_df["prompt_category"])
    quadrants = pd.crosstab(cluster_df["cluster_id"], cluster_df["quadrant"])
    means = cluster_df.groupby("cluster_id")[MAIN_METRICS].mean(numeric_only=True).reset_index()
    outliers = cluster_df.sort_values("outlier_score", ascending=False).head(10)

    lines = [
        "# Cluster Summary v1",
        "",
        f"Main clustering uses the 700 non-baseline rows. Baseline rows are excluded because their baseline-relative deltas are anchor values by construction.",
        "",
        "## Method",
        "",
        f"- selected clusters: `{best_k}`",
        "- preprocessing: median imputation, robust median/IQR scaling, clipping to [-8, 8]",
        "- PCA: NumPy SVD on scaled features",
        "- KMeans: local NumPy implementation over k=3,4,5,6",
        "- outliers: lightweight Isolation Forest style random split trees implemented locally",
        f"- PCA explained variance ratio: {', '.join(f'{v:.4f}' for v in explained)}",
        "",
        "## KMeans selection",
        "",
        "| k | silhouette | inertia |",
        "| ---: | ---: | ---: |",
    ]
    for k, result in k_results.items():
        lines.append(f"| {k} | {result.silhouette:.4f} | {result.inertia:.2f} |")

    lines.extend(["", "## Feature set", "", *[f"- `{col}`" for col in features], ""])
    lines.extend(["## Cluster sizes", "", *markdown_table(size, ["cluster_id", "cases"]), ""])

    lines.extend(["## Prompt category distribution", "", dataframe_markdown(category), ""])
    lines.extend(["## Quadrant distribution", "", dataframe_markdown(quadrants), ""])
    lines.extend(["## Mean metrics by cluster", "", dataframe_markdown(means.set_index("cluster_id")), ""])

    lines.extend(["## Prudential cluster interpretation", ""])
    for _, row in means.iterrows():
        cid = int(row["cluster_id"])
        dominant_prompt = category.loc[cid].sort_values(ascending=False).head(2).to_dict()
        dominant_quad = quadrants.loc[cid].sort_values(ascending=False).head(2).to_dict()
        lines.append(
            f"- Cluster `{cid}`: mean textual_change `{row['textual_change']:.3f}`, visual_change `{row['visual_change']:.3f}`, "
            f"unstable score `{row['unstable_explanation_candidate_score']:.3f}`. "
            f"Dominant prompt categories: {dominant_prompt}. Dominant quadrants: {dominant_quad}. "
            "Interpret as an exploratory grouping for dashboard inspection, not as a causal class."
        )

    lines.extend(["", "## Representative cases", "", *markdown_table(reps, ["cluster_id", "image_id", "prompt_label", "prompt_category", "quadrant", "dashboard_case_url"], 30), ""])
    lines.extend(["## Top outliers", "", *markdown_table(outliers, ["image_id", "prompt_label", "prompt_category", "quadrant", "outlier_score", "dashboard_case_url"], 10), ""])
    lines.extend(
        [
            "## Limits",
            "",
            "- Clusters depend on proxy features and robust scaling choices.",
            "- No semantic word alignment, GT grounding, causal faithfulness metric, or raw all-vs-all map comparison is included.",
            "- The local Isolation Forest implementation is used because scikit-learn is not installed; treat outlier scores as exploratory.",
        ]
    )
    (out_dir / "cluster_summary.md").write_text("\n".join(lines), encoding="utf-8")


def add_case(
    rows: list[dict[str, object]],
    selected_ids: set[str],
    full_df: pd.DataFrame,
    row: pd.Series,
    reason: str,
    score_name: str,
    score_value: float,
) -> None:
    case_id = row["case_id"]
    duplicate_note = "duplicate selection reason for an already selected case" if case_id in selected_ids else ""
    selected_ids.add(case_id)
    full = full_df.loc[full_df["case_id"].eq(case_id)]
    if full.empty:
        full_row = row
    else:
        full_row = full.iloc[0]
    rows.append(
        {
            "image_id": full_row.get("image_id", row.get("image_id")),
            "case_id": case_id,
            "prompt_label": full_row.get("prompt_label", row.get("prompt_label")),
            "prompt_category": full_row.get("prompt_category", row.get("prompt_category")),
            "quadrant": full_row.get("quadrant", row.get("quadrant", "")),
            "main_selection_reason": reason,
            "main_score_name": score_name,
            "main_score_value": score_value,
            "textual_change": full_row.get("textual_change", row.get("textual_change", np.nan)),
            "visual_change": full_row.get("visual_change", row.get("visual_change", np.nan)),
            "response_text": full_row.get("response_text", row.get("response_text", "")),
            "dashboard_case_url": full_row.get("dashboard_case_url", row.get("dashboard_case_url", "")),
            "dashboard_matrix_url": full_row.get("dashboard_matrix_url", row.get("dashboard_matrix_url", "")),
            "dashboard_compare_url": full_row.get("dashboard_compare_url", row.get("dashboard_compare_url", "")),
            "notes": duplicate_note,
        }
    )


def select_case_studies(inputs: dict[str, object], cluster_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    full_df = inputs["case_level"]
    rankings: dict[str, pd.DataFrame] = inputs["rankings"]
    reps: pd.DataFrame = inputs["representatives"]
    rows: list[dict[str, object]] = []
    selected_ids: set[str] = set()

    for filename, (reason, score_name) in RANKING_SPECS.items():
        if filename not in rankings:
            continue
        if reason.startswith("Q"):
            continue
        for _, row in rankings[filename].head(8).iterrows():
            if row["case_id"] not in selected_ids or len([r for r in rows if r["main_selection_reason"] == reason]) < 3:
                add_case(rows, selected_ids, full_df, row, reason, score_name, float(row[score_name]))
            if len([r for r in rows if r["main_selection_reason"] == reason]) >= 3:
                break

    quadrant_targets = {
        "Q1_robust_stable": "Q1_text_stable_visual_stable",
        "Q2_text_changed_visual_changed": "Q2_text_changed_visual_changed",
        "Q3_text_changed_visual_stable": "Q3_text_changed_visual_stable",
        "Q4_text_stable_visual_changed": "Q4_text_stable_visual_changed",
    }
    for reason, quadrant in quadrant_targets.items():
        source = reps.loc[reps["quadrant"].eq(quadrant)].copy()
        if len(source) < 3:
            source = full_df.loc[full_df["quadrant"].eq(quadrant)].copy()
        if quadrant.startswith("Q1"):
            source = source.assign(_score=1.0 - source["textual_change"] + 1.0 - source["visual_change"]).sort_values("_score", ascending=False)
            score_name = "stable_low_change_proxy"
        elif quadrant.startswith("Q3"):
            source = source.assign(_score=source["textual_change"] * (1.0 - source["visual_change"])).sort_values("_score", ascending=False)
            score_name = "text_changed_visual_stable_score"
        elif quadrant.startswith("Q4"):
            source = source.assign(_score=source["visual_change"] * (1.0 - source["textual_change"])).sort_values("_score", ascending=False)
            score_name = "visual_changed_text_stable_score"
        else:
            source = source.assign(_score=source["textual_change"] + source["visual_change"]).sort_values("_score", ascending=False)
            score_name = "text_visual_joint_change_proxy"
        count = 0
        for _, row in source.iterrows():
            add_case(rows, selected_ids, full_df, row, reason, score_name, float(row["_score"]))
            count += 1
            if count >= 3:
                break

    selected = pd.DataFrame(rows)
    cluster_cols = cluster_df[["case_id", "cluster_id", "outlier_score", "is_outlier"]]
    selected = selected.merge(cluster_cols, on="case_id", how="left")
    selected.to_csv(out_dir / "selected_case_studies_v1.csv", index=False)
    write_case_study_report(selected, out_dir)
    return selected


def write_case_study_report(selected: pd.DataFrame, out_dir: Path) -> None:
    first_open = selected.sort_values(["is_outlier", "main_score_value"], ascending=[False, False]).head(10)
    reason_counts = selected.groupby("main_selection_reason").size().reset_index(name="cases")
    dup_count = int(selected["case_id"].duplicated().sum())
    lines = [
        "# Case Study Selection Report",
        "",
        "The selection balances high diagnostic rankings with quadrant representatives, so it is not only a top-score list.",
        "",
        "## Selection coverage",
        "",
        *markdown_table(reason_counts, ["main_selection_reason", "cases"]),
        "",
        f"Overlapping rows after multi-reason selection: `{dup_count}` duplicated case ids in the CSV view.",
        "",
        "## Open first in dashboard",
        "",
        *markdown_table(first_open, ["image_id", "prompt_label", "prompt_category", "quadrant", "main_selection_reason", "main_score_value", "dashboard_case_url"], 10),
        "",
        "## Suggested baseline-vs-prompt comparisons",
        "",
        "- Start with cases selected as `unstable_explanation`, `prompt_dominated`, or high outlier score.",
        "- For Q3, inspect whether response wording changes while TAM-derived visual sensitivity remains relatively stable against baseline.",
        "- For Q4, inspect whether text remains comparatively stable while attribution geometry or diagnostic scores drift.",
        "- For `order_disruption_stress`, treat bbox/object-reference outputs as a grounding-format stress response, not automatically as a bug.",
        "",
        "## Strong thesis/report candidates",
        "",
        "- Use Q2 cases for clear coupled text and visual drift examples.",
        "- Use Q3/Q4 cases to argue that response change and TAM visual sensitivity can decouple.",
        "- Use multipeak/ambiguous cases to motivate qualitative dashboard inspection of spatial ambiguity proxies.",
        "",
        "## Limits",
        "",
        "These are candidate cases for qualitative inspection, not causal findings.",
    ]
    (out_dir / "case_study_selection_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_interpretation_report(inputs: dict[str, object], selected: pd.DataFrame, cluster_df: pd.DataFrame, manifest: dict[str, object], out_dir: Path) -> None:
    label = inputs["summary_label"].copy()
    category = inputs["summary_category"].copy()
    corr = inputs["correlation"]
    ranking_top = []
    for filename, (reason, score_name) in RANKING_SPECS.items():
        df = inputs["rankings"].get(filename)
        if df is not None and not df.empty and score_name in df.columns:
            top = df.iloc[0]
            ranking_top.append(
                {
                    "ranking": reason,
                    "image_id": top["image_id"],
                    "prompt_label": top["prompt_label"],
                    "prompt_category": top["prompt_category"],
                    "score": float(top[score_name]),
                }
            )
    ranking_top_df = pd.DataFrame(ranking_top)
    q_counts = cluster_df.groupby("quadrant").size().reset_index(name="cases")
    top_corr = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1 :]:
            value = corr.loc[a, b]
            if pd.notna(value):
                top_corr.append({"metric_a": a, "metric_b": b, "spearman": float(value), "abs": abs(float(value))})
    top_corr_df = pd.DataFrame(top_corr).sort_values("abs", ascending=False).head(10)

    lines = [
        "# Interpretation Report v2",
        "",
        "This is a preliminary scientific interpretation of the v1 case-level archive. All conclusions use cautious proxy language: candidate, diagnostic score, visual sensitivity, and drift rispetto al baseline.",
        "",
        "## 1. Prompt-label patterns",
        "",
        *markdown_table(
            label.sort_values("unstable_explanation_candidate_score_mean", ascending=False),
            [
                "prompt_label",
                "unstable_explanation_candidate_score_mean",
                "prompt_dominated_candidate_score_mean",
                "weak_grounding_candidate_score_mean",
                "textual_change_mean",
                "visual_change_mean",
            ],
        ),
        "",
        "The strongest average unstable-explanation proxy is associated with `order_disruption_stress`, followed by `colleague_obj_detection_hard` and `misleading_wrong_subject`. This suggests higher prompt sensitivity under grounding-format stress and adversarial/object-reference style conditions.",
        "",
        "## 2. Prompt-category patterns",
        "",
        *markdown_table(
            category.sort_values("visual_change_mean", ascending=False),
            [
                "prompt_category",
                "unstable_explanation_candidate_score_mean",
                "prompt_dominated_candidate_score_mean",
                "weak_grounding_candidate_score_mean",
                "textual_change_mean",
                "visual_change_mean",
            ],
        ),
        "",
        "Categories are currently one-to-one with prompt labels, but the category view is useful for paper language: `grounding_format_stress`, `misleading`, and `object_detection` show the clearest drift pattern.",
        "",
        "## 3. Diagnostic score interpretation",
        "",
        "The diagnostic scores are bounded proxy rankings. High values identify candidates for inspection, not proof of causal grounding, hallucination, or model attention to a specific object.",
        "",
        "## 4. Text-vs-vision matrix",
        "",
        *markdown_table(q_counts, ["quadrant", "cases"]),
        "",
        "Q2 is useful for cases where text and visual sensitivity drift together. Q3 and Q4 are especially interesting because they suggest decoupling: response text can change with relatively stable TAM summaries, or TAM summaries can drift while text remains comparatively stable.",
        "",
        "## 5. Ranking patterns",
        "",
        *markdown_table(ranking_top_df, ["ranking", "image_id", "prompt_label", "prompt_category", "score"]),
        "",
        "The top rankings concentrate many `order_disruption_stress`, `misleading`, `object_detection`, and `ambiguous` cases, but the selected case-study list deliberately preserves quadrant diversity.",
        "",
        "## 6. More informative metrics",
        "",
        "- `textual_change` and `content_jaccard_distance_vs_baseline` summarize response drift clearly.",
        "- `visual_change`, `mean_centroid_shift_vs_baseline`, and scanpath deltas are useful for baseline-relative visual sensitivity.",
        "- `unstable_explanation_candidate_score` and `prompt_dominated_candidate_score` are practical ranking proxies.",
        "- `multipeak_ambiguity_score` remains useful for ambiguous/multipeak candidates.",
        "",
        "## 7. Redundant or less useful metrics",
        "",
        *markdown_table(top_corr_df, ["metric_a", "metric_b", "spearman"], 10),
        "",
        "`content_jaccard_vs_baseline` and its distance are exact complements. Length metrics are strongly redundant. `mean_centroid_shift` and `median_centroid_shift` are also close. Bulk similarity and EMD fields are not useful in this archive because they are mostly missing or intentionally not computed.",
        "",
        "## 8. Methodological limits",
        "",
        "- No new inference, GT grounding, semantic word alignment, causal faithfulness metric, or raw all-vs-all map comparison is included.",
        "- TAM-derived scanpaths are not human gaze.",
        "- Cluster and outlier labels are exploratory summaries of proxy features.",
        "",
        "## 9. Recommended next steps",
        "",
        "- Open selected cases in the dashboard and document qualitative baseline-vs-prompt comparisons.",
        "- Use Q3/Q4 case studies to support the text/vision decoupling argument cautiously.",
        "- Consider a later clustering batch with UMAP/HDBSCAN only if dependencies are intentionally added.",
        "- Keep causal language out of the report unless a future perturbative/GT protocol is added.",
        "",
        f"Clustering selected k: `{manifest['selected_k']}` with silhouette `{manifest['kmeans_scores'][str(manifest['selected_k'])]['silhouette']:.4f}`.",
    ]
    (out_dir / "interpretation_report_v2.md").write_text("\n".join(lines), encoding="utf-8")


def write_crosscheck(inputs: dict[str, object], cluster_df: pd.DataFrame, selected: pd.DataFrame, out_dir: Path) -> None:
    rankings = inputs["rankings"]
    top_cases = []
    for filename, (reason, score_name) in RANKING_SPECS.items():
        df = rankings.get(filename)
        if df is None:
            continue
        for _, row in df.head(20).iterrows():
            top_cases.append({"case_id": row["case_id"], "ranking": reason})
    top_df = pd.DataFrame(top_cases).merge(cluster_df, on="case_id", how="left")
    ranking_cluster = pd.crosstab(top_df["ranking"], top_df["cluster_id"])
    quadrant_cluster = pd.crosstab(cluster_df["quadrant"], cluster_df["cluster_id"])
    category_cluster = pd.crosstab(cluster_df["prompt_category"], cluster_df["cluster_id"])
    ambiguous = cluster_df.loc[cluster_df["prompt_category"].eq("ambiguous")]
    outlier_top = top_df.loc[top_df["is_outlier"].fillna(False)]
    open_first = selected.sort_values(["is_outlier", "main_score_value"], ascending=[False, False]).head(12)

    lines = [
        "# Ranking, Quadrant and Cluster Cross-check",
        "",
        "## 1. Do top-ranked cases fall into the same clusters?",
        "",
        dataframe_markdown(ranking_cluster),
        "",
        "Top-ranked cases are not confined to one cluster, but several high-drift rankings concentrate in clusters with higher textual/visual change. This supports using clusters as grouping aids rather than replacement rankings.",
        "",
        "## 2. Q3 and Q4 cluster behavior",
        "",
        dataframe_markdown(quadrant_cluster),
        "",
        "Q3 and Q4 do not form perfectly isolated clusters. They remain interesting because they mark text/vision decoupling within a broader feature space.",
        "",
        "## 3. Prompt category dominance",
        "",
        dataframe_markdown(category_cluster),
        "",
        "`misleading`, `object_detection`, and `grounding_format_stress` are visibly concentrated in higher-drift cluster regions, while `extra_knowledge` and `image_grounded` contribute more to stable or moderate clusters.",
        "",
        "## 4. Ambiguous/multipeak pattern",
        "",
        f"Ambiguous cases: `{len(ambiguous)}`. Cluster distribution: `{ambiguous['cluster_id'].value_counts().sort_index().to_dict()}`.",
        "Ambiguous cases contribute strongly to multipeak/weak-grounding rankings but do not become a single isolated cluster under this feature set.",
        "",
        "## 5. Outlier overlap with rankings",
        "",
        f"Top-20 ranking entries marked as outliers after merge: `{len(outlier_top)}`.",
        "Outliers partly overlap with diagnostic rankings, especially when bbox-format, prompt-dominated, or extreme text/visual drift features are high.",
        "",
        "## 6. Most convincing dashboard candidates",
        "",
        *markdown_table(open_first, ["image_id", "prompt_label", "prompt_category", "quadrant", "main_selection_reason", "cluster_id", "outlier_score", "dashboard_case_url"], 12),
        "",
        "Use these as first qualitative dashboard checks. The goal is to inspect visual sensitivity and baseline drift, not to claim causal grounding.",
    ]
    (out_dir / "ranking_quadrant_cluster_crosscheck.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    interpretation_dir = args.interpretation_dir.resolve()
    clustering_dir = args.clustering_dir.resolve()
    interpretation_dir.mkdir(parents=True, exist_ok=True)
    clustering_dir.mkdir(parents=True, exist_ok=True)

    inputs = read_inputs(args.v1_dir.resolve())
    cluster_df, manifest = build_clustering(inputs["case_level"], clustering_dir, args.top_outlier_rate)
    selected = select_case_studies(inputs, cluster_df, interpretation_dir)
    write_interpretation_report(inputs, selected, cluster_df, manifest, interpretation_dir)
    write_crosscheck(inputs, cluster_df, selected, interpretation_dir)

    print(f"Wrote interpretation outputs to {interpretation_dir}")
    print(f"Wrote clustering outputs to {clustering_dir}")
    print(f"Selected k={manifest['selected_k']} with {len(manifest['features_used'])} features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
