from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from flask import Flask, Response, abort, render_template, request, send_file
except Exception as exc:  # pragma: no cover
    raise SystemExit("Flask is required for the dashboard. Install requirements.txt first.") from exc

from scripts.dashboard.config import DashboardConfig, resolve_project_path
from scripts.dashboard.data_access import get_case, get_case_words, get_map_row
from scripts.dashboard.db import connect, initialize
from scripts.dashboard.metric_registry import (
    FUTURE_METRICS,
    GROUP_DESCRIPTIONS,
    GUIDE_GROUPS,
    METRIC_REGISTRY,
    OPTIONAL_GT,
    format_metric_value,
    metric_info,
)
from scripts.dashboard.pairwise import compute_pair_for_rows
from scripts.dashboard.rendering import (
    parse_model_locations,
    render_difference,
    render_final_layer_animation,
    render_final_layer_preview,
    render_map,
    render_matrix_cell,
    render_model_location_overlay,
    render_scanpath,
)


CASE_STUDIES_CSV = PROJECT_ROOT / "outputs" / "analysis" / "v3_case_studies" / "final_case_studies.csv"
SAVED_QUERIES_DIR = PROJECT_ROOT / "outputs" / "analysis" / "v5_queries"
V6_ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "analysis" / "v6_horizontal_analysis"
V6_DASHBOARD_VIEWS_DIR = V6_ANALYSIS_DIR / "dashboard_views"
V6_QUESTIONS_DIR = V6_DASHBOARD_VIEWS_DIR / "questions"
FIRST_REVIEW_CASE_IDS = {"CS09", "CS07", "CS08", "CS02", "CS03", "CS05"}
SAVED_QUERY_TITLES = {
    "ambiguous_multipeak_query": "Ambiguous prompt: multipeak patterns",
    "centroid_shift_by_prompt": "Centroid shift by prompt",
    "cluster_prompt_image_breakdown": "Cluster drivers: prompt vs image",
    "coordinate_outputs_by_image": "Images with repeated coordinate outputs",
    "coordinate_outputs_v1": "Coordinate/bbox outputs by prompt",
    "decoupling_by_prompt": "Text-vision decoupling by prompt",
    "diffuse_heatmaps_by_prompt": "Diffuse heatmaps by prompt",
    "extra_knowledge_stability_query": "Extra knowledge stability",
    "images_often_in_cluster2_or_outlier": "Images often in high-drift cluster/outliers",
    "image_grounded_stability_query": "Image-grounded prompt stability",
    "image_prompt_sensitivity_ranking": "Most prompt-sensitive images",
    "long_response_visual_stable": "Long responses with stable visual maps",
    "misleading_prompt_effect": "Misleading prompt effect",
    "outlier_prompt_distribution": "Outliers by prompt",
    "prompt_dominated_but_not_weak_grounding": "Prompt-dominated but not weak-grounding",
    "q3_text_changed_visual_stable": "Q3: text changed, visual stable",
    "q4_text_stable_visual_changed": "Q4: text stable, visual changed",
    "robust_images_ranking": "Most robust images",
    "short_response_high_visual_drift": "Short responses with high visual drift",
    "unexpected_coordinate_outputs": "Unexpected coordinate/bbox outputs",
    "weak_grounding_not_prompt_dominated": "Weak-grounding without prompt dominance",
}


def dashboard_relative_url(value: str) -> str:
    if not value:
        return ""
    parsed = urlsplit(value)
    if parsed.scheme and parsed.netloc:
        path = parsed.path or "/"
        return f"{path}?{parsed.query}" if parsed.query else path
    return value


def dashboard_relative_text(value: str) -> str:
    if not value:
        return ""
    for prefix in (
        "http://127.0.0.1:5050",
        "http://127.0.0.1:4321",
        "https://lory-fisso.tailbaabf0.ts.net",
    ):
        value = value.replace(prefix, "")
    return value


def readable_query_title(slug: str, question: str = "") -> str:
    if slug in SAVED_QUERY_TITLES:
        return SAVED_QUERY_TITLES[slug]
    if question:
        return question.strip().strip("`")
    return slug.replace("_", " ").title()


def load_selected_case_studies() -> dict[str, list[dict[str, str]]]:
    if not CASE_STUDIES_CSV.exists():
        return {"all": [], "first_review": []}
    rows: list[dict[str, str]] = []
    with CASE_STUDIES_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row["dashboard_case_path"] = dashboard_relative_url(row.get("dashboard_case_url", ""))
            row["dashboard_matrix_path"] = dashboard_relative_url(row.get("dashboard_matrix_url", ""))
            row["dashboard_compare_path"] = dashboard_relative_url(row.get("dashboard_compare_url", ""))
            rows.append(row)
    first_review = [row for row in rows if row.get("case_study_id") in FIRST_REVIEW_CASE_IDS]
    first_review.sort(key=lambda row: ["CS09", "CS07", "CS08", "CS02", "CS03", "CS05"].index(row["case_study_id"]))
    rows.sort(key=lambda row: row.get("case_study_id", ""))
    return {"all": rows, "first_review": first_review}


def load_saved_queries() -> list[dict[str, object]]:
    if not SAVED_QUERIES_DIR.exists():
        return []
    queries: list[dict[str, object]] = []
    for path in sorted(SAVED_QUERIES_DIR.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_dir():
            continue
        manifest_path = path / "manifest.json"
        question_path = path / "question.md"
        answer_path = path / "answer.md"
        results_path = path / "results.csv"
        if not (answer_path.exists() or results_path.exists()):
            continue
        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
        question = ""
        if question_path.exists():
            question = question_path.read_text(encoding="utf-8").replace("# Question", "").strip()
        question_text = question or str(manifest.get("question") or path.name)
        queries.append(
            {
                "slug": path.name,
                "title": readable_query_title(path.name, question_text),
                "question": question_text,
                "matching_cases": manifest.get("matching_cases"),
                "matching_images": manifest.get("matching_images"),
                "generated_at": manifest.get("generated_at", ""),
                "has_results": results_path.exists(),
                "has_answer": answer_path.exists(),
            }
        )
    return queries


def saved_query_dir(slug: str) -> Path:
    root = SAVED_QUERIES_DIR.resolve()
    path = (root / slug).resolve()
    if root not in path.parents and path != root:
        abort(404)
    if not path.exists() or not path.is_dir():
        abort(404)
    return path


def load_saved_query_detail(slug: str, row_limit: int = 200) -> dict[str, object]:
    path = saved_query_dir(slug)
    manifest_path = path / "manifest.json"
    question_path = path / "question.md"
    answer_path = path / "answer.md"
    results_path = path / "results.csv"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    answer = dashboard_relative_text(answer_path.read_text(encoding="utf-8")) if answer_path.exists() else ""
    question = question_path.read_text(encoding="utf-8").replace("# Question", "").strip() if question_path.exists() else str(manifest.get("question") or slug)
    columns: list[str] = []
    rows: list[dict[str, str]] = []
    total_rows = 0
    if results_path.exists():
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            for row in reader:
                total_rows += 1
                if len(rows) < row_limit:
                    normalized_row = dict(row)
                    for key in ("dashboard_case_url", "dashboard_matrix_url", "dashboard_compare_url", "dashboard_word_url"):
                        if key in normalized_row:
                            normalized_row[key] = dashboard_relative_url(normalized_row.get(key, ""))
                    rows.append(normalized_row)
    return {
        "slug": slug,
        "title": readable_query_title(slug, question),
        "question": question,
        "answer": answer,
        "manifest": manifest,
        "columns": columns,
        "rows": rows,
        "total_rows": total_rows,
        "row_limit": row_limit,
        "has_csv": results_path.exists(),
        "has_parquet": (path / "results.parquet").exists(),
    }


def safe_float(value, default: float | None = None) -> float | None:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default: int | None = None) -> int | None:
    number = safe_float(value)
    if number is None:
        return default
    return int(round(number))


def fmt_compact(value, digits: int = 3) -> str:
    number = safe_float(value)
    if number is None:
        return "n/a"
    if abs(number) >= 1000 and number == int(number):
        return f"{int(number):,}"
    return f"{number:.{digits}f}"


def compact_preview(value: str, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def read_csv_dicts(path: Path, limit: int | None = None) -> tuple[list[dict[str, str]], list[str], str | None]:
    if not path.exists():
        return [], [], f"missing file: {path.name}"
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = []
            for row in reader:
                rows.append(dict(row))
                if limit is not None and len(rows) >= limit:
                    break
            return rows, list(reader.fieldnames or []), None
    except OSError as exc:
        return [], [], f"could not read {path.name}: {exc}"


def read_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def first_value(rows: list[dict[str, str]], column: str, default: str = "n/a") -> str:
    if not rows:
        return default
    return rows[0].get(column) or default


def rate_value(row: dict[str, str], column: str) -> str:
    value = safe_float(row.get(column))
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def row_for_variant(rows: list[dict[str, str]], token: str) -> dict[str, str]:
    for row in rows:
        if token in row.get("variant", ""):
            return row
    return {}


def plot_exists(filename: str) -> bool:
    return analysis_plot_path(filename) is not None


def analysis_plot_path(filename: str) -> Path | None:
    if "/" in filename or "\\" in filename or not filename.endswith(".png"):
        return None
    for root in (
        (V6_ANALYSIS_DIR / "plots").resolve(),
        V6_DASHBOARD_VIEWS_DIR.resolve(),
        (V6_QUESTIONS_DIR / "claim_plots").resolve(),
    ):
        path = (root / filename).resolve()
        if root in path.parents and path.exists() and path.is_file():
            return path
    return None


def numeric_sort(rows: list[dict[str, str]], column: str, reverse: bool = True) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: safe_float(row.get(column), float("-inf") if reverse else float("inf")), reverse=reverse)


def prompt_interpretation(prompt: str) -> str:
    interpretations = {
        "baseline_neutral": "Reference prompt; baseline-relative deltas are zero by construction.",
        "image_grounded_visible_only": "Comparatively more diffuse attribution and lower top-5 concentration.",
        "ambiguous_open": "Open-ended behavior with relatively diffuse effective area and weaker separability.",
        "misleading_wrong_subject": "High visual change with more concentrated attribution than baseline.",
        "extra_knowledge_context": "Lowest mean visual change among nonbaseline prompts in v6.",
        "reasoning_controlled_brief": "Moderate visual change with concentrated attribution and larger centroid shift.",
        "order_disruption_stress": "High variability and many bbox/location-style outputs.",
        "colleague_obj_detection_hard": "High visual change; concentrated profile and often confused with misleading/order-disruption in separability.",
    }
    return interpretations.get(prompt, "Prompt behavior summary not documented.")


def add_case_links(row: dict[str, str]) -> dict[str, str]:
    row = dict(row)
    row["dashboard_case_path"] = dashboard_relative_url(row.get("dashboard_case_url", ""))
    row["dashboard_compare_path"] = dashboard_relative_url(row.get("dashboard_compare_url", ""))
    if not row["dashboard_compare_path"] and row.get("image_id"):
        row["dashboard_compare_path"] = f"/compare?image_id={row['image_id']}"
    return row


EXPLORER_METRICS = [
    ("visual_change", "Visual change"),
    ("mean_centroid_shift_vs_baseline", "Centroid shift vs baseline"),
    ("entropy_norm_mean", "Absolute entropy"),
    ("top_5_mass_mean", "Absolute top-5 mass"),
    ("effective_area_norm_mean", "Absolute effective area"),
    ("hhi_mean", "HHI"),
    ("hoyer_sparsity_mean", "Hoyer sparsity"),
    ("region_count_thr090", "Region count thr090"),
    ("peak_count_mean_map", "Peak count"),
    ("primary_region_mass_mean", "Primary region mass"),
    ("secondary_primary_ratio_mean_map", "Secondary/primary ratio"),
    ("layer_path_length_mean", "Layer scanpath path length"),
    ("word_path_length_mean", "Word scanpath path length"),
    ("weak_grounding_candidate_score", "Weak-grounding proxy"),
]


EXPLORER_PRESETS = {
    "highest_visual_change": {"label": "Highest visual change", "metric": "visual_change", "desc": True},
    "highest_centroid_shift": {"label": "Highest centroid shift", "metric": "mean_centroid_shift_vs_baseline", "desc": True},
    "most_concentrated": {"label": "Most concentrated cases", "metric": "top_5_mass_mean", "desc": True},
    "most_diffuse": {"label": "Most diffuse cases", "metric": "entropy_norm_mean", "desc": True},
    "most_fragmented": {"label": "Most fragmented cases", "metric": "region_count_thr090", "desc": True},
    "most_stable": {"label": "Most stable cases", "metric": "visual_change", "desc": False},
    "bbox_concentrated": {"label": "BBox-like concentrated cases", "metric": "top_5_mass_mean", "desc": True, "bbox": "strict"},
    "bbox_diffuse": {"label": "BBox-like diffuse cases", "metric": "entropy_norm_mean", "desc": True, "bbox": "strict"},
    "high_visual_no_bbox": {"label": "High visual change but no bbox", "metric": "visual_change", "desc": True, "bbox": "none"},
    "bbox_unexpected": {"label": "BBox unexpected prompt cases", "metric": "visual_change", "desc": True, "bbox_unexpected": True},
}


def build_v6_explorer_context() -> dict[str, object]:
    rows, columns, warning = read_csv_dicts(V6_ANALYSIS_DIR / "case_features_800_v2.csv")
    warnings = [warning] if warning else []
    prompts = sorted({row.get("prompt_label", "") for row in rows if row.get("prompt_label")})
    preset_key = request.args.get("preset", "")
    preset = EXPLORER_PRESETS.get(preset_key, {})
    metric = request.args.get("metric") or str(preset.get("metric") or "visual_change")
    prompt = request.args.get("prompt", "")
    bbox = request.args.get("bbox") or str(preset.get("bbox") or "all")
    sort_dir = request.args.get("sort") or ("desc" if preset.get("desc", True) else "asc")
    limit = request.args.get("limit", default=50, type=int)
    limit = max(10, min(limit, 500))

    filtered = [dict(row) for row in rows]
    if prompt:
        filtered = [row for row in filtered if row.get("prompt_label") == prompt]
    if bbox == "strict":
        filtered = [row for row in filtered if safe_int(row.get("bbox_strict"), 0)]
    elif bbox == "broad":
        filtered = [row for row in filtered if safe_int(row.get("bbox_broad"), 0)]
    elif bbox == "none":
        filtered = [row for row in filtered if not safe_int(row.get("bbox_broad"), 0)]
    if preset.get("bbox_unexpected"):
        filtered = [row for row in filtered if safe_int(row.get("bbox_unexpected"), 0)]
    if metric not in columns:
        warnings.append(f"selected metric not found: {metric}")
        metric = "visual_change"
    filtered = numeric_sort(filtered, metric, reverse=(sort_dir != "asc"))
    total = len(filtered)
    shown = [add_case_links(row) for row in filtered[:limit]]

    return {
        "warnings": warnings,
        "rows": shown,
        "total": total,
        "columns": columns,
        "prompts": prompts,
        "metric": metric,
        "prompt": prompt,
        "bbox": bbox,
        "sort": sort_dir,
        "limit": limit,
        "preset": preset_key,
        "presets": EXPLORER_PRESETS,
        "metric_options": [(key, label) for key, label in EXPLORER_METRICS if key in columns],
        "fmt": fmt_compact,
    }


def explorer_csv_response(context: dict[str, object]) -> Response:
    output = io.StringIO()
    columns = [
        "case_id",
        "image_id",
        "prompt_label",
        context["metric"],
        "visual_change",
        "mean_centroid_shift_vs_baseline",
        "entropy_norm_mean",
        "top_5_mass_mean",
        "effective_area_norm_mean",
        "bbox_strict",
        "bbox_broad",
        "dashboard_case_path",
        "dashboard_compare_path",
    ]
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in context["rows"]:
        writer.writerow(row)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=v6_explorer_filtered.csv"},
    )


def v6_nav_items() -> list[dict[str, str]]:
    return [
        {"label": "Overview", "endpoint": "analysis_v6_view"},
        {"label": "Findings", "endpoint": "analysis_v6_findings"},
        {"label": "Prompts", "endpoint": "analysis_v6_prompts"},
        {"label": "Images", "endpoint": "analysis_v6_images"},
        {"label": "BBox", "endpoint": "analysis_v6_bbox"},
        {"label": "Model locations", "endpoint": "analysis_v6_model_locations"},
        {"label": "Cases", "endpoint": "analysis_v6_cases"},
        {"label": "Explorer", "endpoint": "analysis_v6_explorer"},
    ]


def load_questions_context() -> dict[str, object]:
    warnings: list[str] = []
    qa_rows, _, warning = read_csv_dicts(V6_QUESTIONS_DIR / "question_answer_evidence.csv")
    if warning:
        warnings.append(warning)
    fingerprint_rows, _, warning = read_csv_dicts(V6_QUESTIONS_DIR / "prompt_fingerprint.csv")
    if warning:
        warnings.append(warning)
    family_rows, _, warning = read_csv_dicts(V6_QUESTIONS_DIR / "metric_family_clusters.csv")
    if warning:
        warnings.append(warning)
    plot_specs = [
        ("prompt_fingerprint_heatmap.png", "Prompt fingerprint heatmap", "Z-scored metric families by prompt."),
        ("prompt_behavior_matrix.png", "Prompt behavior matrix", "Prompt ranks by metric family; rank 1 means strongest."),
        ("metric_correlation_heatmap.png", "Metric correlation heatmap", "Spearman correlations among core case-level metrics."),
        ("concordant_discordant_quadrants.png", "Concordant/discordant quadrants", "Visual change versus concentration score, colored by bbox flag."),
        ("image_structure_placeholder_or_summary.png", "Image structure summary", "Preliminary prompt sensitivity stratified by COCO structure features."),
    ]
    plots = []
    for filename, title, caption in plot_specs:
        exists = plot_exists(filename)
        if not exists:
            warnings.append(f"missing plot: {filename}")
        plots.append({"filename": filename, "title": title, "caption": caption, "exists": exists})
    return {"warnings": warnings, "qa_rows": qa_rows, "fingerprint_rows": fingerprint_rows, "family_rows": family_rows, "plots": plots, "fmt": fmt_compact}


def load_v6_prompts_context() -> dict[str, object]:
    base_context = build_analysis_v6_context()
    fingerprint_rows, _, warning = read_csv_dicts(V6_QUESTIONS_DIR / "prompt_fingerprint.csv")
    warnings = list(base_context.get("warnings", []))
    if warning:
        warnings.append(warning)
    sep_nb_rows, _, warning = read_csv_dicts(V6_ANALYSIS_DIR / "tables" / "prompt_separability_scores_nonbaseline.csv")
    if warning:
        warnings.append(warning)
    return {
        **base_context,
        "warnings": warnings,
        "fingerprint_rows": fingerprint_rows,
        "sep_nb_rows": sep_nb_rows,
        "prompt_plots": [
            {"filename": "prompt_fingerprint_heatmap.png", "title": "Prompt fingerprint heatmap"},
            {"filename": "prompt_behavior_matrix.png", "title": "Prompt behavior matrix"},
            {"filename": "prompt_classifier_confusion_matrix_nonbaseline.png", "title": "Nonbaseline confusion matrix"},
            {"filename": "absolute_concentration_by_prompt.png", "title": "Absolute concentration"},
            {"filename": "absolute_diffusion_by_prompt.png", "title": "Absolute diffusion"},
        ],
    }


def load_v6_images_context() -> dict[str, object]:
    warnings: list[str] = []
    image_rows, _, warning = read_csv_dicts(V6_ANALYSIS_DIR / "tables" / "image_sensitivity_ranking.csv")
    if warning:
        warnings.append(warning)
    structure_rows, _, warning = read_csv_dicts(V6_QUESTIONS_DIR / "image_structure_features.csv")
    if warning:
        warnings.append(warning)
    structure_join_rows, _, warning = read_csv_dicts(V6_QUESTIONS_DIR / "image_structure_sensitivity_join.csv")
    if warning:
        warnings.append(warning)
    object_summary, _, warning = read_csv_dicts(V6_QUESTIONS_DIR / "image_structure_summary_by_object_count.csv")
    if warning:
        warnings.append(warning)
    area_summary, _, warning = read_csv_dicts(V6_QUESTIONS_DIR / "image_structure_summary_by_largest_area.csv")
    if warning:
        warnings.append(warning)
    case_object_summary, _, warning = read_csv_dicts(V6_QUESTIONS_DIR / "image_structure_case_summary_by_object_count.csv")
    if warning:
        warnings.append(warning)
    top_sensitive = sorted(image_rows, key=lambda row: safe_float(row.get("sensitivity_rank"), 999999) or 999999)[:20]
    top_stable = sorted(image_rows, key=lambda row: safe_float(row.get("stability_rank"), 999999) or 999999)[:20]
    for row in top_sensitive + top_stable:
        row["compare_path"] = dashboard_relative_url(row.get("compare_link", ""))
    return {
        "warnings": warnings,
        "top_sensitive": top_sensitive,
        "top_stable": top_stable,
        "structure_rows": structure_rows[:30],
        "structure_join_rows": structure_join_rows[:20],
        "structure_count": len(structure_rows),
        "object_summary": object_summary,
        "area_summary": area_summary,
        "case_object_summary": case_object_summary,
        "fmt": fmt_compact,
    }


def load_v6_bbox_context() -> dict[str, object]:
    warnings: list[str] = []
    bbox_rows, _, warning = read_csv_dicts(V6_ANALYSIS_DIR / "tables" / "bbox_by_prompt.csv")
    if warning:
        warnings.append(warning)
    bbox_metric_rows, _, warning = read_csv_dicts(V6_ANALYSIS_DIR / "tables" / "bbox_metric_comparison.csv")
    if warning:
        warnings.append(warning)
    concordant_rows, _, warning = read_csv_dicts(V6_QUESTIONS_DIR / "concordant_cases.csv")
    if warning:
        warnings.append(warning)
    discordant_rows, _, warning = read_csv_dicts(V6_QUESTIONS_DIR / "discordant_cases.csv")
    if warning:
        warnings.append(warning)
    discordant = [add_case_links(row) for row in discordant_rows if row.get("reason") == "bbox_style_but_diffuse_tam"][:12]
    return {
        "warnings": warnings,
        "bbox_rows": bbox_rows,
        "bbox_metric_rows": bbox_metric_rows,
        "concordant_rows": [add_case_links(row) for row in concordant_rows[:12]],
        "discordant_rows": discordant,
        "fmt": fmt_compact,
        "rate": rate_value,
    }


def load_model_locations_context(connection) -> dict[str, object]:
    rows = []
    prompt_counts: dict[str, int] = {}
    query = """
        SELECT c.case_id, c.image_id, c.image_stem, c.image_label, c.prompt_label,
               o.response_text, o.bbox_style_output_flag, o.coordinate_token_count
        FROM cases c
        LEFT JOIN output_diagnostics o ON o.case_id=c.case_id
        ORDER BY c.image_id, c.prompt_label, c.case_id
    """
    for row in connection.execute(query):
        locations = parse_model_locations(row["response_text"] or "")
        if not locations:
            continue
        first = locations[0]
        item = {
            "case_id": row["case_id"],
            "image_id": row["image_id"],
            "image_label": row["image_label"],
            "prompt_label": row["prompt_label"],
            "location_count": len(locations),
            "first_kind": first.get("kind", ""),
            "first_label": first.get("label", ""),
            "first_raw": first.get("raw", ""),
            "response_preview": compact_preview(row["response_text"] or "", 180),
            "bbox_style_output_flag": row["bbox_style_output_flag"],
            "coordinate_token_count": row["coordinate_token_count"],
            "dashboard_case_path": f"/case/{row['case_id']}",
            "dashboard_compare_path": f"/compare?image_id={row['image_id']}",
            "overlay_path": f"/render/model-location/{row['case_id']}.jpg",
        }
        prompt_counts[str(row["prompt_label"])] = prompt_counts.get(str(row["prompt_label"]), 0) + 1
        rows.append(item)
    prompt_rows = [{"prompt_label": key, "parseable_count": value} for key, value in sorted(prompt_counts.items())]
    return {
        "rows": rows,
        "prompt_rows": prompt_rows,
        "total": len(rows),
        "fmt": fmt_compact,
    }


def load_v6_cases_context() -> dict[str, object]:
    warnings: list[str] = []
    reps_rows, _, warning = read_csv_dicts(V6_ANALYSIS_DIR / "tables" / "representative_cases.csv")
    if warning:
        warnings.append(warning)
    concordant_rows, _, warning = read_csv_dicts(V6_QUESTIONS_DIR / "concordant_cases.csv")
    if warning:
        warnings.append(warning)
    discordant_rows, _, warning = read_csv_dicts(V6_QUESTIONS_DIR / "discordant_cases.csv")
    if warning:
        warnings.append(warning)
    return {
        "warnings": warnings,
        "representatives": [add_case_links(row) for row in reps_rows[:40]],
        "concordant_rows": [add_case_links(row) for row in concordant_rows[:30]],
        "discordant_rows": [add_case_links(row) for row in discordant_rows[:40]],
        "fmt": fmt_compact,
    }


def build_analysis_v6_context() -> dict[str, object]:
    base = V6_ANALYSIS_DIR
    table_dir = base / "tables"
    plot_dir = base / "plots"
    warnings: list[str] = []

    case_rows, case_cols, warning = read_csv_dicts(base / "case_features_800_v2.csv")
    if warning:
        warnings.append(warning)
    prompt_rows, prompt_cols, warning = read_csv_dicts(table_dir / "prompt_summary.csv")
    if warning:
        warnings.append(warning)
    image_rows, image_cols, warning = read_csv_dicts(table_dir / "image_sensitivity_ranking.csv")
    if warning:
        warnings.append(warning)
    bbox_rows, bbox_cols, warning = read_csv_dicts(table_dir / "bbox_by_prompt.csv")
    if warning:
        warnings.append(warning)
    bbox_metric_rows, bbox_metric_cols, warning = read_csv_dicts(table_dir / "bbox_metric_comparison.csv")
    if warning:
        warnings.append(warning)
    sep_rows, sep_cols, warning = read_csv_dicts(table_dir / "prompt_separability_scores.csv")
    if warning:
        warnings.append(warning)
    sep_nb_rows, sep_nb_cols, warning = read_csv_dicts(table_dir / "prompt_separability_scores_nonbaseline.csv")
    if warning:
        warnings.append(warning)
    reps_rows, reps_cols, warning = read_csv_dicts(table_dir / "representative_cases.csv")
    if warning:
        warnings.append(warning)
    manifest = read_json_file(base / "manifest.json")

    distinct_images = len({row.get("image_id") for row in case_rows if row.get("image_id")})
    distinct_prompts = len({row.get("prompt_label") for row in case_rows if row.get("prompt_label")})
    row_counts = manifest.get("row_counts", {}) if isinstance(manifest, dict) else {}
    bbox_strict = sum(safe_int(row.get("bbox_strict"), 0) or 0 for row in case_rows)
    bbox_broad = sum(safe_int(row.get("bbox_broad"), 0) or 0 for row in case_rows)
    nb_tam = row_for_variant(sep_nb_rows, "tam_only")
    nb_text = row_for_variant(sep_nb_rows, "tam_plus_text")

    kpis = [
        {"label": "cases", "value": f"{len(case_rows):,}" if case_rows else fmt_compact(row_counts.get("actual_cases"), 0)},
        {"label": "images", "value": f"{distinct_images:,}" if distinct_images else fmt_compact(row_counts.get("distinct_images"), 0)},
        {"label": "prompts", "value": f"{distinct_prompts:,}" if distinct_prompts else fmt_compact(row_counts.get("distinct_prompts"), 0)},
        {"label": "maps", "value": fmt_compact(row_counts.get("map_count_total"), 0)},
        {"label": "map metrics", "value": fmt_compact(row_counts.get("map_metrics_count_total"), 0)},
        {"label": "missing map metrics", "value": fmt_compact(row_counts.get("missing_map_metrics_total"), 0)},
        {"label": "bbox strict", "value": f"{bbox_strict:,}" if case_rows else fmt_compact(row_counts.get("bbox_strict_count"), 0)},
        {"label": "bbox broad", "value": f"{bbox_broad:,}" if case_rows else fmt_compact(row_counts.get("bbox_broad_count"), 0)},
        {"label": "nonbaseline TAM-only BA", "value": fmt_compact(nb_tam.get("balanced_accuracy"))},
        {"label": "nonbaseline TAM+text BA", "value": fmt_compact(nb_text.get("balanced_accuracy"))},
        {"label": "nonbaseline chance", "value": fmt_compact(first_value(sep_nb_rows, "chance_balanced_accuracy"))},
    ]

    baseline_plot_specs = [
        ("heatmap_visual_change.png", "Image x prompt visual change", "Which images change most across prompts?", "Rows are images, columns are prompts, and color is the baseline-relative visual-change score.", "The highest changes concentrate in stress/object-location style prompts, but image identity still matters.", "Main evidence."),
        ("boxplot_visual_change_by_prompt.png", "Visual change by prompt", "Which prompts move TAM behavior farthest from baseline?", "Orange line is the median; the box is the middle 50 percent of cases; whiskers show spread.", "Misleading/object-detection/order-disruption prompts show the largest changes.", "Main evidence."),
        ("boxplot_entropy_mean_delta_vs_baseline_by_prompt.png", "Entropy delta vs baseline", "Does attribution become more diffuse or less diffuse?", "Positive values mean more diffuse than baseline; negative values mean less diffuse.", "Image-grounded and ambiguous prompts are relatively diffuse; misleading/object-detection prompts are less diffuse.", "Main evidence."),
        ("boxplot_top5_mass_mean_delta_vs_baseline_by_prompt.png", "Top-5 mass delta vs baseline", "Does attribution concentrate into stronger hotspots?", "Positive values mean more mass in top pixels than baseline.", "Misleading and object-detection prompts increase concentration most.", "Main evidence."),
        ("boxplot_mean_centroid_shift_vs_baseline_by_prompt.png", "Centroid shift vs baseline", "How far does attribution move spatially?", "Higher values mean the attribution centroid moves farther from baseline for the same image.", "Reasoning and misleading prompts have large average centroid shifts.", "Main evidence."),
        ("bbox_count_by_prompt.png", "BBox/location-style output by prompt", "Which prompts trigger coordinate-like output?", "Bars count text responses that look bbox/location-like.", "Order-disruption dominates bbox/location-style responses.", "Response format signal only."),
        ("prompt_classifier_confusion_matrix_nonbaseline.png", "Nonbaseline prompt separability", "Are nonbaseline prompts distinguishable from metrics?", "Rows are true prompts, columns are predicted prompts.", "Prompt classes remain separable above chance after removing baseline.", "Metric-space separability only."),
    ]
    absolute_plot_specs = [
        ("absolute_concentration_by_prompt.png", "Absolute concentration profile", "Which prompts have more localized hotspot mass?", "Higher top-5 mass and Hoyer sparsity suggest stronger concentration.", "Use with entropy/effective area; no single metric proves localization.", "General TAM profile."),
        ("absolute_diffusion_by_prompt.png", "Absolute diffusion profile", "Which prompts are more diffuse overall?", "Higher entropy/effective area means attribution is spread over more of the image.", "This is not baseline-relative; it describes the prompt's absolute TAM profile.", "General TAM profile."),
        ("absolute_hotspot_structure_by_prompt.png", "Hotspot and multipeak profile", "Which prompts create more hotspot or multipeak structure?", "Peak count and secondary/primary ratio summarize hotspot complexity.", "Use as a structure proxy, not semantic object evidence.", "General TAM profile."),
        ("absolute_region_count_by_prompt.png", "Fragmentation by region count", "Which prompts create more fragmented thresholded regions?", "Higher region counts mean more connected components at high activation thresholds.", "Region count depends on threshold and preprocessing.", "General TAM profile."),
        ("absolute_spatial_scanpath_by_prompt.png", "Spatial spread and TAM-derived scanpaths", "Which prompts have more spatially unstable attribution trajectories?", "Spread and path lengths summarize spatial dispersion across layers/words.", "TAM-derived scanpaths are not human eye-tracking.", "General TAM profile."),
        ("localization_proxy_by_prompt.png", "Heuristic localization proxy", "Which prompts look most localized by combined components?", "Higher score combines high top-5/primary mass with low entropy/effective area/region count.", "Heuristic proxy only; not ground-truth localization.", "Summary proxy."),
    ]
    plots = []
    for group, specs in (("baseline", baseline_plot_specs), ("absolute", absolute_plot_specs)):
        for filename, title, question, how_to_read, pattern, caveat in specs:
            exists = plot_exists(filename)
            if not exists:
                warnings.append(f"missing plot: {filename}")
            plots.append({"group": group, "filename": filename, "title": title, "question": question, "how_to_read": how_to_read, "pattern": pattern, "caveat": caveat, "exists": exists})
    optional_specs = [
        ("heatmap_bbox_strict.png", "BBox/location strict heatmap", "Where do bbox-style responses occur?", "Rows are images, columns are prompts, color is the strict bbox flag.", "The pattern is concentrated in the expected location-style prompts.", "Optional descriptive support."),
        ("pca_cases_by_prompt.png", "PCA of case features by prompt", "Are cases visually separated in a low-dimensional projection?", "Each point is a case projected from metric features.", "Use only as exploratory support.", "Exploratory only."),
        ("top20_prompt_sensitive_images.png", "Top prompt-sensitive images", "Which images should we inspect first?", "Bars rank images by mean visual change across prompts.", "Use this as a case-selection guide.", "Use for case selection."),
    ]
    for filename, title, question, how_to_read, pattern, caveat in optional_specs:
        exists = plot_exists(filename)
        if not exists:
            warnings.append(f"missing plot: {filename}")
        plots.append({"group": "support", "filename": filename, "title": title, "question": question, "how_to_read": how_to_read, "pattern": pattern, "caveat": caveat, "exists": exists})

    absolute_rows, absolute_cols, warning = read_csv_dicts(V6_DASHBOARD_VIEWS_DIR / "prompt_absolute_profile_with_proxy.csv")
    if warning:
        warnings.append(warning)
    metric_dictionary, _, warning = read_csv_dicts(V6_DASHBOARD_VIEWS_DIR / "metric_dictionary.csv")
    if warning:
        warnings.append(warning)

    prompt_by_label = {row.get("prompt_label"): row for row in prompt_rows}
    absolute_by_label = {row.get("prompt_label"): row for row in absolute_rows}
    prompt_cards = []
    for prompt in [row.get("prompt_label") for row in prompt_rows]:
        summary_row = prompt_by_label.get(prompt, {})
        absolute_row = absolute_by_label.get(prompt, {})
        prompt_cards.append(
            {
                "prompt_label": prompt,
                "visual_mean": summary_row.get("visual_change_mean"),
                "visual_median": summary_row.get("visual_change_median"),
                "entropy": absolute_row.get("entropy_norm_mean"),
                "top5": absolute_row.get("top_5_mass_mean"),
                "effective_area": absolute_row.get("effective_area_norm_mean"),
                "centroid_shift": summary_row.get("mean_centroid_shift_vs_baseline_mean"),
                "bbox_strict": summary_row.get("bbox_strict_rate"),
                "bbox_broad": summary_row.get("bbox_broad_rate"),
                "weak_grounding": summary_row.get("weak_grounding_candidate_score_mean"),
                "localization_proxy": absolute_row.get("localization_proxy_score"),
                "interpretation": prompt_interpretation(prompt or ""),
            }
        )

    top_sensitive = sorted(image_rows, key=lambda row: safe_float(row.get("sensitivity_rank"), 999999) or 999999)[:20]
    top_stable = sorted(image_rows, key=lambda row: safe_float(row.get("stability_rank"), 999999) or 999999)[:20]
    for row in top_sensitive + top_stable:
        row["compare_path"] = dashboard_relative_url(row.get("compare_link", ""))

    reason_order = [
        "highest_visual_change",
        "highest_centroid_shift",
        "strongest_entropy_increase",
        "strongest_concentration_increase",
        "bbox_like_concentrated_cases",
        "bbox_like_diffuse_cases",
        "stable_cases",
    ]
    representatives = []
    seen: set[tuple[str, str]] = set()
    for reason in reason_order:
        for row in reps_rows:
            key = (row.get("selection_reason", ""), row.get("case_id", ""))
            if row.get("selection_reason") == reason and key not in seen:
                row["dashboard_case_path"] = dashboard_relative_url(row.get("dashboard_case_url", ""))
                row["dashboard_compare_path"] = dashboard_relative_url(row.get("dashboard_compare_url", ""))
                representatives.append(row)
                seen.add(key)
                break
    if len(representatives) < 12:
        for row in reps_rows:
            key = (row.get("selection_reason", ""), row.get("case_id", ""))
            if key in seen:
                continue
            row["dashboard_case_path"] = dashboard_relative_url(row.get("dashboard_case_url", ""))
            row["dashboard_compare_path"] = dashboard_relative_url(row.get("dashboard_compare_url", ""))
            representatives.append(row)
            seen.add(key)
            if len(representatives) >= 12:
                break

    bbox_metric_compact = [row for row in bbox_metric_rows if row.get("comparison") in {"bbox_strict_vs_nonbbox", "expected_localization_prompt_vs_other"}]

    return {
        "base_exists": base.exists(),
        "warnings": warnings,
        "kpis": kpis,
        "plots": plots,
        "baseline_plots": [plot for plot in plots if plot["group"] == "baseline"],
        "absolute_plots": [plot for plot in plots if plot["group"] == "absolute"],
        "support_plots": [plot for plot in plots if plot["group"] == "support"],
        "prompt_rows": prompt_rows,
        "prompt_columns": prompt_cols,
        "absolute_rows": absolute_rows,
        "absolute_columns": absolute_cols,
        "metric_dictionary": metric_dictionary,
        "prompt_cards": prompt_cards,
        "image_columns": image_cols,
        "top_sensitive": top_sensitive,
        "top_stable": top_stable,
        "bbox_rows": bbox_rows,
        "bbox_columns": bbox_cols,
        "bbox_metric_rows": bbox_metric_compact,
        "sep_rows": sep_rows,
        "sep_nb_rows": sep_nb_rows,
        "sep_columns": sep_cols,
        "sep_nb_columns": sep_nb_cols,
        "representatives": representatives,
        "fmt": fmt_compact,
        "rate": rate_value,
    }


def create_app(config: DashboardConfig | None = None) -> Flask:
    config = config or DashboardConfig.default()
    config.ensure_dirs()
    app = Flask(__name__)
    app.config["DASHBOARD_CONFIG"] = config

    def conn():
        connection = connect(config.db_path)
        initialize(connection, rebuild=False)
        return connection

    @app.context_processor
    def inject_metric_helpers():
        return {
            "metric_info": metric_info,
            "metric_registry": METRIC_REGISTRY,
            "metric_group_descriptions": GROUP_DESCRIPTIONS,
            "fmt_metric": format_metric_value,
        }

    @app.route("/")
    def index():
        connection = conn()
        official = "1" in request.args.getlist("official")
        image = request.args.get("image", "")
        clauses = []
        params = []
        if official:
            clauses.append("is_official=1")
        if image:
            clauses.append("(image_label LIKE ? OR image_stem LIKE ? OR condition_label LIKE ?)")
            params.extend([f"%{image}%", f"%{image}%", f"%{image}%"])
        query = "SELECT * FROM cases"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY image_stem, condition_label, run_name"
        cases = connection.execute(query, params).fetchall()
        coverage = get_coverage(connection)
        case_summaries = get_case_metric_summaries(connection, [row["case_id"] for row in cases])
        rankings = get_global_rankings(connection)
        selected_case_studies = load_selected_case_studies()
        return render_template(
            "index.html",
            cases=cases,
            official=official,
            image=image,
            coverage=coverage,
            case_summaries=case_summaries,
            rankings=rankings,
            selected_case_studies=selected_case_studies,
        )

    @app.route("/prompts")
    def prompt_guide():
        connection = conn()
        focus = request.args.get("focus", "")
        rows = connection.execute(
            """
            SELECT condition_label, prompt_label, COUNT(*) AS case_count, MIN(metadata_path) AS metadata_path
            FROM cases
            GROUP BY condition_label, prompt_label
            ORDER BY condition_label
            """
        ).fetchall()
        prompts = []
        for row in rows:
            prompt_text = ""
            metadata_path = resolve_project_path(row["metadata_path"], config.project_root)
            if metadata_path.exists():
                try:
                    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                    prompt_text = str(payload.get("prompt_text") or "")
                except (OSError, json.JSONDecodeError):
                    prompt_text = ""
            label = str(row["condition_label"])
            prompts.append(
                {
                    "condition_label": label,
                    "prompt_label": row["prompt_label"],
                    "prompt_text": prompt_text or "not documented yet",
                    "purpose": condition_purpose(label),
                    "family": condition_family(label),
                    "case_count": row["case_count"],
                    "metric_summary": get_condition_metric_summary(connection, label),
                }
            )
        return render_template("prompts.html", prompts=prompts, focus=focus)

    @app.route("/metrics-guide")
    def metrics_guide():
        groups = []
        for title, keys in GUIDE_GROUPS:
            groups.append({"title": title, "rows": [metric_info(key) for key in keys]})
        return render_template("metrics_guide.html", groups=groups, optional_gt=OPTIONAL_GT, future_metrics=FUTURE_METRICS)

    @app.route("/metrics")
    def metrics_rankings():
        connection = conn()
        return render_template("metrics_rankings.html", rankings=get_global_rankings(connection))

    @app.route("/analysis/v6")
    def analysis_v6_view():
        return render_template("analysis_v6.html", analysis=build_analysis_v6_context(), v6_nav=v6_nav_items())

    @app.route("/analysis")
    def analysis_hub():
        return render_template("analysis_hub.html")

    @app.route("/analysis/v6/findings")
    def analysis_v6_findings():
        return render_template("analysis_v6_findings.html", findings=load_questions_context(), v6_nav=v6_nav_items())

    @app.route("/analysis/v6/prompts")
    def analysis_v6_prompts():
        return render_template("analysis_v6_prompts.html", prompts=load_v6_prompts_context(), v6_nav=v6_nav_items())

    @app.route("/analysis/v6/images")
    def analysis_v6_images():
        return render_template("analysis_v6_images.html", images=load_v6_images_context(), v6_nav=v6_nav_items())

    @app.route("/analysis/v6/bbox")
    def analysis_v6_bbox():
        return render_template("analysis_v6_bbox.html", bbox=load_v6_bbox_context(), v6_nav=v6_nav_items())

    @app.route("/analysis/v6/model-locations")
    def analysis_v6_model_locations():
        connection = conn()
        return render_template(
            "analysis_v6_model_locations.html",
            locations=load_model_locations_context(connection),
            v6_nav=v6_nav_items(),
        )

    @app.route("/analysis/v6/cases")
    def analysis_v6_cases():
        return render_template("analysis_v6_cases.html", cases=load_v6_cases_context(), v6_nav=v6_nav_items())

    @app.route("/analysis/v6/explorer")
    def analysis_v6_explorer():
        context = build_v6_explorer_context()
        if request.args.get("format") == "csv":
            return explorer_csv_response(context)
        return render_template("analysis_v6_explorer.html", explorer=context, v6_nav=v6_nav_items())

    @app.route("/analysis/v6/plot/<filename>")
    def analysis_v6_plot(filename: str):
        path = analysis_plot_path(filename)
        if path is None:
            abort(404)
        return send_file(path, mimetype="image/png")

    @app.route("/queries")
    def saved_queries_view():
        queries = load_saved_queries()
        selected_slug = request.args.get("query") or (queries[0]["slug"] if queries else "")
        row_limit = request.args.get("rows", default=200, type=int)
        row_limit = max(25, min(row_limit, 1000))
        selected = load_saved_query_detail(selected_slug, row_limit=row_limit) if selected_slug else None
        return render_template("queries.html", queries=queries, selected=selected, row_limit=row_limit)

    @app.route("/queries/<slug>/download/<filename>")
    def saved_query_download(slug: str, filename: str):
        if filename not in {"answer.md", "question.md", "manifest.json", "results.csv", "results.parquet"}:
            abort(404)
        path = saved_query_dir(slug) / filename
        if not path.exists():
            abort(404)
        return send_file(path, as_attachment=True)

    @app.route("/case/<case_id>")
    def case_view(case_id: str):
        connection = conn()
        case = get_case(connection, case_id)
        if not case:
            abort(404)
        words = get_case_words(connection, case_id)
        layers = [row["layer_index"] for row in connection.execute("SELECT DISTINCT layer_index FROM maps WHERE case_id=? ORDER BY layer_index", (case_id,))]
        selected_layer = int(request.args.get("layer", layers[-1] if layers else 0))
        metrics = connection.execute(
            "SELECT AVG(entropy_norm) AS entropy, AVG(secondary_primary_ratio) AS secondary_ratio, AVG(peak_count) AS peak_count FROM map_metrics WHERE case_id=?",
            (case_id,),
        ).fetchone()
        metrics_count = connection.execute("SELECT COUNT(*) AS n FROM map_metrics WHERE case_id=?", (case_id,)).fetchone()["n"]
        metadata = load_case_metadata(config, case)
        model_locations = parse_model_locations(metadata.get("response_text", ""))
        selected_word = words[0] if words else None
        selected_word_index = int(request.args.get("word", selected_word["word_index"] if selected_word else 0))
        selected_map_metrics = get_selected_map_metrics(connection, case_id, selected_word_index, selected_layer)
        region_rows = get_region_rows(connection, case_id, selected_word_index, selected_layer)
        layer_scan_metrics = connection.execute("SELECT * FROM layer_scanpaths WHERE case_id=? AND word_index=?", (case_id, selected_word_index)).fetchone()
        case_summary = get_case_summary_metrics(connection, case_id)
        highlights = get_case_metric_highlights(connection, case_id)
        return render_template(
            "case.html",
            case=case,
            words=words,
            layers=layers,
            selected_layer=selected_layer,
            metrics=metrics,
            metrics_count=metrics_count,
            metadata=metadata,
            model_locations=model_locations,
            selected_word=selected_word,
            selected_word_index=selected_word_index,
            selected_map_metrics=selected_map_metrics,
            region_rows=region_rows,
            layer_scan_metrics=layer_scan_metrics,
            case_summary=case_summary,
            highlights=highlights,
        )

    @app.route("/case/<case_id>/matrix")
    def matrix_view(case_id: str):
        connection = conn()
        case = get_case(connection, case_id)
        if not case:
            abort(404)
        words = get_case_words(connection, case_id)
        layers = [row["layer_index"] for row in connection.execute("SELECT DISTINCT layer_index FROM maps WHERE case_id=? ORDER BY layer_index", (case_id,))]
        normalization = request.args.get("norm", "local")
        zoom = request.args.get("zoom", "overview")
        # Fit mode is displayed with adaptive CSS; render thumbnails at a moderate
        # source size so they stay sharp when the viewport has room.
        cell_sizes = {"overview": 64, "fit": 72, "compact": 56, "normal": 88, "large": 132}
        cell_size = cell_sizes.get(zoom, 88)
        metric_rows = connection.execute(
            "SELECT word_index, layer_index, energy_sum, entropy_norm, peak_count, top_5_mass, effective_area_norm, spread_trace, secondary_primary_ratio, global_centroid_x_norm, global_centroid_y_norm FROM map_metrics WHERE case_id=?",
            (case_id,),
        ).fetchall()
        cell_metrics = {
            f"{row['word_index']}:{row['layer_index']}": {
                "energy_sum": row["energy_sum"],
                "entropy_norm": row["entropy_norm"],
                "peak_count": row["peak_count"],
                "top_5_mass": row["top_5_mass"],
                "effective_area_norm": row["effective_area_norm"],
                "spread_trace": row["spread_trace"],
                "secondary_primary_ratio": row["secondary_primary_ratio"],
                "global_centroid_x_norm": row["global_centroid_x_norm"],
                "global_centroid_y_norm": row["global_centroid_y_norm"],
            }
            for row in metric_rows
        }
        return render_template(
            "matrix.html",
            case=case,
            words=words,
            layers=layers,
            normalization=normalization,
            zoom=zoom,
            cell_size=cell_size,
            cell_metrics=cell_metrics,
        )

    @app.route("/case/<case_id>/word/<int:word_index>")
    def word_view(case_id: str, word_index: int):
        connection = conn()
        case = get_case(connection, case_id)
        if not case:
            abort(404)
        word = connection.execute("SELECT * FROM words WHERE case_id=? AND word_index=?", (case_id, word_index)).fetchone()
        if not word:
            abort(404)
        layers = connection.execute(
            """
            SELECT m.*, mm.energy_sum, mm.energy_mean, mm.max_value, mm.entropy_norm, mm.top_5_mass,
                   mm.effective_area_norm, mm.global_centroid_x_norm, mm.global_centroid_y_norm,
                   mm.spread_trace, mm.peak_count, mm.secondary_primary_ratio
            FROM maps m
            LEFT JOIN map_metrics mm ON m.case_id=mm.case_id AND m.word_index=mm.word_index AND m.layer_index=mm.layer_index
            WHERE m.case_id=? AND m.word_index=?
            ORDER BY m.layer_index
            """,
            (case_id, word_index),
        ).fetchall()
        scanpath = connection.execute("SELECT * FROM layer_scanpaths WHERE case_id=? AND word_index=?", (case_id, word_index)).fetchone()
        scanpath_groups = build_grouped_metrics(
            scanpath,
            {
                "movement": ["layer_path_length", "layer_mean_step", "layer_max_jump", "layer_net_displacement", "layer_tortuosity", "layer_large_jump_count", "layer_bbox_area"],
                "adjacent similarity": ["adjacent_layer_cosine_mean", "adjacent_layer_cosine_min", "adjacent_layer_ssim_mean", "adjacent_layer_jsd_mean", "adjacent_layer_emd_mean", "adjacent_layer_top5_iou_mean"],
                "early vs late": ["early_late_cosine", "early_late_jsd", "early_late_centroid_shift", "early_late_spread_delta"],
                "multipeak": ["peak_count_mean", "peak_count_max", "secondary_primary_ratio_mean", "secondary_primary_ratio_max", "multipeak_layer_count", "multipeak_layer_ratio"],
            },
        )
        metrics_count = connection.execute(
            "SELECT COUNT(*) AS n FROM map_metrics WHERE case_id=? AND word_index=?",
            (case_id, word_index),
        ).fetchone()["n"]
        return render_template("word.html", case=case, word=word, layers=layers, scanpath=scanpath, scanpath_groups=scanpath_groups, metrics_count=metrics_count)

    @app.route("/compare")
    def compare_view():
        connection = conn()
        image_id = request.args.get("image_id", type=int)
        cases = connection.execute("SELECT * FROM cases WHERE is_official=1 ORDER BY image_stem, condition_label").fetchall()
        image_cases = []
        if image_id is not None:
            image_cases = connection.execute("SELECT * FROM cases WHERE image_id=? AND is_official=1 ORDER BY condition_label", (image_id,)).fetchall()
        case_a = request.args.get("a")
        case_b = request.args.get("b")
        if image_cases and not case_a:
            baseline = next((case for case in image_cases if case["condition_label"] == "baseline_neutral"), None)
            case_a = baseline["case_id"] if baseline else image_cases[0]["case_id"]
        if image_cases and case_a and not case_b:
            case_b = next((case["case_id"] for case in image_cases if case["case_id"] != case_a), None)
        word_index = request.args.get("word", default=0, type=int)
        layer_index = request.args.get("layer", default=28, type=int)
        pair = None
        aggregate = None
        if case_a and case_b:
            row_a = get_map_row(connection, case_a, word_index, layer_index)
            row_b = get_map_row(connection, case_b, word_index, layer_index)
            if row_a and row_b:
                pair = compute_pair_for_rows(connection, config, row_a, row_b, include_emd=request.args.get("emd") == "1")
            aggregate = aggregate_condition_metrics(connection, case_a, case_b)
        grouped_pair = group_pair_metrics(pair) if pair else None
        return render_template(
            "compare.html",
            cases=cases,
            image_cases=image_cases,
            image_id=image_id,
            case_a=case_a,
            case_b=case_b,
            word_index=word_index,
            layer_index=layer_index,
            pair=pair,
            grouped_pair=grouped_pair,
            aggregate=aggregate,
        )

    @app.route("/render/map/<case_id>/<int:word_index>/<int:layer_index>.png")
    def render_map_route(case_id: str, word_index: int, layer_index: int):
        connection = conn()
        path = render_map(
            connection,
            config,
            case_id,
            word_index,
            layer_index,
            mode=request.args.get("mode", "overlay"),
            threshold=request.args.get("threshold", default=0.90, type=float),
            normalization=request.args.get("norm", "local"),
        )
        return send_file(path)

    @app.route("/render/cell/<case_id>/<int:word_index>/<int:layer_index>.jpg")
    def render_cell_route(case_id: str, word_index: int, layer_index: int):
        connection = conn()
        path = render_matrix_cell(
            connection,
            config,
            case_id,
            word_index,
            layer_index,
            normalization=request.args.get("norm", "local"),
            size=request.args.get("size", default=120, type=int),
        )
        return send_file(path)

    @app.route("/render/final-preview/<case_id>.jpg")
    def render_final_preview_route(case_id: str):
        connection = conn()
        path = render_final_layer_preview(
            connection,
            config,
            case_id,
            max_words=request.args.get("max_words", default=12, type=int),
            threshold=request.args.get("threshold", default=0.90, type=float),
        )
        return send_file(path)

    @app.route("/render/final-animation/<case_id>.gif")
    def render_final_animation_route(case_id: str):
        connection = conn()
        path = render_final_layer_animation(
            connection,
            config,
            case_id,
            max_words=request.args.get("max_words", default=32, type=int),
            duration_ms=request.args.get("duration_ms", default=450, type=int),
        )
        return send_file(path)

    @app.route("/render/original/<case_id>.jpg")
    def render_original_route(case_id: str):
        connection = conn()
        case = get_case(connection, case_id)
        if not case:
            abort(404)
        path = config.project_root / case["image_path"]
        if not path.exists():
            abort(404)
        return send_file(path)

    @app.route("/render/model-location/<case_id>.jpg")
    def render_model_location_route(case_id: str):
        connection = conn()
        case = get_case(connection, case_id)
        if not case:
            abort(404)
        metadata = load_case_metadata(config, case)
        try:
            path = render_model_location_overlay(
                connection,
                config,
                case_id,
                metadata.get("response_text", ""),
                width=request.args.get("width", default=1100, type=int),
            )
        except ValueError:
            abort(404)
        return send_file(path)

    @app.route("/render/scanpath/<mode>/<case_id>.png")
    def render_scanpath_route(mode: str, case_id: str):
        if mode not in {"word", "layer"}:
            abort(404)
        connection = conn()
        path = render_scanpath(
            connection,
            config,
            case_id,
            mode=mode,
            layer_index=request.args.get("layer", type=int),
            word_index=request.args.get("word", type=int),
            threshold=request.args.get("threshold", default=0.90, type=float),
        )
        return send_file(path)

    @app.route("/render/diff/<case_id_a>/<case_id_b>/<int:word_index>/<int:layer_index>.png")
    def render_diff_route(case_id_a: str, case_id_b: str, word_index: int, layer_index: int):
        connection = conn()
        path = render_difference(
            connection,
            config,
            case_id_a,
            case_id_b,
            word_index,
            layer_index,
            mode=request.args.get("mode", "absolute"),
        )
        return send_file(path)

    @app.route("/api/case/<case_id>/words")
    def api_words(case_id: str):
        connection = conn()
        rows = get_case_words(connection, case_id)
        return {"words": [dict(row) for row in rows]}

    @app.route("/api/case/<case_id>/map_metrics")
    def api_map_metrics(case_id: str):
        connection = conn()
        word_index = request.args.get("word_index", type=int)
        layer_index = request.args.get("layer_index", type=int)
        if word_index is None or layer_index is None:
            return {"error": "word_index and layer_index are required"}, 400
        row = get_selected_map_metrics(connection, case_id, word_index, layer_index)
        return {"metrics": row or {}}

    @app.route("/api/case/<case_id>/word_metrics/<int:word_index>")
    def api_word_metrics(case_id: str, word_index: int):
        connection = conn()
        row = connection.execute("SELECT * FROM layer_scanpaths WHERE case_id=? AND word_index=?", (case_id, word_index)).fetchone()
        return {"metrics": dict(row) if row else {}}

    @app.route("/api/case/<case_id>/summary_metrics")
    def api_summary_metrics(case_id: str):
        connection = conn()
        return {"summary": get_case_summary_metrics(connection, case_id)}

    @app.route("/api/case/<case_id>/metric_highlights")
    def api_metric_highlights(case_id: str):
        connection = conn()
        return {"highlights": get_case_metric_highlights(connection, case_id)}

    @app.route("/api/case/<case_id>/regions")
    def api_regions(case_id: str):
        connection = conn()
        word_index = request.args.get("word_index", type=int)
        layer_index = request.args.get("layer_index", type=int)
        if word_index is None or layer_index is None:
            return {"error": "word_index and layer_index are required"}, 400
        return {"regions": get_region_rows(connection, case_id, word_index, layer_index)}

    return app


def get_coverage(connection) -> dict:
    names = ["cases", "maps", "map_metrics", "regions", "layer_scanpaths", "word_scanpaths", "map_pairs", "cache_manifest"]
    out = {}
    for name in names:
        out[name] = int(connection.execute(f"SELECT COUNT(*) AS n FROM {name}").fetchone()["n"])
    out["map_metrics_coverage"] = (out["map_metrics"] / out["maps"]) if out["maps"] else 0.0
    return out


def get_case_metric_summaries(connection, case_ids: list[str]) -> dict[str, dict]:
    if not case_ids:
        return {}
    placeholders = ",".join("?" for _ in case_ids)
    rows = connection.execute(
        f"""
        SELECT case_id, AVG(entropy_norm) AS mean_entropy, AVG(secondary_primary_ratio) AS mean_secondary_ratio, AVG(peak_count) AS mean_peak_count
        FROM map_metrics
        WHERE case_id IN ({placeholders})
        GROUP BY case_id
        """,
        case_ids,
    ).fetchall()
    word_rows = connection.execute(
        f"SELECT case_id, AVG(word_path_length) AS mean_word_path_length FROM word_scanpaths WHERE case_id IN ({placeholders}) GROUP BY case_id",
        case_ids,
    ).fetchall()
    out = {row["case_id"]: dict(row) for row in rows}
    for row in word_rows:
        out.setdefault(row["case_id"], {})["mean_word_path_length"] = row["mean_word_path_length"]
    return out


def get_condition_metric_summary(connection, condition_label: str) -> dict:
    row = connection.execute(
        """
        SELECT
          AVG(mm.entropy_norm) AS mean_entropy,
          AVG(mm.secondary_primary_ratio) AS mean_secondary_ratio,
          AVG(mm.peak_count) AS mean_peak_count
        FROM map_metrics mm
        JOIN cases c ON c.case_id=mm.case_id
        WHERE c.condition_label=?
        """,
        (condition_label,),
    ).fetchone()
    word = connection.execute(
        """
        SELECT AVG(ws.word_path_length) AS mean_word_path_length
        FROM word_scanpaths ws
        JOIN cases c ON c.case_id=ws.case_id
        WHERE c.condition_label=?
        """,
        (condition_label,),
    ).fetchone()
    return {
        "mean_entropy": row["mean_entropy"] if row else None,
        "mean_secondary_ratio": row["mean_secondary_ratio"] if row else None,
        "mean_peak_count": row["mean_peak_count"] if row else None,
        "mean_word_path_length": word["mean_word_path_length"] if word else None,
    }


def get_selected_map_metrics(connection, case_id: str, word_index: int, layer_index: int) -> dict | None:
    row = connection.execute("SELECT * FROM map_metrics WHERE case_id=? AND word_index=? AND layer_index=?", (case_id, word_index, layer_index)).fetchone()
    return dict(row) if row else None


def get_region_rows(connection, case_id: str, word_index: int, layer_index: int) -> list[dict]:
    rows = connection.execute(
        """
        SELECT threshold, rank, mass, ratio_to_primary, centroid_x_norm, centroid_y_norm,
               bbox_x0_norm, bbox_y0_norm, bbox_x1_norm, bbox_y1_norm, area, peak_value
        FROM regions
        WHERE case_id=? AND word_index=? AND layer_index=?
        ORDER BY threshold, rank
        LIMIT 200
        """,
        (case_id, word_index, layer_index),
    ).fetchall()
    return [dict(row) for row in rows]


def get_case_summary_metrics(connection, case_id: str) -> dict:
    row = connection.execute(
        """
        SELECT
          AVG(entropy_norm) AS mean_entropy,
          AVG(top_5_mass) AS mean_top5_mass,
          AVG(effective_area_norm) AS mean_effective_area_norm,
          AVG(peak_count) AS mean_peak_count,
          MAX(peak_count) AS max_peak_count,
          AVG(secondary_primary_ratio) AS mean_secondary_ratio,
          MAX(secondary_primary_ratio) AS max_secondary_ratio,
          AVG(spread_trace) AS mean_spread_trace,
          AVG(is_multipeak) AS multipeak_ratio
        FROM map_metrics
        WHERE case_id=?
        """,
        (case_id,),
    ).fetchone()
    w = connection.execute(
        "SELECT AVG(word_path_length) AS avg_word_path_length, MAX(word_max_jump) AS max_word_jump FROM word_scanpaths WHERE case_id=?",
        (case_id,),
    ).fetchone()
    return {
        "mean_entropy": row["mean_entropy"],
        "mean_top5_mass": row["mean_top5_mass"],
        "mean_effective_area_norm": row["mean_effective_area_norm"],
        "mean_peak_count": row["mean_peak_count"],
        "max_peak_count": row["max_peak_count"],
        "mean_secondary_ratio": row["mean_secondary_ratio"],
        "max_secondary_ratio": row["max_secondary_ratio"],
        "mean_spread_trace": row["mean_spread_trace"],
        "multipeak_ratio": row["multipeak_ratio"],
        "avg_word_path_length": w["avg_word_path_length"] if w else None,
        "max_word_jump": w["max_word_jump"] if w else None,
    }


def get_case_metric_highlights(connection, case_id: str) -> dict:
    def row(query: str):
        value = connection.execute(query, (case_id,)).fetchone()
        return dict(value) if value else None

    return {
        "most_diffuse": row("SELECT word_index, layer_index, entropy_norm AS value FROM map_metrics WHERE case_id=? ORDER BY entropy_norm DESC LIMIT 1"),
        "most_concentrated": row("SELECT word_index, layer_index, top_5_mass AS value FROM map_metrics WHERE case_id=? ORDER BY top_5_mass DESC LIMIT 1"),
        "strongest_multipeak": row("SELECT word_index, layer_index, secondary_primary_ratio AS value FROM map_metrics WHERE case_id=? ORDER BY secondary_primary_ratio DESC LIMIT 1"),
        "largest_peak_count": row("SELECT word_index, layer_index, peak_count AS value FROM map_metrics WHERE case_id=? ORDER BY peak_count DESC LIMIT 1"),
        "largest_layer_jump": row("SELECT word_index, layer_max_jump AS value FROM layer_scanpaths WHERE case_id=? ORDER BY layer_max_jump DESC LIMIT 1"),
        "largest_word_jump": row("SELECT layer_index, word_max_jump AS value FROM word_scanpaths WHERE case_id=? ORDER BY word_max_jump DESC LIMIT 1"),
    }


def get_global_rankings(connection) -> dict:
    return {
        "diffuse_cases": connection.execute(
            """
            SELECT c.case_id, c.image_label, c.condition_label, AVG(mm.entropy_norm) AS score
            FROM map_metrics mm JOIN cases c ON c.case_id=mm.case_id
            GROUP BY c.case_id ORDER BY score DESC LIMIT 8
            """
        ).fetchall(),
        "multipeak_cases": connection.execute(
            """
            SELECT c.case_id, c.image_label, c.condition_label, AVG(mm.secondary_primary_ratio) AS score
            FROM map_metrics mm JOIN cases c ON c.case_id=mm.case_id
            GROUP BY c.case_id ORDER BY score DESC LIMIT 8
            """
        ).fetchall(),
        "unstable_layer_scanpaths": connection.execute(
            """
            SELECT l.case_id, c.image_label, c.condition_label, AVG(l.layer_max_jump) AS score
            FROM layer_scanpaths l JOIN cases c ON c.case_id=l.case_id
            GROUP BY l.case_id ORDER BY score DESC LIMIT 8
            """
        ).fetchall(),
        "largest_word_jumps": connection.execute(
            """
            SELECT w.case_id, c.image_label, c.condition_label, AVG(w.word_max_jump) AS score
            FROM word_scanpaths w JOIN cases c ON c.case_id=w.case_id
            GROUP BY w.case_id ORDER BY score DESC LIMIT 8
            """
        ).fetchall(),
    }


def build_grouped_metrics(row, groups: dict[str, list[str]]) -> dict[str, list[dict]]:
    if not row:
        return {}
    out = {}
    for group_name, keys in groups.items():
        out[group_name] = [{"key": key, "value": row[key] if key in row.keys() else None} for key in keys]
    return out


def group_pair_metrics(pair: dict) -> dict[str, list[dict]]:
    if not pair:
        return {}
    groups = {
        "Map similarity": ["cosine_similarity", "pearson_correlation", "ssim", "jsd", "emd_2d"],
        "Pixel/value distances": ["l1_distance", "l2_distance"],
        "Hotspot overlap": ["top_1_iou", "top_5_iou", "top_10_iou", "hotspot_iou_percentile_90", "hotspot_iou_percentile_95"],
        "Spatial shift": ["global_centroid_shift", "primary_centroid_shift", "spread_delta", "anisotropy_delta", "radial_profile_distance"],
        "Debug/secondary": ["argmax_distance", "hausdorff_peak_distance"],
    }
    return {name: [{"key": key, "value": pair.get(key)} for key in keys] for name, keys in groups.items()}


def aggregate_condition_metrics(connection, case_a: str, case_b: str) -> dict:
    pair_rows = connection.execute(
        """
        SELECT
          a.word_index,
          a.layer_index,
          a.global_centroid_x_norm AS ax,
          a.global_centroid_y_norm AS ay,
          b.global_centroid_x_norm AS bx,
          b.global_centroid_y_norm AS by,
          a.entropy_norm AS entropy_a,
          b.entropy_norm AS entropy_b,
          a.spread_trace AS spread_a,
          b.spread_trace AS spread_b,
          a.peak_count AS peaks_a,
          b.peak_count AS peaks_b,
          a.secondary_primary_ratio AS secondary_a,
          b.secondary_primary_ratio AS secondary_b
        FROM map_metrics a
        JOIN map_metrics b
          ON a.word_index=b.word_index AND a.layer_index=b.layer_index
        WHERE a.case_id=? AND b.case_id=?
        """,
        (case_a, case_b),
    ).fetchall()
    centroid_shifts = []
    entropy_deltas = []
    spread_deltas = []
    peak_deltas = []
    secondary_deltas = []
    for row in pair_rows:
        if row["ax"] is not None and row["ay"] is not None and row["bx"] is not None and row["by"] is not None:
            centroid_shifts.append(((float(row["ax"]) - float(row["bx"])) ** 2 + (float(row["ay"]) - float(row["by"])) ** 2) ** 0.5)
        append_abs_delta(entropy_deltas, row["entropy_a"], row["entropy_b"])
        append_abs_delta(spread_deltas, row["spread_a"], row["spread_b"])
        append_abs_delta(peak_deltas, row["peaks_a"], row["peaks_b"])
        append_abs_delta(secondary_deltas, row["secondary_a"], row["secondary_b"])

    layer_rows = connection.execute(
        """
        SELECT
          a.layer_path_length AS a_len,
          b.layer_path_length AS b_len,
          a.layer_max_jump AS a_jump,
          b.layer_max_jump AS b_jump,
          a.multipeak_layer_ratio AS a_multi,
          b.multipeak_layer_ratio AS b_multi
        FROM layer_scanpaths a
        JOIN layer_scanpaths b ON a.word_index=b.word_index
        WHERE a.case_id=? AND b.case_id=?
        """,
        (case_a, case_b),
    ).fetchall()
    layer_path_deltas = []
    layer_jump_deltas = []
    multipeak_deltas = []
    for row in layer_rows:
        append_abs_delta(layer_path_deltas, row["a_len"], row["b_len"])
        append_abs_delta(layer_jump_deltas, row["a_jump"], row["b_jump"])
        append_abs_delta(multipeak_deltas, row["a_multi"], row["b_multi"])

    return {
        "matched_map_count": len(pair_rows),
        "matched_layer_scanpath_count": len(layer_rows),
        "centroid_shift_norm": summarize(centroid_shifts),
        "entropy_abs_delta": summarize(entropy_deltas),
        "spread_abs_delta": summarize(spread_deltas),
        "peak_count_abs_delta": summarize(peak_deltas),
        "secondary_ratio_abs_delta": summarize(secondary_deltas),
        "layer_path_length_abs_delta": summarize(layer_path_deltas),
        "layer_max_jump_abs_delta": summarize(layer_jump_deltas),
        "multipeak_ratio_abs_delta": summarize(multipeak_deltas),
    }


def load_case_metadata(config: DashboardConfig, case) -> dict:
    metadata_path = resolve_project_path(case["metadata_path"], config.project_root)
    if not metadata_path.exists():
        return {"prompt_text": "", "response_text": ""}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"prompt_text": "", "response_text": ""}
    return {
        "prompt_text": str(payload.get("prompt_text") or ""),
        "response_text": str(payload.get("response_text") or ""),
        "system_prompt": str(payload.get("system_prompt") or ""),
    }


def condition_purpose(label: str) -> str:
    purposes = {
        "baseline_neutral": "Neutral reference prompt used as the baseline condition.",
        "image_grounded_visible_only": "Asks the model to focus only on directly visible image evidence.",
        "ambiguous_open": "Open-ended image understanding condition.",
        "misleading_wrong_subject": "Stress condition with a misleading subject in the prompt.",
        "extra_knowledge_context": "Condition that may encourage contextual or functional knowledge beyond simple description.",
        "reasoning_controlled_brief": "Asks for a brief relevance explanation while keeping the response controlled.",
        "order_disruption_stress": (
            "Grounding-format stress condition. This prompt can trigger Qwen2-VL object-reference / "
            "bounding-box style output; it is kept intentionally and should not be compared to purely "
            "descriptive conditions without this caveat."
        ),
        "colleague_obj_detection_hard": "Object-detection style prompt from the colleague line of work.",
    }
    return purposes.get(label, "not documented yet")


def condition_family(label: str) -> str:
    if label == "baseline_neutral":
        return "baseline"
    if "colleague" in label:
        return "colleague_reference"
    if label == "order_disruption_stress":
        return "grounding-format stress"
    if "misleading" in label or "stress" in label or "disruption" in label:
        return "stress"
    if "grounded" in label:
        return "image_grounded"
    if "knowledge" in label:
        return "context"
    if "reasoning" in label:
        return "reasoning"
    if "ambiguous" in label:
        return "open"
    return "not documented yet"


def append_abs_delta(values: list[float], a, b) -> None:
    if a is None or b is None:
        return
    values.append(abs(float(a) - float(b)))


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "iqr": None, "min": None, "max": None}
    ordered = sorted(float(value) for value in values)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    q1 = ordered[max(0, int(0.25 * (n - 1)))]
    q3 = ordered[max(0, int(0.75 * (n - 1)))]
    return {
        "mean": sum(ordered) / n,
        "median": median,
        "iqr": q3 - q1,
        "min": ordered[0],
        "max": ordered[-1],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local TAM dashboard.")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
