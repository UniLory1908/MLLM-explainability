from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from textwrap import shorten

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_DIR = PROJECT_ROOT / "outputs" / "analysis" / "v1_case_level"
V2_INTERPRETATION_DIR = PROJECT_ROOT / "outputs" / "analysis" / "v2_interpretation"
V2_CLUSTERING_DIR = PROJECT_ROOT / "outputs" / "analysis" / "v2_clustering"
OUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "v3_case_studies"

FINAL_CASE_TARGETS = [
    (504074, "colleague_obj_detection_hard", "CS01", "prompt_dominated / high drift object-detection response"),
    (156924, "order_disruption_stress", "CS02", "grounding-format stress with bbox-style output and outlier flag"),
    (1761, "colleague_obj_detection_hard", "CS03", "high unstable/prompt-dominated object-detection candidate"),
    (378099, "ambiguous_open", "CS04", "ambiguous weak-grounding / multipeak candidate in Q4"),
    (402615, "order_disruption_stress", "CS05", "grounding-format stress with weak-grounding proxy and outlier flag"),
    (30213, "ambiguous_open", "CS06", "Q4 outlier candidate with text-vision decoupling"),
    (364557, "colleague_obj_detection_hard", "CS07", "object-detection outlier in Q2"),
    (555412, "misleading_wrong_subject", "CS08", "misleading prompt with high joint drift"),
    (69213, "colleague_obj_detection_hard", "CS09", "object-detection high-drift outlier"),
    (262682, "order_disruption_stress", "CS10", "grounding-format stress candidate with high unstable score"),
    (504074, "ambiguous_open", "CS11", "Q3 text changes with comparatively stable visual summary"),
    (262682, "extra_knowledge_context", "CS12", "Q1/extra-knowledge stable contrast case"),
]

SCORE_BY_PATTERN = [
    ("prompt_dominated", "prompt_dominated_candidate_score"),
    ("weak-grounding", "weak_grounding_candidate_score"),
    ("weak_grounding", "weak_grounding_candidate_score"),
    ("multipeak", "multipeak_ambiguity_score"),
    ("unstable", "unstable_explanation_candidate_score"),
    ("bbox", "bbox_or_grounding_format_score"),
    ("grounding-format", "bbox_or_grounding_format_score"),
    ("centroid", "mean_centroid_shift_vs_baseline"),
    ("Q3", "textual_change"),
    ("Q4", "visual_change"),
    ("Q1", "visual_change"),
]

FIGURES = [
    (
        "outputs/analysis/v1_case_level/figures/boxplot_scores_by_prompt_category.png",
        "Distribuzione degli score diagnostici per prompt category.",
        "Sezione risultati per prompt/category.",
        "Le categorie di stress/misleading/object-detection tendono ad avere proxy piu elevati.",
        "Score diagnostici proxy, non causal faithfulness.",
    ),
    (
        "outputs/analysis/v1_case_level/figures/scatter_text_vs_visual_change.png",
        "Scatter della matrice text-vs-vision.",
        "Sezione decoupling testo-visione.",
        "Mostra Q2 e i casi Q3/Q4 in cui testo e visual sensitivity si separano.",
        "Assi normalizzati in modo esplorativo; non sono scale fisiche.",
    ),
    (
        "outputs/analysis/v1_case_level/figures/correlation_matrix_spearman.png",
        "Matrice di correlazione Spearman tra metriche.",
        "Sezione metric redundancy / feature selection.",
        "Evidenzia metriche ridondanti e famiglie di proxy correlate.",
        "Correlazione non implica causalita o equivalenza semantica.",
    ),
    (
        "outputs/analysis/v2_clustering/figures/pca_clusters.png",
        "PCA colorata per cluster KMeans.",
        "Sezione clustering leggero.",
        "Cluster 2 concentra molti casi grounding-format/object-detection ad alto drift.",
        "PCA/KMeans sono descrittivi e dipendono dalle feature aggregate.",
    ),
    (
        "outputs/analysis/v2_clustering/figures/pca_prompt_category.png",
        "PCA colorata per prompt category.",
        "Sezione pattern per prompt.",
        "Mostra la relazione tra categorie prompt e regioni dello spazio feature.",
        "Non dimostra separazione causale tra categorie.",
    ),
    (
        "outputs/analysis/v2_clustering/figures/pca_quadrants.png",
        "PCA colorata per quadrante text-vs-vision.",
        "Sezione decoupling e cluster.",
        "Q3/Q4 sono dispersi ma visibili come casi di decoupling.",
        "Quadranti derivati da soglie esplorative.",
    ),
    (
        "outputs/analysis/v2_clustering/figures/outlier_score_distribution.png",
        "Distribuzione degli outlier score.",
        "Sezione anomaly detection leggera.",
        "Indica la coda dei casi da ispezionare per pattern estremi.",
        "Isolation-style score locale, non classificazione supervisionata.",
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build final report-ready TAM case studies from v1/v2 outputs.")
    parser.add_argument("--v1-dir", type=Path, default=V1_DIR)
    parser.add_argument("--v2-interpretation-dir", type=Path, default=V2_INTERPRETATION_DIR)
    parser.add_argument("--v2-clustering-dir", type=Path, default=V2_CLUSTERING_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser


def read_required_inputs(v1_dir: Path, v2i: Path, v2c: Path) -> dict[str, object]:
    inputs = {
        "case_level": pd.read_parquet(v1_dir / "analysis_case_level_v1.parquet"),
        "prompt_report": (v1_dir / "report_prompt_effects_v1.md").read_text(encoding="utf-8"),
        "quadrants": pd.read_parquet(v1_dir / "text_visual_quadrants.parquet"),
        "quadrant_report": (v1_dir / "text_visual_quadrants_summary.md").read_text(encoding="utf-8"),
        "quadrant_reps": pd.read_csv(v1_dir / "representative_cases_by_quadrant.csv"),
        "interpretation_report": (v2i / "interpretation_report_v2.md").read_text(encoding="utf-8"),
        "selected_v2": pd.read_csv(v2i / "selected_case_studies_v1.csv"),
        "case_study_report_v2": (v2i / "case_study_selection_report.md").read_text(encoding="utf-8"),
        "crosscheck": (v2i / "ranking_quadrant_cluster_crosscheck.md").read_text(encoding="utf-8"),
        "clusters": pd.read_parquet(v2c / "clusters_v1.parquet"),
        "cluster_summary": (v2c / "cluster_summary.md").read_text(encoding="utf-8"),
        "cluster_reps": pd.read_csv(v2c / "cluster_representative_cases.csv"),
        "outliers": pd.read_csv(v2c / "outliers_v1.csv"),
        "summary_label": pd.read_csv(v1_dir / "summary_by_prompt_label.csv"),
        "summary_category": pd.read_csv(v1_dir / "summary_by_prompt_category.csv"),
    }
    rank_dir = v1_dir / "rankings_v1"
    inputs["rankings"] = {p.name: pd.read_csv(p) for p in rank_dir.glob("top_*.csv")}
    return inputs


def main_score_for(reason: str, row: pd.Series) -> tuple[str, float]:
    for needle, col in SCORE_BY_PATTERN:
        if needle.lower() in reason.lower() and col in row:
            return col, float(row[col])
    if row.get("quadrant") == "Q2_text_changed_visual_changed":
        return "text_visual_joint_change_proxy", float(row["textual_change"] + row["visual_change"])
    return "unstable_explanation_candidate_score", float(row["unstable_explanation_candidate_score"])


def pattern_for(row: pd.Series, reason: str) -> str:
    parts = []
    if row["prompt_category"] in {"grounding_format_stress", "object_detection", "misleading"}:
        parts.append(row["prompt_category"])
    if row["quadrant"].startswith("Q2"):
        parts.append("joint text-vision drift")
    elif row["quadrant"].startswith("Q3"):
        parts.append("text changes / visual stable")
    elif row["quadrant"].startswith("Q4"):
        parts.append("text stable / visual changes")
    elif row["quadrant"].startswith("Q1"):
        parts.append("stable contrast")
    if bool(row.get("is_outlier", False)):
        parts.append("outlier")
    if not parts:
        parts.append(reason)
    return "; ".join(parts)


def recommended_action(row: pd.Series) -> str:
    if row["quadrant"].startswith("Q3"):
        return "Open compare view, then inspect matrix to verify text drift with relatively stable TAM summaries."
    if row["quadrant"].startswith("Q4"):
        return "Open matrix and compare view to inspect visual sensitivity drift despite comparatively stable text."
    if row["prompt_category"] == "grounding_format_stress":
        return "Open case and compare view; inspect bbox/object-reference output against baseline and matrix."
    if row["prompt_category"] == "object_detection":
        return "Open case and matrix; inspect object-reference response and whether attribution concentrates or shifts."
    if row["prompt_category"] == "misleading":
        return "Open compare view; inspect response divergence and whether visual drift follows the misleading prompt."
    return "Open compare view first, then inspect matrix for baseline-vs-prompt attribution drift."


def select_final_cases(inputs: dict[str, object]) -> pd.DataFrame:
    case_level: pd.DataFrame = inputs["case_level"]
    clusters: pd.DataFrame = inputs["clusters"]
    cluster_cols = clusters[["case_id", "cluster_id", "is_outlier", "outlier_score"]].drop_duplicates("case_id")
    full = case_level.merge(cluster_cols, on="case_id", how="left")
    baselines = (
        case_level.loc[case_level["prompt_label"].eq("baseline_neutral"), ["image_id", "response_text"]]
        .rename(columns={"response_text": "baseline_response_text"})
        .drop_duplicates("image_id")
    )
    full = full.merge(baselines, on="image_id", how="left")

    rows = []
    used = set()
    for image_id, prompt_label, cs_id, reason in FINAL_CASE_TARGETS:
        matches = full.loc[(full["image_id"].eq(image_id)) & (full["prompt_label"].eq(prompt_label))]
        if matches.empty:
            continue
        row = matches.iloc[0].copy()
        if row["case_id"] in used:
            continue
        used.add(row["case_id"])
        score_name, score_value = main_score_for(reason, row)
        row["case_study_id"] = cs_id
        row["selection_reason"] = reason
        row["main_pattern"] = pattern_for(row, reason)
        row["main_score_name"] = score_name
        row["main_score_value"] = score_value
        row["recommended_dashboard_action"] = recommended_action(row)
        row["notes"] = (
            "Candidate/proxy case for qualitative dashboard inspection; not causal evidence. "
            "Compare against baseline_neutral for the same image."
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    keep = [
        "case_study_id",
        "image_id",
        "case_id",
        "prompt_label",
        "prompt_category",
        "quadrant",
        "cluster_id",
        "is_outlier",
        "selection_reason",
        "main_pattern",
        "main_score_name",
        "main_score_value",
        "textual_change",
        "visual_change",
        "unstable_explanation_candidate_score",
        "prompt_dominated_candidate_score",
        "weak_grounding_candidate_score",
        "multipeak_ambiguity_score",
        "bbox_or_grounding_format_score",
        "response_text",
        "baseline_response_text",
        "dashboard_case_url",
        "dashboard_matrix_url",
        "dashboard_compare_url",
        "recommended_dashboard_action",
        "notes",
    ]
    return out[keep].sort_values("case_study_id")


def md_table(df: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    use = df[cols].head(max_rows) if max_rows else df[cols]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in use.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                text = str(val).replace("\n", " ")
                vals.append(shorten(text, width=90, placeholder="..."))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def quote_block(text: object, limit: int = 700) -> str:
    clean = str(text) if pd.notna(text) else ""
    clean = clean.replace("\r", " ").replace("\n", " ").strip()
    if len(clean) > limit:
        clean = clean[: limit - 3].rstrip() + "..."
    return "> " + clean


def write_cards(final: pd.DataFrame, out_dir: Path) -> None:
    lines = ["# Final Case Study Cards", ""]
    for idx, (_, row) in enumerate(final.iterrows(), start=1):
        lines.extend(
            [
                f"## Case Study {idx} - {row['image_id']} / {row['prompt_label']}",
                "",
                "Pattern principale:",
                f"- {row['main_pattern']}",
                "",
                "Perche e stato selezionato:",
                f"- {row['selection_reason']}",
                "",
                "Categoria prompt:",
                f"- {row['prompt_category']}",
                "",
                "Quadrante testo-vs-visione:",
                f"- {row['quadrant']}",
                "",
                "Cluster:",
                f"- {int(row['cluster_id']) if pd.notna(row['cluster_id']) else 'n/a'}",
                "",
                "Outlier:",
                f"- {'si' if bool(row['is_outlier']) else 'no'}",
                "",
                "Baseline response:",
                quote_block(row["baseline_response_text"]),
                "",
                "Perturbed response:",
                quote_block(row["response_text"]),
                "",
                "Metriche principali:",
                f"- textual_change: {row['textual_change']:.4f}",
                f"- visual_change: {row['visual_change']:.4f}",
                f"- unstable_explanation_candidate_score: {row['unstable_explanation_candidate_score']:.4f}",
                f"- prompt_dominated_candidate_score: {row['prompt_dominated_candidate_score']:.4f}",
                f"- weak_grounding_candidate_score: {row['weak_grounding_candidate_score']:.4f}",
                f"- multipeak_ambiguity_score: {row['multipeak_ambiguity_score']:.4f}",
                f"- bbox_or_grounding_format_score: {row['bbox_or_grounding_format_score']:.4f}",
                "",
                "Dashboard:",
                f"- case: {row['dashboard_case_url']}",
                f"- matrix: {row['dashboard_matrix_url']}",
                f"- compare: {row['dashboard_compare_url']}",
                "",
                "Cosa controllare visualmente:",
                "- confrontare baseline vs prompt perturbato",
                "- verificare se la heatmap si sposta o resta simile",
                "- verificare se l'output cambia coerentemente con il drift visuale",
                "- annotare eventuale decoupling testo-visione",
                "",
                "Interpretazione prudente:",
                f"- questo caso e un candidato a {row['main_pattern']} secondo diagnostic score/proxy aggregati",
                "- non e una prova causale di grounding, hallucination o attenzione certa a una regione",
                "",
            ]
        )
    (out_dir / "case_study_cards.md").write_text("\n".join(lines), encoding="utf-8")


def write_report_tables(inputs: dict[str, object], final: pd.DataFrame, out_dir: Path) -> None:
    label = inputs["summary_label"].copy()
    label_small = label[
        [
            "prompt_label",
            "unstable_explanation_candidate_score_mean",
            "prompt_dominated_candidate_score_mean",
            "weak_grounding_candidate_score_mean",
            "textual_change_mean",
            "visual_change_mean",
        ]
    ].sort_values("unstable_explanation_candidate_score_mean", ascending=False)
    category = inputs["summary_category"].copy()
    category_small = category[
        [
            "prompt_category",
            "unstable_explanation_candidate_score_mean",
            "prompt_dominated_candidate_score_mean",
            "textual_change_mean",
            "visual_change_mean",
        ]
    ].sort_values("visual_change_mean", ascending=False)
    quadrants = inputs["quadrants"].groupby("quadrant").agg(
        cases=("case_id", "count"),
        textual_change_mean=("textual_change", "mean"),
        visual_change_mean=("visual_change", "mean"),
    ).reset_index()
    clusters = inputs["clusters"].groupby("cluster_id").agg(
        cases=("case_id", "count"),
        textual_change_mean=("textual_change", "mean"),
        visual_change_mean=("visual_change", "mean"),
        outlier_rate=("is_outlier", "mean"),
    ).reset_index()
    limits = pd.DataFrame(
        [
            {"limit": "Diagnostic scores", "interpretation": "Proxy/candidate rankings, not causal evidence."},
            {"limit": "Baseline alignment", "interpretation": "Baseline comparisons use baseline_neutral and positional/common-index alignment."},
            {"limit": "No GT/causal metrics", "interpretation": "No COCO grounding, perturbative causal faithfulness, or GT mask/box metric."},
            {"limit": "Raw-map similarities", "interpretation": "No bulk EMD or raw all-vs-all pairwise computation in this batch."},
            {"limit": "Dashboard inspection", "interpretation": "Final claims require qualitative visual inspection of selected cases."},
        ]
    )
    lines = [
        "# Report-ready Tables",
        "",
        "## 1. Prompt labels principali",
        md_table(label_small, list(label_small.columns)),
        "",
        "## 1b. Prompt categories principali",
        md_table(category_small, list(category_small.columns)),
        "",
        "## 2. Quadranti testo-vs-visione",
        md_table(quadrants, list(quadrants.columns)),
        "",
        "## 3. Cluster principali",
        md_table(clusters, list(clusters.columns)),
        "",
        "## 4. Casi studio finali",
        md_table(final, ["case_study_id", "image_id", "prompt_label", "prompt_category", "quadrant", "cluster_id", "is_outlier", "main_pattern"]),
        "",
        "## 5. Limiti metodologici",
        md_table(limits, ["limit", "interpretation"]),
        "",
    ]
    (out_dir / "report_ready_tables.md").write_text("\n".join(lines), encoding="utf-8")


def write_figures_index(out_dir: Path) -> None:
    rows = []
    for path, desc, where, message, limits in FIGURES:
        p = PROJECT_ROOT / path
        rows.append(
            {
                "path_figura": path if p.exists() else f"{path} (not found)",
                "descrizione": desc,
                "dove_usarla": where,
                "messaggio_principale": message,
                "limiti_interpretativi": limits,
            }
        )
    df = pd.DataFrame(rows)
    lines = ["# Report-ready Figures Index", "", md_table(df, list(df.columns))]
    (out_dir / "report_ready_figures_index.md").write_text("\n".join(lines), encoding="utf-8")


def write_main_findings(final: pd.DataFrame, out_dir: Path) -> None:
    first = final.sort_values(["is_outlier", "main_score_value"], ascending=[False, False]).head(6)
    lines = [
        "# Main findings preliminari",
        "",
        "## Dataset",
        "- 100 immagini",
        "- 800 casi",
        "- 8 prompt per immagine",
        "- analisi case-level su prompt perturbation",
        "",
        "## Risultati principali",
        "1. Le condition `grounding_format_stress`, `misleading` e `object_detection` mostrano i pattern piu forti di drift rispetto al baseline nei diagnostic score/proxy.",
        "2. Il cluster 2 concentra molti casi Q2 con drift congiunto testo-visione, soprattutto grounding-format e object-detection.",
        "3. I casi `ambiguous` emergono soprattutto come candidate weak-grounding/multipeak, piu che come cluster separato.",
        "4. Q3 e Q4 sono utili per discutere decoupling testo-visione: testo e visual sensitivity non cambiano sempre insieme.",
        "5. I casi finali includono esempi high-drift, bbox-format, misleading, ambiguous, Q3, Q4, Q1 stabile e outlier.",
        "",
        "## Interpretazione",
        "- Le metriche sono diagnostic score e proxy esplorativi. Indicano visual sensitivity e drift rispetto al baseline, non grounding causale.",
        "- I casi selezionati sono candidati da ispezionare qualitativamente in dashboard prima di inserirli come esempi nel report/tesi.",
        "",
        "## Casi studio consigliati",
        md_table(first, ["case_study_id", "image_id", "prompt_label", "quadrant", "cluster_id", "is_outlier", "main_pattern"]),
        "",
        "## Limiti",
        "- Nessuna nuova inferenza Qwen/TAM.",
        "- Nessuna metrica GT/COCO grounding.",
        "- Nessuna causal faithfulness perturbativa.",
        "- Nessun bulk EMD o raw-map all-vs-all.",
        "- Le dashboard heatmap vanno interpretate come TAM-derived visual sensitivity, non come prova che il modello guarda certamente una regione.",
        "",
        "## Prossimi passi",
        "- Aprire i casi selezionati in dashboard e annotare baseline-vs-prompt.",
        "- Scegliere 4-6 casi finali per figure nel report, mantenendo almeno un Q3/Q4 e un Q1 di contrasto.",
        "- Scrivere una sezione metodi che distingua stabilita, plausibility/alignment proxy e causal faithfulness non misurata.",
    ]
    (out_dir / "main_findings_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(final: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> None:
    files = sorted(str(p.relative_to(out_dir)).replace("\\", "/") for p in out_dir.rglob("*") if p.is_file())
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir),
        "input_paths": {
            "v1_dir": str(args.v1_dir.resolve()),
            "v2_interpretation_dir": str(args.v2_interpretation_dir.resolve()),
            "v2_clustering_dir": str(args.v2_clustering_dir.resolve()),
        },
        "case_study_count": int(len(final)),
        "case_studies": final[["case_study_id", "image_id", "prompt_label", "quadrant", "cluster_id", "main_pattern"]].to_dict(orient="records"),
        "methodological_note": "Report-ready case-study selection from existing v1/v2 tabular outputs only; no raw-map processing, new inference, DB/cache changes, or notebook edits.",
        "created_files": files,
    }
    (out_dir / "batch3_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = read_required_inputs(args.v1_dir.resolve(), args.v2_interpretation_dir.resolve(), args.v2_clustering_dir.resolve())
    final = select_final_cases(inputs)
    final.to_csv(out_dir / "final_case_studies.csv", index=False)
    final.to_parquet(out_dir / "final_case_studies.parquet", index=False)
    write_cards(final, out_dir)
    write_report_tables(inputs, final, out_dir)
    write_figures_index(out_dir)
    write_main_findings(final, out_dir)
    write_manifest(final, out_dir, args)
    print(f"Wrote {len(final)} final case studies to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
