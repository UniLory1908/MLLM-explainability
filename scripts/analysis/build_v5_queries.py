from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUERY_ROOT = PROJECT_ROOT / "outputs" / "analysis" / "v5_queries"
CASE_PATH = PROJECT_ROOT / "outputs" / "analysis" / "v1_case_level" / "analysis_case_level_v1.parquet"
CLUSTER_PATH = PROJECT_ROOT / "outputs" / "analysis" / "v2_clustering" / "clusters_v1.parquet"
CASE_STUDIES_PATH = PROJECT_ROOT / "outputs" / "analysis" / "v3_case_studies" / "final_case_studies.parquet"

Q1 = "Q1_text_stable_visual_stable"
Q2 = "Q2_text_changed_visual_changed"
Q3 = "Q3_text_changed_visual_stable"
Q4 = "Q4_text_stable_visual_changed"

SCORE_COLS = [
    "textual_change",
    "visual_change",
    "prompt_dominated_candidate_score",
    "unstable_explanation_candidate_score",
    "weak_grounding_candidate_score",
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def safe_mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(0.0, index=df.index)
    return df[existing].fillna(0).mean(axis=1)


def coord_mask(df: pd.DataFrame) -> pd.Series:
    coord_regex = r"\(\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\)|\[\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?"
    return (
        df.get("coordinate_token_count", 0).fillna(0).astype(float).gt(0)
        | df.get("bbox_style_output_flag", 0).fillna(0).astype(float).gt(0)
        | df.get("has_box_tokens", 0).fillna(0).astype(float).gt(0)
        | df.get("response_text", "").fillna("").str.contains(coord_regex, regex=True)
    )


def add_baseline_response(df: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        df.loc[df["prompt_label"].eq("baseline_neutral"), ["image_id", "response_text"]]
        .rename(columns={"response_text": "baseline_response_text"})
        .drop_duplicates("image_id")
    )
    return df.merge(baseline, on="image_id", how="left")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case = pd.read_parquet(CASE_PATH)
    clusters = pd.read_parquet(CLUSTER_PATH)
    studies = pd.read_parquet(CASE_STUDIES_PATH)
    cluster_extra = clusters[["case_id", "cluster_id", "outlier_score", "is_outlier"]].copy()
    merged = case.merge(cluster_extra, on="case_id", how="left", suffixes=("", "_cluster"))
    merged = add_baseline_response(merged)
    return case, clusters, studies, merged


def qcount(frame: pd.DataFrame, name: str) -> int:
    return int(frame["quadrant"].eq(name).sum()) if "quadrant" in frame else 0


def image_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for image_id, g in df.groupby("image_id", dropna=False):
        row = {
            "image_id": image_id,
            "n_cases": len(g),
            "n_prompts": g["prompt_label"].nunique(),
            "mean_textual_change": g["textual_change"].mean(),
            "mean_visual_change": g["visual_change"].mean(),
            "mean_prompt_dominated_candidate_score": g["prompt_dominated_candidate_score"].mean(),
            "mean_unstable_explanation_candidate_score": g["unstable_explanation_candidate_score"].mean(),
            "mean_weak_grounding_candidate_score": g["weak_grounding_candidate_score"].mean(),
            "count_Q1": qcount(g, Q1),
            "count_Q2": qcount(g, Q2),
            "count_Q3": qcount(g, Q3),
            "count_Q4": qcount(g, Q4),
            "count_cluster_2": int(g.get("cluster_id", pd.Series(dtype=float)).eq(2).sum()),
            "count_outlier": int(g.get("is_outlier", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
            "dashboard_compare_url": g["dashboard_compare_url"].dropna().iloc[0] if g["dashboard_compare_url"].notna().any() else "",
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    out["image_sensitivity_score"] = safe_mean(
        out,
        [
            "mean_textual_change",
            "mean_visual_change",
            "mean_prompt_dominated_candidate_score",
            "mean_unstable_explanation_candidate_score",
        ],
    )
    out["decoupled_count"] = out["count_Q3"] + out["count_Q4"]
    return out


def prompt_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(["prompt_label", "prompt_category"], dropna=False):
        label, category = keys
        n = len(g)
        row = {
            "prompt_label": label,
            "prompt_category": category,
            "n_cases": n,
            "count_Q1": qcount(g, Q1),
            "count_Q2": qcount(g, Q2),
            "count_Q3": qcount(g, Q3),
            "count_Q4": qcount(g, Q4),
            "mean_textual_change": g["textual_change"].mean(),
            "mean_visual_change": g["visual_change"].mean(),
            "mean_prompt_dominated_candidate_score": g["prompt_dominated_candidate_score"].mean(),
            "mean_unstable_explanation_candidate_score": g["unstable_explanation_candidate_score"].mean(),
            "mean_weak_grounding_candidate_score": g["weak_grounding_candidate_score"].mean(),
            "mean_multipeak_ambiguity_score": g["multipeak_ambiguity_score"].mean(),
            "mean_mean_centroid_shift_vs_baseline": g["mean_centroid_shift_vs_baseline"].mean(),
        }
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            row[f"rate_{q}"] = row[f"count_{q}"] / n if n else 0
        row["rate_decoupling_Q3_Q4"] = (row["count_Q3"] + row["count_Q4"]) / n if n else 0
        rows.append(row)
    return pd.DataFrame(rows)


def select_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]].copy()


def md_table(df: pd.DataFrame, max_rows: int = 15) -> list[str]:
    if df.empty:
        return ["Nessuna riga trovata."]
    shown = df.head(max_rows).copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else str(x).replace("\n", " ")[:160])
    lines = ["| " + " | ".join(shown.columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(shown.columns)) + " |")
    for _, row in shown.iterrows():
        vals = [str(row[c]).replace("|", "\\|") for c in shown.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_bundle(
    name: str,
    question: str,
    result: pd.DataFrame,
    *,
    sources: list[Path],
    criteria: list[str],
    main_finding: str,
    limits: list[str],
    aggregation: str,
    top_cases: pd.DataFrame | None = None,
    missing_columns: list[str] | None = None,
    extra_files: list[str] | None = None,
) -> dict[str, str]:
    out_dir = QUERY_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "results.csv", index=False)
    result.to_parquet(out_dir / "results.parquet", index=False)
    (out_dir / "question.md").write_text(f"# Question\n\n{question}\n", encoding="utf-8")

    lines = [
        f"# Query: {name}",
        "",
        "## 1. Domanda Interpretata",
        "",
        question,
        "",
        "## 2. File Sorgente Usati",
        "",
    ]
    lines.extend([f"- `{rel(path)}`" for path in sources])
    lines.extend(["", "## 3. Criteri/Filtro Usato", ""])
    lines.extend([f"- {item}" for item in criteria])
    lines.extend(["", "## 4. Conteggio Totale", ""])
    lines.append(f"- Righe nel risultato: `{len(result)}`")
    if "n_cases" in result.columns:
        lines.append(f"- Casi rappresentati nel risultato: `{int(result['n_cases'].sum())}`")
    if missing_columns:
        lines.append(f"- Query parziale: colonne mancanti `{', '.join(missing_columns)}`")
    lines.extend(["", "## 5. Breakdown / Ranking Principale", ""])
    lines.extend(md_table(result, 20))
    lines.extend(["", "## 6. Top Casi Con Dashboard Links", ""])
    if top_cases is None:
        lines.append("Non applicabile: questa query produce principalmente un aggregato.")
    else:
        lines.extend(md_table(top_cases, 20))
    lines.extend(["", "## 7. File Creati", ""])
    files = ["question.md", "answer.md", "results.csv", "results.parquet", "manifest.json"]
    files.extend(extra_files or [])
    lines.extend([f"- `outputs/analysis/v5_queries/{name}/{file}`" for file in files])
    lines.extend(["", "## 8. Limiti Della Query", ""])
    lines.extend([f"- {item}" for item in limits])
    lines.extend(["", "## 9. Conferma Aree Protette", ""])
    lines.append("Non sono stati modificati raw `.npy`, `outputs/prompt_sensitivity/`, DB/cache dashboard, archive canonico o notebook.")
    lines.extend(["", "## Main Finding Breve", "", main_finding])
    (out_dir / "answer.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query": name,
        "question": question,
        "sources": [str(path.resolve()) for path in sources],
        "output_dir": str(out_dir.resolve()),
        "aggregation": aggregation,
        "rows": int(len(result)),
        "criteria": criteria,
        "main_finding": main_finding,
        "missing_columns": missing_columns or [],
        "notes": [
            "Exploratory analysis over exported tables.",
            "Diagnostic/proxy rankings only; not causal evidence.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "query_name": name,
        "question": question,
        "main_aggregation": aggregation,
        "main_output_file": f"outputs/analysis/v5_queries/{name}/results.csv",
        "main_finding_breve": main_finding,
        "status": "done_partial" if missing_columns else "done",
    }


def top_case_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "image_id",
        "case_id",
        "prompt_label",
        "prompt_category",
        "quadrant",
        "textual_change",
        "visual_change",
        "prompt_dominated_candidate_score",
        "unstable_explanation_candidate_score",
        "weak_grounding_candidate_score",
        "mean_centroid_shift_vs_baseline",
        "response_text",
        "baseline_response_text",
        "dashboard_case_url",
        "dashboard_compare_url",
    ]
    return select_cols(df, cols)


def build_queries() -> list[dict[str, str]]:
    case, clusters, _studies, merged = load_data()
    nonbase = merged.loc[~merged["is_baseline"].astype(bool)].copy()
    index_rows: list[dict[str, str]] = []

    image_rank = image_aggregate(nonbase).sort_values("image_sensitivity_score", ascending=False)
    image_rank["robustness_rank_score"] = (
        (1 - image_rank["mean_textual_change"].fillna(0))
        + (1 - image_rank["mean_visual_change"].fillna(0))
        + (1 - image_rank["mean_prompt_dominated_candidate_score"].fillna(0))
        + (1 - image_rank["mean_unstable_explanation_candidate_score"].fillna(0))
        + image_rank["count_Q1"] / image_rank["n_cases"].clip(lower=1)
        - image_rank["decoupled_count"] / image_rank["n_cases"].clip(lower=1)
    )
    index_rows.append(write_bundle(
        "image_prompt_sensitivity_ranking",
        "Quali immagini sono piu sensibili al cambio prompt?",
        image_rank,
        sources=[CASE_PATH, CLUSTER_PATH],
        criteria=["Solo prompt non-baseline.", "Aggregazione per image_id.", "Score composito medio su textual_change, visual_change, prompt_dominated e unstable."],
        main_finding=f"Immagine piu sensibile candidata: `{int(image_rank.iloc[0]['image_id'])}` con score `{image_rank.iloc[0]['image_sensitivity_score']:.3f}`.",
        limits=["Ranking esplorativo basato su proxy diagnostici.", "Lo score composito pesa le componenti in modo uniforme."],
        aggregation="image_id",
    ))

    robust = image_rank.sort_values("robustness_rank_score", ascending=False).copy()
    (QUERY_ROOT / "robust_images_ranking").mkdir(parents=True, exist_ok=True)
    robust.to_csv(QUERY_ROOT / "robust_images_ranking" / "top_robust_images.csv", index=False)
    index_rows.append(write_bundle(
        "robust_images_ranking",
        "Quali immagini restano piu stabili su quasi tutti i prompt?",
        robust,
        sources=[CASE_PATH, CLUSTER_PATH],
        criteria=["Solo prompt non-baseline.", "Aggregazione per image_id.", "Preferisce basso drift, bassi diagnostic scores, piu Q1 e meno Q2/Q3/Q4."],
        main_finding=f"Immagine piu robusta candidata: `{int(robust.iloc[0]['image_id'])}` con robustness score `{robust.iloc[0]['robustness_rank_score']:.3f}`.",
        limits=["Contrasto diagnostico rispetto alle immagini instabili.", "Non dimostra robustezza causale."],
        aggregation="image_id",
        extra_files=["top_robust_images.csv"],
    ))

    dec = prompt_aggregate(nonbase).sort_values("rate_decoupling_Q3_Q4", ascending=False)
    index_rows.append(write_bundle(
        "decoupling_by_prompt",
        "Quali prompt producono piu decoupling testo-visione?",
        dec,
        sources=[CASE_PATH],
        criteria=["Solo prompt non-baseline.", "Decoupling = Q3 + Q4.", "Aggregazione per prompt_label e prompt_category."],
        main_finding=f"Prompt con rate decoupling piu alto: `{dec.iloc[0]['prompt_label']}` ({dec.iloc[0]['rate_decoupling_Q3_Q4']:.1%}).",
        limits=["Q3/Q4 sono quadranti diagnostici basati su soglie interne.", "Non e una misura causale di separazione testo-visione."],
        aggregation="prompt_label, prompt_category",
    ))

    q3 = nonbase.loc[nonbase["quadrant"].eq(Q3)].copy()
    q3["q3_rank_score"] = q3["textual_change"].fillna(0) - q3["visual_change"].fillna(0) + q3["prompt_dominated_candidate_score"].fillna(0)
    q3 = q3.sort_values(["q3_rank_score", "textual_change"], ascending=False)
    index_rows.append(write_bundle(
        "q3_text_changed_visual_stable",
        "Quali casi hanno testo molto cambiato ma heatmap/TAM relativamente stabile?",
        top_case_cols(q3),
        sources=[CASE_PATH],
        criteria=["Filtro quadrant == Q3_text_changed_visual_stable.", "Ordinamento per alto textual_change, basso visual_change e prompt_dominated alto."],
        main_finding=f"Trovati `{len(q3)}` casi Q3 da ispezionare qualitativamente.",
        limits=["Visual stability e textual drift sono proxy derivati dal confronto con baseline.", "La risposta baseline e riportata per contesto, non per scoring semantico profondo."],
        aggregation="case_id",
        top_cases=top_case_cols(q3),
    ))

    q4 = nonbase.loc[nonbase["quadrant"].eq(Q4)].copy()
    q4["q4_rank_score"] = q4["visual_change"].fillna(0) - q4["textual_change"].fillna(0) + q4["unstable_explanation_candidate_score"].fillna(0) + q4["mean_centroid_shift_vs_baseline"].fillna(0)
    q4 = q4.sort_values(["q4_rank_score", "visual_change"], ascending=False)
    index_rows.append(write_bundle(
        "q4_text_stable_visual_changed",
        "Quali casi hanno testo stabile ma heatmap/TAM molto cambiata?",
        top_case_cols(q4),
        sources=[CASE_PATH],
        criteria=["Filtro quadrant == Q4_text_stable_visual_changed.", "Ordinamento per alto visual_change, basso textual_change, unstable alto e centroid shift alto."],
        main_finding=f"Trovati `{len(q4)}` casi Q4 da ispezionare qualitativamente.",
        limits=["Q4 indica drift visuale relativo alla baseline, non fallimento causale.", "Centroid shift e un proxy spaziale aggregato."],
        aggregation="case_id",
        top_cases=top_case_cols(q4),
    ))

    prompt_cmp = prompt_aggregate(nonbase).sort_values("mean_visual_change", ascending=False)
    misleading = nonbase.loc[nonbase["prompt_label"].eq("misleading_wrong_subject")].copy()
    misleading["misleading_rank_score"] = safe_mean(misleading, ["textual_change", "visual_change", "prompt_dominated_candidate_score", "weak_grounding_candidate_score", "unstable_explanation_candidate_score"])
    misleading = misleading.sort_values("misleading_rank_score", ascending=False)
    index_rows.append(write_bundle(
        "misleading_prompt_effect",
        "`misleading_wrong_subject` produce piu drift rispetto agli altri prompt?",
        prompt_cmp,
        sources=[CASE_PATH],
        criteria=["Confronto tra prompt non-baseline.", "Focus interpretativo su misleading_wrong_subject.", "Metriche medie di drift e score diagnostici."],
        main_finding="`misleading_wrong_subject` e nel confronto prompt-level, ma va letto come proxy esplorativo e non come effetto causale.",
        limits=["Confronto descrittivo tra prompt, senza controllo causale.", "La categoria misleading puo produrre pattern eterogenei per immagine."],
        aggregation="prompt_label, prompt_category",
        top_cases=top_case_cols(misleading),
    ))

    grounded = prompt_cmp.copy()
    index_rows.append(write_bundle(
        "image_grounded_stability_query",
        "`image_grounded_visible_only` sembra stabilizzare rispetto agli altri prompt?",
        grounded.sort_values("mean_visual_change"),
        sources=[CASE_PATH],
        criteria=["Solo prompt non-baseline.", "Confronto per prompt su drift medio, score diagnostici e rate Q1/Q2/Q3/Q4.", "Focus su image_grounded_visible_only."],
        main_finding="`image_grounded_visible_only` va confrontato con gli altri prompt sul ranking: eventuale stabilizzazione resta un candidato proxy.",
        limits=["Risposta prudente e non causale.", "Le differenze possono dipendere dalla composizione delle immagini."],
        aggregation="prompt_label, prompt_category",
    ))

    extra = prompt_cmp.copy()
    for cid in sorted(clusters["cluster_id"].dropna().unique()):
        counts = nonbase.assign(hit=nonbase["cluster_id"].eq(cid)).groupby("prompt_label")["hit"].sum()
        extra[f"count_cluster_{int(cid)}"] = extra["prompt_label"].map(counts).fillna(0).astype(int)
    index_rows.append(write_bundle(
        "extra_knowledge_stability_query",
        "`extra_knowledge_context` introduce prior o resta stabile?",
        extra.sort_values("prompt_label"),
        sources=[CASE_PATH, CLUSTER_PATH],
        criteria=["Confronto prompt-level, con distribuzione cluster quando disponibile.", "Focus su extra_knowledge_context rispetto ad altri prompt non-baseline."],
        main_finding="La query riporta se `extra_knowledge_context` cade prevalentemente nei cluster piu stabili o in cluster di drift.",
        limits=["Cluster e proxy diagnostici, non etichette causali.", "Baseline esclusa dal confronto cluster per coerenza con clusters_v1."],
        aggregation="prompt_label, prompt_category",
    ))

    amb = prompt_aggregate(nonbase)
    amb_extra_cols = [
        "prompt_label",
        "prompt_category",
        "n_cases",
        "mean_weak_grounding_candidate_score",
        "mean_multipeak_ambiguity_score",
        "mean_multipeak_ratio_delta_vs_baseline",
        "mean_secondary_primary_ratio_mean_delta_vs_baseline",
        "mean_peak_count_mean_delta_vs_baseline",
        "rate_Q4",
    ]
    amb_metrics = nonbase.groupby(["prompt_label", "prompt_category"], dropna=False).agg(
        mean_multipeak_ratio_delta_vs_baseline=("multipeak_ratio_delta_vs_baseline", "mean"),
        mean_secondary_primary_ratio_mean_delta_vs_baseline=("secondary_primary_ratio_mean_delta_vs_baseline", "mean"),
        mean_peak_count_mean_delta_vs_baseline=("peak_count_mean_delta_vs_baseline", "mean"),
    ).reset_index()
    amb = amb.merge(amb_metrics, on=["prompt_label", "prompt_category"], how="left")
    amb = select_cols(amb, amb_extra_cols).sort_values("mean_multipeak_ambiguity_score", ascending=False)
    amb_cases = nonbase.loc[nonbase["prompt_label"].eq("ambiguous_open")].sort_values("multipeak_ambiguity_score", ascending=False)
    index_rows.append(write_bundle(
        "ambiguous_multipeak_query",
        "`ambiguous_open` produce piu ambiguita visuale/multipeak?",
        amb,
        sources=[CASE_PATH],
        criteria=["Aggregazione per prompt.", "Ranking su multipeak_ambiguity_score e metriche correlate ai picchi.", "Top casi solo per ambiguous_open."],
        main_finding=f"Prompt con multipeak ambiguity media piu alta: `{amb.iloc[0]['prompt_label']}`.",
        limits=["Multipeak e un proxy di struttura della heatmap, non una prova di ambiguita semantica.", "Richiede ispezione qualitativa dei casi top."],
        aggregation="prompt_label, prompt_category",
        top_cases=top_case_cols(amb_cases),
    ))

    centroid = nonbase.groupby(["prompt_label", "prompt_category"], dropna=False).agg(
        n_cases=("case_id", "count"),
        mean_centroid_shift=("mean_centroid_shift_vs_baseline", "mean"),
        median_centroid_shift=("mean_centroid_shift_vs_baseline", "median"),
        max_centroid_shift=("mean_centroid_shift_vs_baseline", "max"),
        mean_visual_change=("visual_change", "mean"),
    ).reset_index().sort_values("mean_centroid_shift", ascending=False)
    centroid_cases = nonbase.sort_values("mean_centroid_shift_vs_baseline", ascending=False)
    index_rows.append(write_bundle(
        "centroid_shift_by_prompt",
        "Quali prompt spostano di piu il centroide della heatmap?",
        centroid,
        sources=[CASE_PATH],
        criteria=["Solo prompt non-baseline.", "Aggregazione per prompt.", "Ranking per mean_centroid_shift_vs_baseline medio."],
        main_finding=f"Prompt con centroid shift medio piu alto: `{centroid.iloc[0]['prompt_label']}`.",
        limits=["Centroide e una sintesi spaziale: puo perdere dettagli locali o multipeak.", "Non valuta correttezza del grounding."],
        aggregation="prompt_label, prompt_category",
        top_cases=top_case_cols(centroid_cases),
    ))

    diffuse = nonbase.groupby(["prompt_label", "prompt_category"], dropna=False).agg(
        n_cases=("case_id", "count"),
        mean_entropy_delta=("entropy_mean_delta_vs_baseline", "mean"),
        mean_effective_area_delta=("effective_area_norm_mean_delta_vs_baseline", "mean"),
        mean_spread_trace_delta=("spread_trace_mean_delta_vs_baseline", "mean"),
        mean_top5_mass_delta=("top5_mass_mean_delta_vs_baseline", "mean"),
        mean_weak_grounding_candidate_score=("weak_grounding_candidate_score", "mean"),
    ).reset_index()
    diffuse["diffuse_proxy_score"] = diffuse[["mean_entropy_delta", "mean_effective_area_delta", "mean_spread_trace_delta"]].fillna(0).mean(axis=1) - diffuse["mean_top5_mass_delta"].fillna(0)
    diffuse = diffuse.sort_values("diffuse_proxy_score", ascending=False)
    index_rows.append(write_bundle(
        "diffuse_heatmaps_by_prompt",
        "Quali prompt rendono le heatmap piu diffuse o meno concentrate?",
        diffuse,
        sources=[CASE_PATH],
        criteria=["Aggregazione per prompt.", "Diffusione: entropia/effective area/spread piu alte; top5 mass piu alta indica concentrazione nei picchi."],
        main_finding=f"Prompt con diffuse proxy piu alto: `{diffuse.iloc[0]['prompt_label']}`.",
        limits=["Le scale delle metriche non sono identiche; diffuse_proxy_score e solo ordinamento diagnostico.", "Top5 mass si interpreta in direzione opposta rispetto alla diffusione."],
        aggregation="prompt_label, prompt_category",
    ))

    short = nonbase.copy()
    short["short_visual_score"] = short["visual_change"].fillna(0) - short["generated_word_count"].fillna(short["response_word_length"]).rank(pct=True)
    short = short.sort_values(["short_visual_score", "visual_change"], ascending=False)
    index_rows.append(write_bundle(
        "short_response_high_visual_drift",
        "Ci sono risposte brevi con forte drift visuale?",
        top_case_cols(short),
        sources=[CASE_PATH],
        criteria=["Solo prompt non-baseline.", "Ranking per visual_change alto e lunghezza risposta bassa.", "Usa generated_word_count/response_word_length."],
        main_finding="La tabella evidenzia casi in cui poche parole coincidono con forte visual drift TAM.",
        limits=["Lunghezza breve non implica causalmente drift visuale.", "Il ranking e utile per ispezione qualitativa."],
        aggregation="case_id",
        top_cases=top_case_cols(short),
    ))

    long_stable = nonbase.copy()
    long_stable["long_stable_score"] = long_stable["response_length_delta_vs_baseline"].fillna(0).rank(pct=True) - long_stable["visual_change"].fillna(0)
    long_stable = long_stable.sort_values(["long_stable_score", "response_length_delta_vs_baseline"], ascending=False)
    index_rows.append(write_bundle(
        "long_response_visual_stable",
        "Ci sono risposte molto piu lunghe ma visualmente stabili?",
        top_case_cols(long_stable),
        sources=[CASE_PATH],
        criteria=["Solo prompt non-baseline.", "Ranking per response_length_delta_vs_baseline alto e visual_change basso."],
        main_finding="La tabella raccoglie casi candidati in cui la verbosita aumenta senza forte drift visuale.",
        limits=["La lunghezza e confrontata alla baseline per immagine.", "Stabilita visuale resta un proxy TAM."],
        aggregation="case_id",
        top_cases=top_case_cols(long_stable),
    ))

    outlier = clusters.groupby(["prompt_label", "prompt_category", "quadrant", "cluster_id"], dropna=False).agg(
        n_cases=("case_id", "count"),
        count_outlier=("is_outlier", "sum"),
        mean_outlier_score=("outlier_score", "mean"),
    ).reset_index()
    outlier["outlier_rate"] = outlier["count_outlier"] / outlier["n_cases"].clip(lower=1)
    outlier = outlier.sort_values(["outlier_rate", "count_outlier"], ascending=False)
    index_rows.append(write_bundle(
        "outlier_prompt_distribution",
        "Gli outlier sono concentrati in pochi prompt?",
        outlier,
        sources=[CLUSTER_PATH],
        criteria=["Usa clusters_v1.parquet.", "Aggregazione per prompt, categoria, quadrante e cluster.", "Calcola count/rate outlier e mean outlier_score."],
        main_finding=f"Gruppo con outlier_rate piu alto: `{outlier.iloc[0]['prompt_label']}` / cluster `{outlier.iloc[0]['cluster_id']}`.",
        limits=["Outlier dipende dal modello di clustering/PCA usato in v2.", "Conta associazioni, non cause."],
        aggregation="prompt_label, prompt_category, quadrant, cluster_id",
    ))

    breakdowns = []
    for breakdown_type, col in [
        ("cluster_x_prompt_category", "prompt_category"),
        ("cluster_x_prompt_label", "prompt_label"),
        ("cluster_x_quadrant", "quadrant"),
        ("cluster_x_image_id", "image_id"),
    ]:
        tmp = clusters.groupby(["cluster_id", col], dropna=False).agg(
            n_cases=("case_id", "count"),
            mean_textual_change=("textual_change", "mean"),
            mean_visual_change=("visual_change", "mean"),
            mean_prompt_dominated_candidate_score=("prompt_dominated_candidate_score", "mean"),
            mean_unstable_explanation_candidate_score=("unstable_explanation_candidate_score", "mean"),
            mean_weak_grounding_candidate_score=("weak_grounding_candidate_score", "mean"),
        ).reset_index().rename(columns={col: "group_value"})
        totals = tmp.groupby("cluster_id")["n_cases"].transform("sum")
        tmp["rate_within_cluster"] = tmp["n_cases"] / totals.clip(lower=1)
        tmp.insert(0, "breakdown_type", breakdown_type)
        tmp["group_value"] = tmp["group_value"].astype(str)
        breakdowns.append(tmp)
    cluster_breakdown = pd.concat(breakdowns, ignore_index=True).sort_values(["breakdown_type", "cluster_id", "rate_within_cluster"], ascending=[True, True, False])
    index_rows.append(write_bundle(
        "cluster_prompt_image_breakdown",
        "I cluster sono guidati dal prompt o dalle immagini?",
        cluster_breakdown,
        sources=[CLUSTER_PATH],
        criteria=["Breakdown cluster x prompt_category, prompt_label, quadrant, image_id.", "Calcola rate within cluster e medie diagnostiche."],
        main_finding="Il file consente di vedere se ciascun cluster e dominato da prompt, quadranti o immagini specifiche.",
        limits=["Serve lettura per cluster; non assegna automaticamente una causa.", "Le immagini con pochi casi possono avere rate alti per numerosita ridotta."],
        aggregation="cluster_id x group",
    ))

    img_cluster = image_aggregate(nonbase).sort_values(["count_cluster_2", "count_outlier", "mean_visual_change"], ascending=False)
    img_cluster["rate_cluster_2"] = img_cluster["count_cluster_2"] / img_cluster["n_cases"].clip(lower=1)
    img_cluster["rate_outlier"] = img_cluster["count_outlier"] / img_cluster["n_cases"].clip(lower=1)
    index_rows.append(write_bundle(
        "images_often_in_cluster2_or_outlier",
        "Quali immagini finiscono spesso nel cluster instabile/alto drift?",
        img_cluster,
        sources=[CASE_PATH, CLUSTER_PATH],
        criteria=["Solo prompt non-baseline.", "Aggregazione per image_id.", "Conta cluster_id == 2 e is_outlier."],
        main_finding=f"Immagine piu frequente in cluster 2/outlier: `{int(img_cluster.iloc[0]['image_id'])}`.",
        limits=["Cluster 2 e letto come cluster ad alto drift solo in senso diagnostico.", "Outlier e cluster sono prodotti del v2 clustering."],
        aggregation="image_id",
    ))

    weak_thr = nonbase["weak_grounding_candidate_score"].quantile(0.75)
    prompt_med = nonbase["prompt_dominated_candidate_score"].median()
    weak_not_prompt = nonbase.loc[
        nonbase["weak_grounding_candidate_score"].ge(weak_thr)
        & nonbase["prompt_dominated_candidate_score"].le(prompt_med)
    ].copy()
    weak_not_prompt["contrast_score"] = weak_not_prompt["weak_grounding_candidate_score"] - weak_not_prompt["prompt_dominated_candidate_score"]
    weak_not_prompt = weak_not_prompt.sort_values("contrast_score", ascending=False)
    index_rows.append(write_bundle(
        "weak_grounding_not_prompt_dominated",
        "Dove il weak-grounding score e alto, ma prompt-dominated e basso/moderato?",
        top_case_cols(weak_not_prompt),
        sources=[CASE_PATH],
        criteria=[f"weak_grounding_candidate_score >= p75 ({weak_thr:.3f}).", f"prompt_dominated_candidate_score <= mediana ({prompt_med:.3f})."],
        main_finding=f"Trovati `{len(weak_not_prompt)}` casi candidati weak-grounding non dominati dal prompt.",
        limits=["Soglie quantiliche scelte per ranking diagnostico.", "Weak-grounding e prompt-dominated sono proxy, non labels definitive."],
        aggregation="case_id",
        top_cases=top_case_cols(weak_not_prompt),
    ))

    prompt_thr = nonbase["prompt_dominated_candidate_score"].quantile(0.75)
    weak_med = nonbase["weak_grounding_candidate_score"].median()
    prompt_not_weak = nonbase.loc[
        nonbase["prompt_dominated_candidate_score"].ge(prompt_thr)
        & nonbase["weak_grounding_candidate_score"].le(weak_med)
    ].copy()
    prompt_not_weak["contrast_score"] = prompt_not_weak["prompt_dominated_candidate_score"] - prompt_not_weak["weak_grounding_candidate_score"]
    prompt_not_weak = prompt_not_weak.sort_values("contrast_score", ascending=False)
    index_rows.append(write_bundle(
        "prompt_dominated_but_not_weak_grounding",
        "Dove il prompt-dominated score e alto, ma weak-grounding e basso?",
        top_case_cols(prompt_not_weak),
        sources=[CASE_PATH],
        criteria=[f"prompt_dominated_candidate_score >= p75 ({prompt_thr:.3f}).", f"weak_grounding_candidate_score <= mediana ({weak_med:.3f})."],
        main_finding=f"Trovati `{len(prompt_not_weak)}` casi candidati prompt-dominated ma non weak-grounding.",
        limits=["Soglie quantiliche scelte per ranking diagnostico.", "Non implica che il grounding sia corretto, solo che il proxy weak-grounding non e alto."],
        aggregation="case_id",
        top_cases=top_case_cols(prompt_not_weak),
    ))

    unexpected = merged.loc[coord_mask(merged) & ~merged["prompt_label"].isin(["order_disruption_stress", "colleague_obj_detection_hard"])].copy()
    unexpected["coordinate_like_score"] = (
        unexpected["bbox_style_output_flag"].fillna(0).astype(float)
        + unexpected["has_box_tokens"].fillna(0).astype(float)
        + unexpected["coordinate_token_count"].fillna(0).astype(float).clip(0, 8) / 8.0
        + unexpected["bbox_or_grounding_format_score"].fillna(0).astype(float)
    )
    unexpected = unexpected.sort_values("coordinate_like_score", ascending=False)
    index_rows.append(write_bundle(
        "unexpected_coordinate_outputs",
        "Quali coordinate/bbox outputs compaiono in prompt non attesi?",
        top_case_cols(unexpected),
        sources=[CASE_PATH],
        criteria=["Coordinate/bbox flag o regex true.", "Esclude order_disruption_stress e colleague_obj_detection_hard.", "Include anche baseline_neutral per intercettare output coordinate inattesi."],
        main_finding=f"Trovati `{len(unexpected)}` output coordinate/bbox inattesi nei prompt non previsti.",
        limits=["Rilevazione testuale/tabellare, non validazione geometrica.", "La presenza di coordinate nella baseline e descrittiva: non indica da sola errore o causalita."],
        aggregation="case_id",
        top_cases=top_case_cols(unexpected),
    ))

    coord_df = merged.loc[coord_mask(merged)].copy()
    coord_by_image = coord_df.groupby("image_id", dropna=False).agg(
        coordinate_cases=("case_id", "count"),
        coordinate_prompts=("prompt_label", lambda s: ", ".join(sorted(s.astype(str).unique()))),
        dashboard_compare_url=("dashboard_compare_url", "first"),
    ).reset_index()
    totals = merged.groupby("image_id")["case_id"].count()
    coord_by_image["total_cases"] = coord_by_image["image_id"].map(totals).fillna(0).astype(int)
    coord_by_image["coordinate_case_rate"] = coord_by_image["coordinate_cases"] / coord_by_image["total_cases"].clip(lower=1)
    coord_by_image = coord_by_image.sort_values(["coordinate_cases", "coordinate_case_rate"], ascending=False)
    index_rows.append(write_bundle(
        "coordinate_outputs_by_image",
        "Quali immagini generano spesso coordinate/bbox indipendentemente dal prompt?",
        coord_by_image,
        sources=[CASE_PATH],
        criteria=["Coordinate/bbox flag o regex true.", "Aggregazione per image_id su tutti gli 8 prompt.", "Riporta prompt coinvolti."],
        main_finding=f"Immagine con piu output coordinate/bbox: `{int(coord_by_image.iloc[0]['image_id'])}` con `{int(coord_by_image.iloc[0]['coordinate_cases'])}` prompt.",
        limits=["Predisposizione dell'immagine e solo una lettura descrittiva.", "Non verifica se le coordinate siano corrette o richieste."],
        aggregation="image_id",
    ))

    return index_rows


def write_index(rows: list[dict[str, str]]) -> None:
    out = QUERY_ROOT / "QUERY_INDEX.md"
    lines = [
        "# V5 Query Index",
        "",
        "Analisi esplorative basate su Parquet/CSV gia esportati. Tutti i risultati sono proxy diagnostici, non prove causali.",
        "",
        "| query_name | question | main aggregation | main output file | main finding breve | status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        vals = [
            row["query_name"],
            row["question"],
            row["main_aggregation"],
            row["main_output_file"],
            row["main_finding_breve"],
            row["status"],
        ]
        vals = [str(v).replace("|", "\\|").replace("\n", " ") for v in vals]
        lines.append("| " + " | ".join(vals) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    QUERY_ROOT.mkdir(parents=True, exist_ok=True)
    rows = build_queries()
    write_index(rows)
    print(f"Wrote {len(rows)} query bundles under {rel(QUERY_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
