from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.dashboard.precompute_derived_metrics import (
    FUNCTION_WORDS,
    categorize_token,
    content_words,
    lexical_comparison,
    response_words,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_PARQUET = PROJECT_ROOT / "outputs" / "statistical_archive" / "stat_timebox_20260523_progress" / "parquet"
V1_DIR = PROJECT_ROOT / "outputs" / "analysis" / "v1_case_level"
OUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "v4_qualitative_review"

SHORT_WORDS = [
    "dog",
    "cat",
    "bus",
    "car",
    "cup",
    "bed",
    "man",
    "boy",
    "cow",
    "tie",
    "ski",
    "hat",
    "bag",
    "bat",
    "bird",
    "boat",
    "bear",
    "kite",
    "bowl",
]

EXAMPLE_TARGETS = [
    (504074, "baseline_neutral"),
    (504074, "colleague_obj_detection_hard"),
    (156924, "order_disruption_stress"),
    (1761, "colleague_obj_detection_hard"),
    (378099, "ambiguous_open"),
    (402615, "order_disruption_stress"),
    (30213, "ambiguous_open"),
    (364557, "colleague_obj_detection_hard"),
    (555412, "misleading_wrong_subject"),
    (69213, "colleague_obj_detection_hard"),
    (262682, "extra_knowledge_context"),
]

FILES_READ = [
    "scripts/dashboard/precompute_derived_metrics.py",
    "scripts/dashboard/precompute_metrics.py",
    "scripts/dashboard/metrics.py",
    "scripts/dashboard/metric_registry.py",
    "scripts/analysis/build_case_level_analysis.py",
    "scripts/analysis/build_v2_interpretation_and_clustering.py",
    "outputs/statistical_archive/stat_timebox_20260523_progress/parquet/output_diagnostics.parquet",
    "outputs/statistical_archive/stat_timebox_20260523_progress/parquet/token_category_summary.parquet",
    "outputs/analysis/v1_case_level/analysis_case_level_v1.parquet",
    "outputs/analysis/v1_case_level/schema_snapshot.md",
    "outputs/analysis/v1_case_level/schema_snapshot.json",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit text metrics and current content-word filtering.")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser


def bool_text(value: bool) -> str:
    return "yes" if value else "no"


def shorten_text(text: object, limit: int = 220) -> str:
    clean = "" if pd.isna(text) else str(text).replace("\n", " ").strip()
    return clean if len(clean) <= limit else clean[: limit - 3].rstrip() + "..."


def make_short_word_check() -> pd.DataFrame:
    rows = []
    for word in SHORT_WORDS:
        normalized = response_words(word)
        kept = bool(normalized) and normalized[0] in content_words(word)
        if not normalized:
            reason = "not matched by response_words regex"
        elif normalized[0] in FUNCTION_WORDS:
            reason = "discarded because normalized word is in FUNCTION_WORDS"
        else:
            reason = "kept: regex token and not in FUNCTION_WORDS; no minimum length rule"
        rows.append(
            {
                "word": word,
                "length": len(word),
                "normalized_tokens": " ".join(normalized),
                "kept_by_current_filter": bool_text(kept),
                "token_category_if_word_label": categorize_token(word),
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def make_examples(case_level: pd.DataFrame) -> pd.DataFrame:
    baselines = (
        case_level.loc[case_level["prompt_label"].eq("baseline_neutral"), ["image_id", "response_text"]]
        .rename(columns={"response_text": "baseline_response_text"})
        .drop_duplicates("image_id")
    )
    merged = case_level.merge(baselines, on="image_id", how="left")
    rows = []
    for image_id, prompt_label in EXAMPLE_TARGETS:
        hit = merged.loc[merged["image_id"].eq(image_id) & merged["prompt_label"].eq(prompt_label)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        response = str(row.get("response_text") or "")
        baseline = str(row.get("baseline_response_text") or "")
        response_word_list = response_words(response)
        baseline_word_list = response_words(baseline)
        recomputed = lexical_comparison(response_word_list, baseline_word_list)
        rows.append(
            {
                "image_id": int(row["image_id"]),
                "case_id": row["case_id"],
                "prompt_label": row["prompt_label"],
                "response_text": response,
                "baseline_response_text": baseline,
                "content_words_baseline_extracted": " ".join(sorted(content_words(baseline))),
                "content_words_response_extracted": " ".join(sorted(content_words(response))),
                "content_jaccard_vs_baseline": row.get("content_jaccard_vs_baseline"),
                "content_jaccard_distance_vs_baseline": row.get("content_jaccard_distance_vs_baseline"),
                "matched_word_coverage_vs_baseline": row.get("matched_word_coverage_vs_baseline"),
                "first_divergence_word_index": row.get("first_divergence_word_index"),
                "first_divergence_ratio": row.get("first_divergence_ratio"),
                "recomputed_content_jaccard": recomputed["content_jaccard_vs_baseline"],
                "recomputed_matched_word_coverage": recomputed["matched_word_coverage_vs_baseline"],
                "recomputed_first_divergence_word_index": recomputed["first_divergence_word_index"],
                "recomputed_first_divergence_ratio": recomputed["first_divergence_ratio"],
            }
        )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    use = df[columns].head(max_rows) if max_rows else df[columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in use.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}" if pd.notna(value) else "")
            else:
                values.append(shorten_text(value, 120).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    short_df: pd.DataFrame,
    examples_df: pd.DataFrame,
    output_diag: pd.DataFrame,
    token_summary: pd.DataFrame,
    case_level: pd.DataFrame,
    schema: dict,
) -> None:
    output_cols = list(output_diag.columns)
    token_cols = list(token_summary.columns)
    case_cols = list(case_level.columns)
    text_metric_cols = [
        c
        for c in case_cols
        if any(
            key in c
            for key in [
                "jaccard",
                "matched_word",
                "first_divergence",
                "textual_change",
                "response_length",
                "token_overlap",
            ]
        )
    ]
    lines = [
        "# Text Metric Audit Report",
        "",
        "Audit in sola lettura delle metriche testuali e del filtro content words/content tokens.",
        "",
        "## File letti/ispezionati",
        "",
        *[f"- `{path}`" for path in FILES_READ],
        "",
        "## A. Regola esatta di selezione parole/content tokens",
        "",
        "- Tokenizzazione/normalizzazione: `response_words(text)` in `scripts/dashboard/precompute_derived_metrics.py`.",
        "- Implementazione: `re.findall(r\"[A-Za-z0-9_]+\", text.lower())`.",
        "- La punteggiatura viene rimossa perche non matcha la regex.",
        "- Le parole vengono lowercase.",
        "- Numeri e token alfanumerici/underscore vengono inclusi da `response_words` se matchano la regex.",
        "- Content words per le metriche lessicali: `content_words(text)` e la logica equivalente dentro `lexical_comparison`.",
        "- Regola content word: token in `response_words(text)` meno `FUNCTION_WORDS`.",
        "- Lista stopword presente: `FUNCTION_WORDS`.",
        f"- Numero stopword/function words: `{len(FUNCTION_WORDS)}`.",
        f"- Stopword/function words: `{', '.join(sorted(FUNCTION_WORDS))}`.",
        "- Soglia minima di lunghezza: Non presente nel codice ispezionato.",
        "- Whitelist/eccezioni per oggetti corti: Non presente nel codice ispezionato.",
        "- Esclusione numeri per `content_jaccard`: Non presente nel codice ispezionato; numeri non in `FUNCTION_WORDS` restano content words.",
        "- Stemming/lemmatizzazione: Non presente nel codice ispezionato.",
        "- Singular/plural folding: Non presente nel codice ispezionato.",
        "- Sinonimi: Non presente nel codice ispezionato.",
        "- LLM/semantic matching: Non presente nel codice ispezionato; confronto solo lessicale.",
        "",
        "Nota distinta: `token_category_summary` usa `categorize_token(token)`, non `content_words(text)`. Questa funzione classifica word labels in `special_token_like`, `coordinate_like`, `number_like`, `punctuation_like`, `function_like`, `attribute_like`, `content_like`, ma non determina `content_jaccard`.",
        "",
        "## B. Test parole corte visivamente importanti",
        "",
        md_table(short_df, ["word", "length", "kept_by_current_filter", "token_category_if_word_label", "reason"]),
        "",
        "Risultato: tutte le parole testate vengono tenute dal filtro content words corrente, perche sono token regex e non compaiono in `FUNCTION_WORDS`.",
        "",
        "## C. Colonne reali nei Parquet finali",
        "",
        f"- `output_diagnostics.parquet`: {len(output_diag)} righe, {len(output_cols)} colonne.",
        f"- `token_category_summary.parquet`: {len(token_summary)} righe, {len(token_cols)} colonne.",
        f"- `analysis_case_level_v1.parquet`: {len(case_level)} righe, {len(case_cols)} colonne.",
        "",
        "Colonne testuali/diagnostiche rilevanti in `analysis_case_level_v1`:",
        "",
        *[f"- `{col}`" for col in text_metric_cols],
        "",
        "Colonne `output_diagnostics`:",
        "",
        *[f"- `{col}`" for col in output_cols],
        "",
        "Colonne `token_category_summary`:",
        "",
        *[f"- `{col}`" for col in token_cols],
        "",
        "## D. Esempi reali dal dataset",
        "",
        "Le content words non sono salvate esplicitamente nei Parquet; qui sono ricostruite usando `content_words()` dal codice reale.",
        "",
        md_table(
            examples_df,
            [
                "image_id",
                "prompt_label",
                "response_text",
                "content_words_baseline_extracted",
                "content_words_response_extracted",
                "content_jaccard_vs_baseline",
                "content_jaccard_distance_vs_baseline",
                "matched_word_coverage_vs_baseline",
                "first_divergence_word_index",
                "first_divergence_ratio",
            ],
            max_rows=11,
        ),
        "",
        "CSV completo con baseline response e content words estratte: `text_metric_examples.csv`.",
        "",
        "## E. Formula reale delle metriche testuali",
        "",
        "### `response_words(text)`",
        "",
        "```python",
        "re.findall(r\"[A-Za-z0-9_]+\", text.lower())",
        "```",
        "",
        "### `content_jaccard_vs_baseline`",
        "",
        "```python",
        "content = {w for w in words if w not in FUNCTION_WORDS}",
        "base_content = {w for w in baseline if w not in FUNCTION_WORDS}",
        "union = content | base_content",
        "jaccard = len(content & base_content) / len(union) if union else 1.0",
        "content_jaccard_distance_vs_baseline = 1.0 - jaccard",
        "```",
        "",
        "### `matched_word_coverage_vs_baseline`",
        "",
        "```python",
        "word_counter = Counter(words)",
        "base_counter = Counter(baseline)",
        "overlap = sum((word_counter & base_counter).values())",
        "matched_word_count_vs_baseline = overlap",
        "matched_word_coverage_vs_baseline = overlap / max(len(words), 1)",
        "token_overlap_vs_baseline = overlap / max(len(words), len(baseline), 1)",
        "```",
        "",
        "Nota: `matched_word_coverage_vs_baseline` usa tutti i response words, incluse function words e numeri, non solo content words.",
        "",
        "### `first_divergence_word_index` / `first_divergence_ratio`",
        "",
        "```python",
        "matched_prefix = 0",
        "for left, right in zip(words, baseline):",
        "    if left != right:",
        "        break",
        "    matched_prefix += 1",
        "first_div = None if matched_prefix == len(words) == len(baseline) else matched_prefix",
        "first_divergence_ratio = None if first_div is None else first_div / max(len(words), len(baseline), 1)",
        "```",
        "",
        "## F. Formula reale di `textual_change`",
        "",
        "`textual_change` viene creato in `scripts/analysis/build_case_level_analysis.py`, funzione `add_text_visual_axes()`.",
        "",
        "Feature usate:",
        "",
        "- `content_jaccard_distance_vs_baseline`, rank percentile diretto.",
        "- `response_length_delta_vs_baseline`, valore assoluto poi rank percentile.",
        "- `response_length_ratio_vs_baseline`, rank percentile diretto.",
        "- `first_divergence_ratio`, invertito (`-x`) poi rank percentile.",
        "- `matched_word_coverage_vs_baseline`, invertito (`-x`) poi rank percentile.",
        "",
        "Pseudocodice fedele:",
        "",
        "```python",
        "for each feature:",
        "    s = numeric series",
        "    if absolute: s = abs(s)",
        "    if invert: s = -s",
        "    part = s.rank(pct=True, method='average').fillna(0).clip(0, 1)",
        "textual_change = mean(parts available).clip(0, 1)",
        "text_threshold = median(textual_change over non-baseline rows)",
        "text_changed = textual_change >= text_threshold",
        "```",
        "",
        f"Nel dataset v1, soglia testuale salvata: `{case_level['text_visual_threshold_text'].dropna().iloc[0]:.4f}`.",
        "",
        "Quindi usa rank percentile, non min-max, non z-score, non raw average. Include content Jaccard distance, response length delta/ratio, first divergence e matched word coverage.",
        "",
        "## G. Limiti reali confermati dal codice",
        "",
        "- Parole corte importanti come dog/cat/bus/car/cup/bed/man/boy/cow non vengono escluse dal filtro corrente.",
        "- Il rischio principale non e la lunghezza, ma la mancanza di normalizzazione semantica e morfologica.",
        "- Sinonimi non riconosciuti: `dog/animal`, `man/person`, `bicycle/bike` sono diversi se le stringhe differiscono.",
        "- Singular/plural non gestito: `dog/dogs`, `person/people`, `bus/buses` sono diversi se entrambi non compaiono identici.",
        "- Numeri e coordinate possono entrare nei content sets di `content_jaccard` se passano `response_words` e non sono in `FUNCTION_WORDS`; questo puo pesare nei prompt bbox/coordinate.",
        "- `content_jaccard` e robusto come segnale lessicale set-based, ma perde frequenze e ordine.",
        "- `matched_word_coverage` cattura overlap multiset ma include function words e non e semantico.",
        "- `first_divergence` e molto sensibile ai primi token e all'ordine; e utile per divergenza iniziale ma debole per parafrasi.",
        "- `textual_change` attenua singole debolezze combinando cinque feature rank-based, ma eredita i limiti lessicali delle feature sorgente.",
        "",
        "## H. Proposta futura: `text_diagnostics_v2` senza implementazione",
        "",
        "- Salvare esplicitamente `content_words_response` e `content_words_baseline` in una tabella diagnostica o export.",
        "- Aggiungere whitelist oggetti COCO corti e visivi, anche se oggi non sono scartati, per proteggere eventuali futuri filtri di lunghezza.",
        "- Escludere o separare numeri/coordinate dal Jaccard lessicale generale, mantenendo una metrica bbox-format dedicata.",
        "- Aggiungere normalizzazione semplice singolare/plurale per casi comuni.",
        "- Aggiungere mapping sinonimi controllato e documentato: `man/person`, `woman/person`, `bike/bicycle`, `dog/animal`, `cat/animal`, `airplane/plane`.",
        "- Tenere sia metriche lexical-strict sia lexical-normalized per non nascondere differenze reali.",
        "- Creare un report di casi in cui strict Jaccard e normalized Jaccard divergono molto.",
        "",
        "## Conferma scope",
        "",
        "Questo audit ha letto codice e Parquet esportati e ha creato solo report/CSV di audit. Non modifica metriche, Parquet, raw maps, DB/cache, dashboard runtime, archive canonico, notebook o standalone tool.",
        "",
    ]
    (out_dir / "text_metric_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    output_diag = pd.read_parquet(ARCHIVE_PARQUET / "output_diagnostics.parquet")
    token_summary = pd.read_parquet(ARCHIVE_PARQUET / "token_category_summary.parquet")
    case_level = pd.read_parquet(V1_DIR / "analysis_case_level_v1.parquet")
    schema = json.loads((V1_DIR / "schema_snapshot.json").read_text(encoding="utf-8"))
    # Read Markdown snapshot as part of the audit inputs.
    _schema_md = (V1_DIR / "schema_snapshot.md").read_text(encoding="utf-8")

    short_df = make_short_word_check()
    examples_df = make_examples(case_level)

    short_df.to_csv(out_dir / "text_metric_short_word_check.csv", index=False)
    examples_df.to_csv(out_dir / "text_metric_examples.csv", index=False)
    write_report(out_dir, short_df, examples_df, output_diag, token_summary, case_level, schema)
    print(f"Wrote text metric audit to {out_dir}")
    print(f"short_word_rows={len(short_df)} example_rows={len(examples_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
