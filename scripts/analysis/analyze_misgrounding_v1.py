from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from prompt_word_utils import build_word_lookup


CONTENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "its",
    "no",
    "of",
    "on",
    "or",
    "that",
    "the",
    "there",
    "this",
    "to",
    "was",
    "what",
    "why",
    "with",
}

PROMPT_RISK_PRIOR = {
    "baseline": 0.0,
    "image_grounded": 0.05,
    "ambiguous": 0.2,
    "misleading": 0.45,
    "contradictory": 0.5,
    "bias_prior": 0.4,
    "extra_knowledge": 0.35,
    "reasoning": 0.25,
    "micro_perturbation_lexical_ablation": 0.3,
    "micro_perturbation_order_disruption": 0.3,
}


def parse_args() -> argparse.Namespace:
    # Questa analisi non misura "misgrounding vero".
    # Costruisce solo una prima lista di casi candidati da leggere con prudenza.
    parser = argparse.ArgumentParser(
        description="Produce a first misgrounding candidate analysis from prompt sweep outputs and existing analyses.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a prompt sweep run directory.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def normalize_text(value: str) -> str:
    # Tolgo solo differenze banali di spaziatura.
    return re.sub(r"\s+", " ", value.strip())


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", word.strip().lower())


def content_words(words: list[str]) -> list[str]:
    # Per la divergenza della risposta tengo una vista lessicale semplice:
    # ripulita, leggibile e senza introdurre matching semantici opachi.
    cleaned = []
    for word in words:
        normalized = normalize_word(word)
        if not normalized:
            continue
        if normalized in CONTENT_STOPWORDS:
            continue
        cleaned.append(normalized)
    return cleaned


def content_words_from_metadata(metadata: dict) -> list[str]:
    # Per la divergenza della risposta uso le parole ricostruite,
    # non i token spezzati dal tokenizer.
    words = build_word_lookup(metadata)
    return content_words([word["word_label"] for word in words.values()])


def jaccard_similarity(words_a: list[str], words_b: list[str]) -> float:
    # Uso Jaccard perche' e' una misura facile da spiegare:
    # quanta parte del contenuto lessicale si sovrappone tra baseline e prompt.
    set_a = set(words_a)
    set_b = set(words_b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def to_float(value: str) -> float | None:
    raw = str(value).strip()
    if raw == "" or raw.lower() == "none":
        return None
    return float(raw)


def to_int(value: str) -> int:
    return int(str(value).strip())


def load_prompt_category_map(project_root: Path) -> dict[tuple[str, str], dict]:
    # Recupero la categoria del prompt dai JSON gia' presenti nel repo.
    # Se non la trovo, il sistema continua comunque a funzionare.
    prompt_map: dict[tuple[str, str], dict] = {}
    prompt_set_dir = project_root / "prompt_sets"
    if not prompt_set_dir.exists():
        return prompt_map

    for json_path in prompt_set_dir.glob("*.json"):
        payload = load_json(json_path)
        for entry in payload.get("prompts", []):
            key = (str(entry.get("id", "")), normalize_text(str(entry.get("prompt", ""))))
            prompt_map[key] = entry
    return prompt_map


def lookup_prompt_context(prompt_map: dict[tuple[str, str], dict], prompt_id: str, prompt_text: str) -> dict:
    return prompt_map.get((prompt_id, normalize_text(prompt_text)), {})


def stability_weakness(v4_score: float | None) -> float:
    # Traduco uno score di stabilita' in una piccola misura di "debolezza".
    # Se manca del tutto la stabilita', considero il segnale come massimo lato rischio.
    if v4_score is None:
        return 1.0
    return max(0.0, min(1.0, (0.999 - v4_score) / 0.02))


def coverage_weakness(valid_match_count: int, coverage_ratio: float) -> float:
    # Poca copertura non prova il misgrounding,
    # ma rende piu' fragile l'idea che il supporto visivo sia davvero solido.
    if valid_match_count <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - coverage_ratio))


def risk_prior(category: str | None) -> float:
    if not category:
        return 0.15
    return PROMPT_RISK_PRIOR.get(category, 0.15)


def label_case(
    valid_match_count: int,
    coverage_ratio: float,
    response_changed: bool,
    candidate_score: float,
) -> str:
    # Le label vogliono essere prudenti.
    # Se non ho abbastanza evidenza visiva, preferisco dirlo apertamente.
    if valid_match_count == 0:
        return "insufficient_evidence"
    if response_changed and coverage_ratio < 0.35 and candidate_score >= 0.55:
        return "prompt_dominated_candidate"
    if candidate_score >= 0.4:
        return "weakly_grounded_candidate"
    return "stable_grounded_candidate"


def build_case_rows(run_dir: Path) -> tuple[list[dict], list[dict], dict]:
    # Qui metto insieme i due lati della storia:
    # - come cambia la risposta
    # - quanto supporto visivo recupero da v4
    project_root = run_dir.parent.parent.parent.parent
    prompt_map = load_prompt_category_map(project_root)
    manifest = load_json(run_dir / "run_manifest.json")
    v4_rows = read_csv(run_dir / "analysis_v4" / "baseline_vs_prompt_aggregate_v4.csv")
    v4_by_prompt = {row["prompt_b"]: row for row in v4_rows}

    prompt_runs = manifest["prompt_runs"]
    baseline_run = prompt_runs[0]
    baseline_metadata = load_json(Path(baseline_run["metadata_path"]))
    baseline_response = baseline_metadata.get("response_text", "")
    baseline_content = content_words_from_metadata(baseline_metadata)
    baseline_available_words = len(build_word_lookup(baseline_metadata))

    case_rows = []
    summary_rows = []

    for prompt_run in prompt_runs[1:]:
        metadata = load_json(Path(prompt_run["metadata_path"]))
        prompt_label = prompt_run["prompt_label"]
        prompt_text = prompt_run["prompt_text"]
        prompt_id = prompt_run["prompt_id"]
        prompt_context = lookup_prompt_context(prompt_map, prompt_id, prompt_text)
        prompt_category = prompt_context.get("category", "")
        prompt_response = metadata.get("response_text", "")
        prompt_content = content_words_from_metadata(metadata)

        # Primo segnale semplice: la risposta e' cambiata oppure no.
        response_changed = normalize_text(prompt_response) != normalize_text(baseline_response)
        response_length_ratio = len(build_word_lookup(metadata)) / max(1, baseline_available_words)
        content_jaccard = jaccard_similarity(baseline_content, prompt_content)
        response_divergence_score = 1.0 - content_jaccard

        v4_row = v4_by_prompt[prompt_label]
        valid_match_count = to_int(v4_row["kept_comparisons"])
        coverage_ratio = valid_match_count / max(1, baseline_available_words)
        v4_score = to_float(v4_row["aggregate_heatmap_stability_score"])
        map_weakness = stability_weakness(v4_score)
        coverage_gap = coverage_weakness(valid_match_count, coverage_ratio)
        prior = risk_prior(prompt_category)

        # Lo score finale e' volutamente esplicito.
        # Ogni pezzo pesa poco ma in modo leggibile.
        candidate_score = (
            0.35 * (1.0 if response_changed else 0.0)
            + 0.25 * response_divergence_score
            + 0.20 * coverage_gap
            + 0.10 * map_weakness
            + 0.10 * prior
        )
        candidate_score = round(min(1.0, max(0.0, candidate_score)), 6)

        label = label_case(
            valid_match_count=valid_match_count,
            coverage_ratio=coverage_ratio,
            response_changed=response_changed,
            candidate_score=candidate_score,
        )

        # Salvo sia gli ingredienti dello score sia il risultato finale.
        # In questo modo il report resta verificabile, non una black box.
        case_row = {
            "prompt_id": prompt_id,
            "prompt_label": prompt_label,
            "prompt_category": prompt_category,
            "baseline_response": baseline_response,
            "prompt_response": prompt_response,
            "response_changed": response_changed,
            "response_length_ratio": round(response_length_ratio, 6),
            "baseline_content_signature": ", ".join(sorted(set(baseline_content))[:8]),
            "prompt_content_signature": ", ".join(sorted(set(prompt_content))[:8]),
            "content_jaccard_vs_baseline": round(content_jaccard, 6),
            "response_divergence_score": round(response_divergence_score, 6),
            "aggregate_heatmap_stability_score_v4": v4_score,
            "valid_match_count": valid_match_count,
            "coverage_ratio": round(coverage_ratio, 6),
            "coverage_weakness": round(coverage_gap, 6),
            "map_stability_weakness": round(map_weakness, 6),
            "prompt_risk_prior": round(prior, 6),
            "misgrounding_candidate_score": candidate_score,
            "label": label,
            "v4_excluded_reason_counts": v4_row["excluded_reason_counts"],
        }
        case_rows.append(case_row)

        summary_rows.append({
            "prompt_label": prompt_label,
            "prompt_category": prompt_category,
            "misgrounding_candidate_score": candidate_score,
            "label": label,
            "response_changed": response_changed,
            "aggregate_heatmap_stability_score_v4": v4_score,
            "valid_match_count": valid_match_count,
            "coverage_ratio": round(coverage_ratio, 6),
        })

    case_rows = sorted(case_rows, key=lambda row: row["misgrounding_candidate_score"], reverse=True)
    summary_rows = sorted(summary_rows, key=lambda row: row["misgrounding_candidate_score"], reverse=True)
    return case_rows, summary_rows, manifest


def build_markdown(case_rows: list[dict], manifest: dict, path: Path) -> None:
    # Il markdown deve essere leggibile anche da solo,
    # senza aprire i CSV o ricostruire a mente la formula.
    sufficient = [row for row in case_rows if row["label"] != "insufficient_evidence"]
    insufficient = [row for row in case_rows if row["label"] == "insufficient_evidence"]

    lines = [
        "# Misgrounding Candidate Analysis v1",
        "",
        f"- run_name: `{manifest['run_name']}`",
        f"- image_stem: `{manifest['image_stem']}`",
        f"- model_name: `{manifest['model_name']}`",
        "",
        "## Operational definition",
        "",
        "- This v1 does not claim true misgrounding.",
        "- It flags `candidate` cases where response divergence from baseline is combined with weak or sparse cross-prompt heatmap support.",
        "- Heatmap support comes from `analysis_v4`, while response divergence is computed directly from generated word content.",
        "",
        "## Signals used",
        "",
        "- `response_changed`: exact normalized response differs from baseline.",
        "- `response_divergence_score`: `1 - content_jaccard_vs_baseline`.",
        "- `aggregate_heatmap_stability_score_v4`: stability signal from v4.",
        "- `valid_match_count` and `coverage_ratio`: evidence coverage from v4.",
        "- `prompt_risk_prior`: small category-based contextual prior, never decisive on its own.",
        "",
        "## Scoring rule",
        "",
        "- `0.35 * response_changed`",
        "- `0.25 * response_divergence_score`",
        "- `0.20 * coverage_weakness`",
        "- `0.10 * map_stability_weakness`",
        "- `0.10 * prompt_risk_prior`",
        "",
        "## Ranking of candidate cases",
        "",
        "| prompt_label | category | score | label | response_changed | v4_score | valid_match_count | coverage_ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in case_rows:
        lines.append(
            f"| {row['prompt_label']} | {row['prompt_category']} | {row['misgrounding_candidate_score']} | "
            f"{row['label']} | {row['response_changed']} | {row['aggregate_heatmap_stability_score_v4']} | "
            f"{row['valid_match_count']} | {row['coverage_ratio']} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
    ])

    if sufficient:
        lines.append(
            f"- Most suspicious among cases with some evidence: `{sufficient[0]['prompt_label']}` with score `{sufficient[0]['misgrounding_candidate_score']}`."
        )
    if insufficient:
        lines.append(
            f"- Prompts with insufficient evidence: `{', '.join(row['prompt_label'] for row in insufficient)}`."
        )

    lines.extend([
        "- A high score here means `candidate for further inspection`, not proof of misgrounding.",
        "- Sparse coverage can itself be suspicious, but it also weakens interpretability, so those cases are labeled conservatively.",
        "",
        "## Limits",
        "",
        "- The score is an explicit heuristic, not a validated final metric.",
        "- Response divergence is lexical/content-based, not semantic.",
        "- Heatmap support still depends on the v4 matching and the overlay-derived stability measure.",
        "- Prompt category contributes only a small prior and should not be over-interpreted.",
        "",
        "## What would improve v2",
        "",
        "- A stronger but still transparent notion of response divergence.",
        "- More reliable visual evidence coverage per prompt.",
        "- Cross-image aggregation instead of a single-image ranking.",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    # Anche questa analisi e' completamente offline sugli artifact gia' presenti.
    # Questo la rende economica da rilanciare quando cambiano solo i criteri interpretativi.
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    case_rows, summary_rows, manifest = build_case_rows(run_dir)

    output_dir = run_dir / "analysis_misgrounding_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(case_rows, output_dir / "misgrounding_cases.csv")
    write_csv(summary_rows, output_dir / "misgrounding_summary.csv")
    write_json(
        {
            "manifest_summary": {
                "run_name": manifest["run_name"],
                "image_stem": manifest["image_stem"],
                "image_path": manifest["image_path"],
                "model_name": manifest["model_name"],
                "prompt_count": manifest["prompt_count"],
            },
            "scoring_definition": {
                "response_changed_weight": 0.35,
                "response_divergence_weight": 0.25,
                "coverage_weakness_weight": 0.20,
                "map_stability_weakness_weight": 0.10,
                "prompt_risk_prior_weight": 0.10,
            },
            "cases": case_rows,
        },
        output_dir / "misgrounding_report.json",
    )
    build_markdown(case_rows, manifest, output_dir / "misgrounding_report.md")

    print("Saved:")
    print(output_dir / "misgrounding_cases.csv")
    print(output_dir / "misgrounding_summary.csv")
    print(output_dir / "misgrounding_report.json")
    print(output_dir / "misgrounding_report.md")


if __name__ == "__main__":
    main()
