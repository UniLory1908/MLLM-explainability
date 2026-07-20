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

from analyze_prompt_sweep_v2 import (
    cosine_similarity,
    load_records,
    normalized_l1_similarity,
)
from prompt_word_utils import canonicalize_word_text, combined_word_heatmap


HARD_STOPWORDS = {
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
    "with",
}


def parse_args() -> argparse.Namespace:
    # La v4 prova a recuperare un po' di copertura rispetto alla v3,
    # ma solo con regole lessicali facili da spiegare.
    parser = argparse.ArgumentParser(
        description="Build an intermediate baseline-vs-prompt heatmap stability analysis with controlled word matching.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a prompt sweep run directory containing analysis_v2 and analysis_v3 outputs.",
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


def normalize_word(word: str) -> str:
    # Base minima comune a tutti i confronti lessicali della v4.
    return re.sub(r"[^a-zA-Z0-9]+", "", word.strip().lower())


def canonicalize_word(word: str) -> str:
    return canonicalize_word_text(word)


def is_lexical_anchor(word: str) -> bool:
    # Non tutte le parole sono buoni candidati per il matching.
    # Tengo solo quelle che hanno abbastanza contenuto lessicale da essere utili.
    normalized = canonicalize_word(word)
    if not normalized:
        return False
    if len(normalized) < 4:
        return False
    if normalized.isdigit():
        return False
    if normalized in HARD_STOPWORDS:
        return False
    return True


def has_heatmap(step: dict) -> bool:
    # Il matching della v4 ha senso solo se poi posso davvero confrontare le mappe.
    return any(Path(path).exists() for path in step.get("source_heatmap_paths", []))


def match_type_for_steps(step_a: int, step_b: int) -> str:
    if step_a == step_b:
        return "exact_same_step"
    if abs(step_a - step_b) <= 2:
        return "local_window_match"
    return "global_relocated_match"


def find_controlled_match(
    baseline_step: dict,
    current_steps: dict[int, dict],
    used_current_steps: set[int],
) -> tuple[dict | None, str]:
    # Questa e' la regola chiave della v4.
    # Cerco un match piccolo ma difendibile:
    # prima stessa posizione, poi piccolo spostamento, poi match unico altrove.
    baseline_word = baseline_step.get("word_label", "")
    if not is_lexical_anchor(baseline_word):
        return None, "baseline_word_not_lexical_anchor"

    canonical_a = canonicalize_word(baseline_word)
    candidates = []
    for step_idx, step in current_steps.items():
        if step_idx in used_current_steps:
            continue
        if not has_heatmap(step):
            continue
        current_word = step.get("word_label", "")
        if canonicalize_word(current_word) != canonical_a:
            continue
        candidates.append((step_idx, step))

    if not candidates:
        return None, "no_equivalent_word_found"

    same_step = [item for item in candidates if item[0] == baseline_step["word_index"]]
    if len(same_step) == 1:
        return same_step[0][1], "exact_same_step"
    if len(same_step) > 1:
        return None, "ambiguous_same_step_match"

    local_window = [item for item in candidates if abs(item[0] - baseline_step["word_index"]) <= 2]
    if local_window:
        local_window = sorted(local_window, key=lambda item: (abs(item[0] - baseline_step["word_index"]), item[0]))
        best_distance = abs(local_window[0][0] - baseline_step["word_index"])
        best = [item for item in local_window if abs(item[0] - baseline_step["word_index"]) == best_distance]
        if len(best) == 1:
            return best[0][1], "local_window_match"
        return None, "ambiguous_local_window_match"

    if len(candidates) == 1:
        return candidates[0][1], "global_relocated_match"

    return None, "ambiguous_global_match"


def compare_prompt_pair_v4(baseline: dict, current: dict) -> tuple[list[dict], dict]:
    # Anche qui resto in baseline-vs-prompt.
    # La differenza rispetto alla v2 e' che il match non e' piu' puramente posizionale.
    baseline_steps = baseline["word_lookup"]
    current_steps = current["word_lookup"]
    used_current_steps: set[int] = set()
    step_rows = []
    exclusion_counts = Counter()

    for word_idx in sorted(baseline_steps):
        baseline_step = baseline_steps[word_idx]
        if not has_heatmap(baseline_step):
            exclusion_counts["missing_baseline_artifact"] += 1
            continue

        matched_step, match_reason = find_controlled_match(baseline_step, current_steps, used_current_steps)
        if matched_step is None:
            exclusion_counts[match_reason] += 1
            continue

        # Ogni step del prompt corrente puo' essere usato una volta sola.
        # Questo evita match gonfiati artificialmente.
        used_current_steps.add(int(matched_step["word_index"]))
        heatmap_a = combined_word_heatmap(baseline_step, baseline["metadata"]["image_path"])
        heatmap_b = combined_word_heatmap(matched_step, current["metadata"]["image_path"])
        if heatmap_a is None or heatmap_b is None:
            exclusion_counts["missing_word_artifact"] += 1
            continue
        cosine = cosine_similarity(heatmap_a, heatmap_b)
        l1_similarity = normalized_l1_similarity(heatmap_a, heatmap_b)

        step_rows.append({
            "prompt_a": baseline["prompt_label"],
            "prompt_b": current["prompt_label"],
            "word_index_a": baseline_step["word_index"],
            "word_index_b": matched_step["word_index"],
            "word_a": baseline_step.get("word_label", ""),
            "word_b": matched_step.get("word_label", ""),
            "canonical_word_a": canonicalize_word(baseline_step.get("word_label", "")),
            "canonical_word_b": canonicalize_word(matched_step.get("word_label", "")),
            "matching_rule": "controlled_match",
            "match_type": match_reason,
            "heatmap_cosine_similarity": round(cosine, 6),
            "heatmap_l1_similarity": round(l1_similarity, 6),
            "heatmap_stability_score": round(cosine, 6),
            "heatmap_a_path": ";".join(str(path) for path in baseline_step.get("source_heatmap_paths", [])),
            "heatmap_b_path": ";".join(str(path) for path in matched_step.get("source_heatmap_paths", [])),
            "note": "" if baseline_step["word_index"] == matched_step["word_index"] else "position_shifted",
        })

    cosine_values = [row["heatmap_stability_score"] for row in step_rows]
    l1_values = [row["heatmap_l1_similarity"] for row in step_rows]

    aggregate = {
        "prompt_a": baseline["prompt_label"],
        "prompt_b": current["prompt_label"],
        "matching_rule": "controlled_match",
        "available_baseline_words": len(baseline_steps),
        "kept_comparisons": len(step_rows),
        "excluded_comparisons": sum(exclusion_counts.values()),
        "excluded_reason_counts": json.dumps(dict(exclusion_counts), ensure_ascii=True, sort_keys=True),
        "aggregate_heatmap_stability_score": round(sum(cosine_values) / len(cosine_values), 6) if cosine_values else None,
        "aggregate_heatmap_l1_similarity": round(sum(l1_values) / len(l1_values), 6) if l1_values else None,
        "stability_std": round(float(__import__("numpy").std(cosine_values)), 6) if cosine_values else None,
        "comparison_note": "controlled_match_subset" if step_rows else "no_rows_kept",
    }
    return step_rows, aggregate


def build_comparison_rows(v2_rows: list[dict], v3_rows: list[dict], v4_rows: list[dict]) -> list[dict]:
    # Qui metto fianco a fianco le tre letture:
    # v2 larga, v3 stretta, v4 intermedia.
    v2_by_prompt = {row["prompt_b"]: row for row in v2_rows}
    v3_content_by_prompt = {
        row["prompt_b"]: row for row in v3_rows if row["filter_mode"] == "content_match"
    }
    v4_by_prompt = {row["prompt_b"]: row for row in v4_rows}

    rows = []
    for prompt_b in sorted(v4_by_prompt):
        v2 = v2_by_prompt[prompt_b]
        v3 = v3_content_by_prompt.get(prompt_b)
        v4 = v4_by_prompt[prompt_b]
        v2_score = float(v2["aggregate_heatmap_stability_score"]) if v2["aggregate_heatmap_stability_score"] else None
        v3_score = float(v3["v3_filtered_score"]) if v3 and v3["v3_filtered_score"] else None
        v4_score = float(v4["aggregate_heatmap_stability_score"]) if v4["aggregate_heatmap_stability_score"] else None
        rows.append({
            "prompt_b": prompt_b,
            "v2_used_comparisons": int(v2["matched_words"]),
            "v2_score": v2_score,
            "v3_used_comparisons": int(v3["kept_comparisons"]) if v3 else 0,
            "v3_score": v3_score,
            "v4_used_comparisons": int(v4["kept_comparisons"]),
            "v4_score": v4_score,
            "delta_v4_vs_v2": round(v4_score - v2_score, 6) if v4_score is not None and v2_score is not None else None,
            "delta_v4_vs_v3": round(v4_score - v3_score, 6) if v4_score is not None and v3_score is not None else None,
            "recovered_vs_v3": int(v4["kept_comparisons"]) - (int(v3["kept_comparisons"]) if v3 else 0),
        })
    return rows


def build_markdown(
    manifest: dict,
    v2_rows: list[dict],
    v3_rows: list[dict],
    v4_rows: list[dict],
    comparison_rows: list[dict],
    output_path: Path,
) -> None:
    # Il report deve far vedere se la v4 recupera davvero qualcosa di utile,
    # non solo se produce un numero in piu'.
    ranked_v4 = sorted(
        [row for row in v4_rows if row["aggregate_heatmap_stability_score"] is not None],
        key=lambda row: float(row["aggregate_heatmap_stability_score"]),
        reverse=True,
    )
    total_v2 = sum(int(row["matched_words"]) for row in v2_rows)
    total_v3 = sum(int(row["kept_comparisons"]) for row in v3_rows if row["filter_mode"] == "content_match")
    total_v4 = sum(int(row["kept_comparisons"]) for row in v4_rows)
    recovered = total_v4 - total_v3

    lines = [
        "# Prompt Sweep Heatmap Stability v4",
        "",
        f"- run_name: `{manifest['run_name']}`",
        f"- image_stem: `{manifest['image_stem']}`",
        f"- model_name: `{manifest['model_name']}`",
        "",
        "## controlled_match definition",
        "",
        "- Work only on final-layer-only runs.",
        "- Keep baseline-vs-prompt only.",
        "- Candidate baseline word must be a lexical anchor: alphanumeric, length >= 4, non-numeric, not in a small hard stopword list.",
        "- Word equivalence is controlled and transparent: lowercase normalization, punctuation removal, simple plural folding (`s`, `es`, `ies`).",
        "- Matching priority: same word position, then local window `+/- 2`, then unique global relocated match.",
        "- Matching is one-to-one on current prompt words.",
        "",
        "## Coverage",
        "",
        f"- v2 used comparisons: `{total_v2}`",
        f"- v3 content_match used comparisons: `{total_v3}`",
        f"- v4 controlled_match used comparisons: `{total_v4}`",
        f"- additional comparisons recovered vs v3: `{recovered}`",
        "",
        "## Ranking v4",
        "",
        "| prompt_b | kept_comparisons | aggregate_heatmap_stability_score | stability_std | note |",
        "| --- | --- | --- | --- | --- |",
    ]

    for row in ranked_v4:
        lines.append(
            f"| {row['prompt_b']} | {row['kept_comparisons']} | {row['aggregate_heatmap_stability_score']} | "
            f"{row['stability_std']} | {row['comparison_note']} |"
        )

    lines.extend([
        "",
        "## v2 vs v3 vs v4",
        "",
        "| prompt_b | v2_used | v2_score | v3_used | v3_score | v4_used | v4_score | recovered_vs_v3 | delta_v4_vs_v2 | delta_v4_vs_v3 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])

    for row in comparison_rows:
        lines.append(
            f"| {row['prompt_b']} | {row['v2_used_comparisons']} | {row['v2_score']} | "
            f"{row['v3_used_comparisons']} | {row['v3_score']} | {row['v4_used_comparisons']} | "
            f"{row['v4_score']} | {row['recovered_vs_v3']} | {row['delta_v4_vs_v2']} | {row['delta_v4_vs_v3']} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
    ])

    if ranked_v4:
        lines.append(
            f"- Most stable in v4: `{ranked_v4[0]['prompt_b']}` with score `{ranked_v4[0]['aggregate_heatmap_stability_score']}`."
        )
        lines.append(
            f"- Least stable in v4: `{ranked_v4[-1]['prompt_b']}` with score `{ranked_v4[-1]['aggregate_heatmap_stability_score']}`."
        )

    lines.extend([
        "- v4 is broader than v3 because it allows small position shifts and simple lexical equivalence, but it remains more defensible than v2 because it still requires explicit, manual matching rules.",
        "- Scores should still be interpreted cautiously: the same image stays fixed and the heatmap is reconstructed from the saved overlay.",
        "",
        "## Limits",
        "",
        "- `controlled_match` is still heuristic and lexical, not semantic.",
        "- Global relocated matches are allowed only when unique, but they are still weaker evidence than same-step matches.",
        "- Prompts with no lexical overlap remain uncovered.",
        "- This is probably strong enough for the current phase, but not yet a final grounding metric.",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    # La v4 riusa v2 e v3 perche' non voglio duplicare infrastruttura.
    # Mi interessa migliorare il confronto, non rifare la pipeline.
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    manifest, records = load_records(run_dir)
    baseline = records[0]

    analysis_v2_dir = run_dir / "analysis_v2"
    analysis_v3_dir = run_dir / "analysis_v3"
    v2_rows = read_csv(analysis_v2_dir / "baseline_vs_prompt_aggregate.csv")
    v3_rows = read_csv(analysis_v3_dir / "baseline_vs_prompt_aggregate_filtered.csv")

    all_step_rows = []
    aggregate_rows = []
    for record in records[1:]:
        step_rows, aggregate = compare_prompt_pair_v4(baseline, record)
        all_step_rows.extend(step_rows)
        aggregate_rows.append(aggregate)

    comparison_rows = build_comparison_rows(v2_rows, v3_rows, aggregate_rows)

    analysis_v4_dir = run_dir / "analysis_v4"
    analysis_v4_dir.mkdir(parents=True, exist_ok=True)

    write_csv(all_step_rows, analysis_v4_dir / "baseline_vs_prompt_steps_v4.csv")
    write_csv(aggregate_rows, analysis_v4_dir / "baseline_vs_prompt_aggregate_v4.csv")
    write_csv(comparison_rows, analysis_v4_dir / "comparison_v2_v3_v4.csv")
    write_json(
        {
            "manifest_summary": {
                "run_name": manifest["run_name"],
                "image_stem": manifest["image_stem"],
                "image_path": manifest["image_path"],
                "model_name": manifest["model_name"],
                "prompt_count": manifest["prompt_count"],
            },
            "matching_definition": {
                "controlled_match": {
                    "baseline_word_filter": "lexical anchor",
                    "canonicalization": "lowercase + punctuation removal + simple plural folding",
                    "priority": ["exact_same_step", "local_window_match", "global_relocated_match"],
                    "one_to_one_matching": True,
                }
            },
            "aggregate_rows": aggregate_rows,
            "comparison_rows": comparison_rows,
        },
        analysis_v4_dir / "heatmap_stability_report_v4.json",
    )
    build_markdown(
        manifest=manifest,
        v2_rows=v2_rows,
        v3_rows=v3_rows,
        v4_rows=aggregate_rows,
        comparison_rows=comparison_rows,
        output_path=analysis_v4_dir / "heatmap_stability_report_v4.md",
    )

    print("Saved:")
    print(analysis_v4_dir / "baseline_vs_prompt_steps_v4.csv")
    print(analysis_v4_dir / "baseline_vs_prompt_aggregate_v4.csv")
    print(analysis_v4_dir / "comparison_v2_v3_v4.csv")
    print(analysis_v4_dir / "heatmap_stability_report_v4.json")
    print(analysis_v4_dir / "heatmap_stability_report_v4.md")


if __name__ == "__main__":
    main()
