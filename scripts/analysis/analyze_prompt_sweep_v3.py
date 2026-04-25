from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


STOPWORDS = {
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
    "only",
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

META_WORDS = {
    "describe",
    "explains",
    "explain",
    "image",
    "object",
    "objects",
    "scene",
    "step",
    "visible",
}


def parse_args() -> argparse.Namespace:
    # La v3 non ricalcola le heatmap.
    # Stringe solo il sottoinsieme dei confronti gia' costruiti dalla v2.
    parser = argparse.ArgumentParser(
        description="Build a more conservative heatmap stability analysis from analysis_v2 outputs.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a prompt sweep run directory containing analysis_v2 outputs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_word(word: str) -> str:
    # Qui tengo una forma pulita della parola per i filtri esatti.
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "", word.strip().lower())
    return cleaned


def is_content_word(word: str) -> bool:
    # Questa euristica vuole essere trasparente, non furba:
    # tolgo parole troppo corte, numeriche o troppo funzionali.
    normalized = normalize_word(word)
    if not normalized:
        return False
    if len(normalized) < 4:
        return False
    if normalized.isdigit():
        return False
    if normalized in STOPWORDS:
        return False
    if normalized in META_WORDS:
        return False
    return True


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def to_float(value: str) -> float | None:
    value = str(value).strip()
    if value == "" or value.lower() == "none":
        return None
    return float(value)


def build_filtered_rows(step_rows: list[dict]) -> list[dict]:
    # Costruisco una vista arricchita dei confronti word-level:
    # stessa riga di v2, ma con i flag dei filtri conservativi.
    filtered = []
    for row in step_rows:
        word_a = row.get("word_a", "")
        word_b = row.get("word_b", "")
        exact_match = to_bool(row.get("normalized_word_match", False))
        content_match = exact_match and is_content_word(word_a) and is_content_word(word_b)

        exclusion_reasons = []
        if not exact_match:
            exclusion_reasons.append("word_mismatch")
        elif not content_match:
            exclusion_reasons.append("non_content_word")

        filtered.append({
            **row,
            "keep_mode_exact_match": exact_match,
            "keep_mode_content_match": content_match,
            "filter_exclusion_reason": "" if not exclusion_reasons else ";".join(exclusion_reasons),
        })
    return filtered


def aggregate_mode(
    prompt_rows: list[dict],
    v2_row: dict,
    filter_mode: str,
    keep_key: str,
) -> dict:
    # Ogni modalita' di filtro produce un suo score aggregato.
    # Cosi' posso confrontare "quanto perdo" in copertura e "quanto guadagno" in credibilita'.
    kept_rows = [row for row in prompt_rows if to_bool(row[keep_key])]
    total_rows = len(prompt_rows)
    excluded_rows = total_rows - len(kept_rows)
    exclusion_counts = Counter()
    for row in prompt_rows:
        if to_bool(row[keep_key]):
            continue
        reason = row["filter_exclusion_reason"] or "unknown"
        exclusion_counts[reason] += 1

    cosine_values = [to_float(row["heatmap_stability_score"]) for row in kept_rows]
    cosine_values = [value for value in cosine_values if value is not None]

    aggregate_score = round(sum(cosine_values) / len(cosine_values), 6) if cosine_values else None
    v2_global = to_float(v2_row["aggregate_heatmap_stability_score"])
    delta_vs_v2 = round(aggregate_score - v2_global, 6) if aggregate_score is not None and v2_global is not None else None

    return {
        "prompt_a": v2_row["prompt_a"],
        "prompt_b": v2_row["prompt_b"],
        "filter_mode": filter_mode,
        "initial_v2_comparisons": total_rows,
        "kept_comparisons": len(kept_rows),
        "excluded_comparisons": excluded_rows,
        "excluded_reason_counts": json.dumps(dict(exclusion_counts), ensure_ascii=True, sort_keys=True),
        "v2_global_score": v2_global,
        "v3_filtered_score": aggregate_score,
        "delta_vs_v2": delta_vs_v2,
        "missing_artifact_words_from_v2": v2_row["missing_artifact_words"],
        "comparison_note": (
            "no_rows_kept"
            if not kept_rows
            else ("strict_content_subset" if filter_mode == "content_match" else "exact_word_subset")
        ),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_markdown(
    manifest: dict,
    aggregate_rows: list[dict],
    output_path: Path,
) -> None:
    # Qui il punto non e' solo il ranking.
    # Voglio rendere visibile anche quanto poco materiale resta dopo i filtri stretti.
    exact_rows = [row for row in aggregate_rows if row["filter_mode"] == "exact_match"]
    content_rows = [row for row in aggregate_rows if row["filter_mode"] == "content_match"]
    ranked_content = sorted(
        [row for row in content_rows if row["v3_filtered_score"] is not None],
        key=lambda row: row["v3_filtered_score"],
        reverse=True,
    )

    total_initial = sum(int(row["initial_v2_comparisons"]) for row in content_rows)
    total_kept = sum(int(row["kept_comparisons"]) for row in content_rows)
    total_excluded = sum(int(row["excluded_comparisons"]) for row in content_rows)

    lines = [
        "# Prompt Sweep Heatmap Stability v3",
        "",
        f"- run_name: `{manifest['run_name']}`",
        f"- image_stem: `{manifest['image_stem']}`",
        f"- model_name: `{manifest['model_name']}`",
        "- baseline comparison: `baseline vs each other prompt`",
        "",
        "## Filters",
        "",
        "- `exact_match`: keep only rows where `normalized_word_match == True`.",
        "- `content_match`: keep only rows already in `exact_match` and with word judged content-bearing by a small heuristic.",
        "- Content heuristic: alphanumeric word, length >= 4, non-numeric, not in a small stopword/meta-word list.",
        "",
        "## Conservative subset summary",
        "",
        f"- initial_v2_comparisons: `{total_initial}`",
        f"- kept_in_v3_content_match: `{total_kept}`",
        f"- excluded_in_v3_content_match: `{total_excluded}`",
        "",
        "## v3 ranking (content_match)",
        "",
        "| prompt_b | kept_comparisons | v2_global_score | v3_filtered_score | delta_vs_v2 | note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in ranked_content:
        lines.append(
            f"| {row['prompt_b']} | {row['kept_comparisons']} | {row['v2_global_score']} | "
            f"{row['v3_filtered_score']} | {row['delta_vs_v2']} | {row['comparison_note']} |"
        )

    lines.extend([
        "",
        "## exact_match vs content_match",
        "",
        "| prompt_b | exact_match_kept | content_match_kept | exact_match_score | content_match_score |",
        "| --- | --- | --- | --- | --- |",
    ])

    exact_by_prompt = {row["prompt_b"]: row for row in exact_rows}
    content_by_prompt = {row["prompt_b"]: row for row in content_rows}
    for prompt_b in sorted(content_by_prompt):
        exact_row = exact_by_prompt[prompt_b]
        content_row = content_by_prompt[prompt_b]
        lines.append(
            f"| {prompt_b} | {exact_row['kept_comparisons']} | {content_row['kept_comparisons']} | "
            f"{exact_row['v3_filtered_score']} | {content_row['v3_filtered_score']} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
    ])

    if ranked_content:
        lines.append(
            f"- Most stable under the stricter v3 filter: `{ranked_content[0]['prompt_b']}` with `{ranked_content[0]['v3_filtered_score']}`."
        )
        lines.append(
            f"- Least stable under the stricter v3 filter: `{ranked_content[-1]['prompt_b']}` with `{ranked_content[-1]['v3_filtered_score']}`."
        )
    else:
        lines.append("- No prompt kept enough rows under the strict content filter.")

    lines.extend([
        "- v3 is more credible than v2 because it discards positional matches and keeps only exact word agreement, optionally restricted to content-bearing words.",
        "- This is still a cautious subset analysis, not a full semantic alignment method.",
        "",
        "## Limits",
        "",
        "- The filter is intentionally simple and heuristic-based.",
        "- Some meaningful words may still be excluded if they look too short or too generic.",
        "- If exact word overlap is rare, the conservative subset becomes small.",
        "- The underlying heatmap measure remains the v2 approximation computed from saved overlays.",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    # La v3 vive sopra la v2:
    # se la v2 non c'e', questa analisi non ha senso metodologico.
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    manifest = load_json(run_dir / "run_manifest.json")
    analysis_v2_dir = run_dir / "analysis_v2"

    step_rows = read_csv(analysis_v2_dir / "baseline_vs_prompt_steps.csv")
    aggregate_v2_rows = read_csv(analysis_v2_dir / "baseline_vs_prompt_aggregate.csv")

    filtered_rows = build_filtered_rows(step_rows)

    rows_by_prompt = defaultdict(list)
    for row in filtered_rows:
        rows_by_prompt[row["prompt_b"]].append(row)

    aggregate_rows = []
    for v2_row in aggregate_v2_rows:
        prompt_rows = rows_by_prompt[v2_row["prompt_b"]]
        aggregate_rows.append(aggregate_mode(prompt_rows, v2_row, "exact_match", "keep_mode_exact_match"))
        aggregate_rows.append(aggregate_mode(prompt_rows, v2_row, "content_match", "keep_mode_content_match"))

    analysis_v3_dir = run_dir / "analysis_v3"
    analysis_v3_dir.mkdir(parents=True, exist_ok=True)

    write_csv(filtered_rows, analysis_v3_dir / "baseline_vs_prompt_steps_filtered.csv")
    write_csv(aggregate_rows, analysis_v3_dir / "baseline_vs_prompt_aggregate_filtered.csv")
    write_json(
        {
            "manifest_summary": {
                "run_name": manifest["run_name"],
                "image_stem": manifest["image_stem"],
                "image_path": manifest["image_path"],
                "model_name": manifest["model_name"],
                "prompt_count": manifest["prompt_count"],
            },
            "filter_definition": {
                "exact_match": "normalized_word_match == True",
                "content_match": "exact_match plus content-bearing word heuristic",
                "stopwords": sorted(STOPWORDS),
                "meta_words": sorted(META_WORDS),
            },
            "aggregate_rows": aggregate_rows,
        },
        analysis_v3_dir / "heatmap_stability_report_v3.json",
    )
    build_markdown(manifest, aggregate_rows, analysis_v3_dir / "heatmap_stability_report_v3.md")

    print("Saved:")
    print(analysis_v3_dir / "baseline_vs_prompt_steps_filtered.csv")
    print(analysis_v3_dir / "baseline_vs_prompt_aggregate_filtered.csv")
    print(analysis_v3_dir / "heatmap_stability_report_v3.json")
    print(analysis_v3_dir / "heatmap_stability_report_v3.md")


if __name__ == "__main__":
    main()
