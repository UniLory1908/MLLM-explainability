from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    # Analisi offline sugli artifact gia' prodotti dal prompt sweep.
    parser = argparse.ArgumentParser(
        description="Analyze outputs produced by run_qwen_tam_prompt_sweep.py",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a prompt sweep run directory containing run_manifest.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_word(word: str) -> str:
    # Normalizzazione leggera per confronti esplorativi.
    word = word.strip()
    if not word:
        return ""
    if word.startswith("<|") and word.endswith("|>"):
        return ""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "", word.lower())
    return cleaned


def normalize_text(value: str) -> str:
    # Elimina differenze banali di spaziatura nelle risposte.
    compact = re.sub(r"\s+", " ", value.strip())
    return compact


def first_heatmap_path(step_records: list[dict]) -> str:
    # Recupera un path minimale da mostrare nel report.
    if not step_records:
        return ""
    first = step_records[0]
    if "heatmap_path" in first:
        return first["heatmap_path"]
    layer_heatmaps = first.get("layer_heatmaps", {})
    if not layer_heatmaps:
        return ""
    first_key = sorted(layer_heatmaps, key=lambda raw: int(raw))[0]
    return layer_heatmaps[first_key]


def final_heatmap_path(step_records: list[dict]) -> str:
    # Stessa logica del primo path, ma riferita all'ultimo step.
    if not step_records:
        return ""
    last = step_records[-1]
    if "heatmap_path" in last:
        return last["heatmap_path"]
    layer_heatmaps = last.get("layer_heatmaps", {})
    if not layer_heatmaps:
        return ""
    last_key = sorted(layer_heatmaps, key=lambda raw: int(raw))[-1]
    return layer_heatmaps[last_key]


def content_words(word_labels: list[str]) -> list[str]:
    # Rimuove parole speciali e costruisce una traccia testuale semplice.
    cleaned = []
    for word in word_labels:
        normalized = normalize_word(word)
        if normalized:
            cleaned.append(normalized)
    return cleaned


def first_divergence(words_a: list[str], words_b: list[str]) -> tuple[int | None, str, str]:
    # Individua il primo punto di divergenza senza usare matching complessi.
    limit = min(len(words_a), len(words_b))
    for idx in range(limit):
        if words_a[idx] != words_b[idx]:
            return idx, words_a[idx], words_b[idx]
    if len(words_a) != len(words_b):
        next_a = words_a[limit] if limit < len(words_a) else "<END>"
        next_b = words_b[limit] if limit < len(words_b) else "<END>"
        return limit, next_a, next_b
    return None, "", ""


def distinct_content_signature(words: list[str], top_k: int = 6) -> str:
    # Costruisce una piccola firma del contenuto generato.
    counts = Counter(words)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(word for word, _ in ordered[:top_k])


def load_run_records(run_dir: Path) -> tuple[dict, list[dict]]:
    # Usa il manifest per enumerare i prompt e i metadata per i dettagli.
    manifest = load_json(run_dir / "run_manifest.json")
    records = []
    for prompt_run in manifest["prompt_runs"]:
        metadata = load_json(Path(prompt_run["metadata_path"]))
        word_labels = metadata.get("generated_word_labels", metadata.get("generated_token_labels", []))
        content = content_words(word_labels)
        step_records = metadata.get("step_records", [])
        record = {
            "prompt_index": prompt_run["prompt_index"],
            "prompt_id": prompt_run["prompt_id"],
            "prompt_label": prompt_run["prompt_label"],
            "prompt_text": prompt_run["prompt_text"],
            "response_text": metadata.get("response_text", ""),
            "response_text_normalized": normalize_text(metadata.get("response_text", "")),
            "num_generated_words": len(word_labels),
            "generated_word_labels": word_labels,
            "content_words": content,
            "content_signature": distinct_content_signature(content),
            "metadata_path": prompt_run["metadata_path"],
            "prompt_dir": prompt_run["prompt_dir"],
            "first_heatmap_path": first_heatmap_path(step_records),
            "final_heatmap_path": final_heatmap_path(step_records),
            "word_grid_root": metadata.get("grids_dir", ""),
        }
        records.append(record)
    return manifest, sorted(records, key=lambda item: item["prompt_index"])


def build_comparison_rows(records: list[dict]) -> list[dict]:
    # Usa il primo prompt del run come baseline locale.
    if not records:
        return []

    baseline = records[0]
    rows = []
    for record in records:
        divergence_idx, baseline_word, current_word = first_divergence(
            baseline["content_words"],
            record["content_words"],
        )
        rows.append({
            "prompt_index": record["prompt_index"],
            "prompt_id": record["prompt_id"],
            "prompt_label": record["prompt_label"],
            "prompt_text": record["prompt_text"],
            "response_text": record["response_text"],
            "num_generated_words": record["num_generated_words"],
            "matches_baseline_response": record["response_text_normalized"] == baseline["response_text_normalized"],
            "first_divergence_vs_baseline": "" if divergence_idx is None else divergence_idx,
            "baseline_word_at_divergence": baseline_word,
            "current_word_at_divergence": current_word,
            "content_signature": record["content_signature"],
            "metadata_path": record["metadata_path"],
            "first_heatmap_path": record["first_heatmap_path"],
            "final_heatmap_path": record["final_heatmap_path"],
            "word_grid_root": record["word_grid_root"],
        })
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(manifest: dict, comparison_rows: list[dict], path: Path) -> None:
    # Report compatto, leggibile anche senza aprire i CSV.
    lines = [
        "# Prompt Sweep Analysis",
        "",
        f"- run_name: `{manifest['run_name']}`",
        f"- image_stem: `{manifest['image_stem']}`",
        f"- image_path: `{manifest['image_path']}`",
        f"- model_name: `{manifest['model_name']}`",
        f"- prompt_count: `{manifest['prompt_count']}`",
        "",
        "## Per-prompt summary",
        "",
        "| idx | prompt_id | label | generated_words | response | final_heatmap |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in comparison_rows:
        response = row["response_text"].replace("\n", " ").replace("|", "\\|")
        heatmap_name = Path(row["final_heatmap_path"]).name if row["final_heatmap_path"] else ""
        lines.append(
            f"| {row['prompt_index']} | {row['prompt_id']} | {row['prompt_label']} | "
            f"{row['num_generated_words']} | {response} | {heatmap_name} |"
        )

    lines.extend([
        "",
        "## Baseline comparison",
        "",
        "Baseline prompt is the first row in the run.",
        "",
        "| label | same_response_as_baseline | first_divergence_vs_baseline | baseline_word | current_word | content_signature |",
        "| --- | --- | --- | --- | --- | --- |",
    ])

    for row in comparison_rows:
        lines.append(
            f"| {row['prompt_label']} | {row['matches_baseline_response']} | "
            f"{row['first_divergence_vs_baseline']} | {row['baseline_word_at_divergence']} | "
            f"{row['current_word_at_divergence']} | {row['content_signature']} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- `first_divergence_vs_baseline` is computed on a lightweight normalized word stream, not a full alignment metric.",
        "- The report is ready to host later metrics such as cross-prompt stability or more robust word matching.",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    # Analisi offline sugli artifact salvati.
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    manifest, records = load_run_records(run_dir)
    comparison_rows = build_comparison_rows(records)

    analysis_dir = run_dir / "analysis_v1"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    write_csv(comparison_rows, analysis_dir / "prompt_comparison.csv")
    write_json(
        {
            "manifest_summary": {
                "run_name": manifest["run_name"],
                "image_stem": manifest["image_stem"],
                "image_path": manifest["image_path"],
                "model_name": manifest["model_name"],
                "prompt_count": manifest["prompt_count"],
            },
            "comparison_rows": comparison_rows,
            "future_metrics_placeholders": {
                "cross_prompt_stability": None,
                "word_matching_robust": None,
                "misgrounding_rate": None,
                "alignment_metrics": None,
            },
        },
        analysis_dir / "analysis_report.json",
    )
    write_markdown(manifest, comparison_rows, analysis_dir / "analysis_report.md")

    print("Saved:")
    print(analysis_dir / "prompt_comparison.csv")
    print(analysis_dir / "analysis_report.json")
    print(analysis_dir / "analysis_report.md")


if __name__ == "__main__":
    main()
