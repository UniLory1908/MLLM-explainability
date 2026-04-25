from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.prompt_word_utils import build_word_lookup, combined_word_heatmap  # noqa: E402


def parse_args() -> argparse.Namespace:
    # Prima misura di stabilita' costruita sugli artifact gia' salvati.
    parser = argparse.ArgumentParser(
        description="Compute baseline-vs-prompt heatmap stability on final-layer-only prompt sweep runs.",
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
    # Normalizzazione minima usata solo come annotazione.
    word = word.strip()
    if not word:
        return ""
    if word.startswith("<|") and word.endswith("|>"):
        return ""
    return re.sub(r"[^a-zA-Z0-9]+", "", word.lower())


def cosine_similarity(arr_a: np.ndarray, arr_b: np.ndarray) -> float:
    # Score principale di stabilita' tra heatmap.
    a = arr_a.astype(np.float32).reshape(-1)
    b = arr_b.astype(np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def normalized_l1_similarity(arr_a: np.ndarray, arr_b: np.ndarray) -> float:
    # Misura ausiliaria, utile come controllo rispetto alla cosine.
    a = arr_a.astype(np.float32)
    b = arr_b.astype(np.float32)
    mean_abs = float(np.mean(np.abs(a - b)))
    return float(max(0.0, 1.0 - (mean_abs / 255.0)))


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def resize_rgb(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def build_word_lookup_cached(metadata: dict) -> dict[int, dict]:
    # La v2 lavora a livello di parola ricostruita dai pezzi del tokenizer.
    return build_word_lookup(metadata)


def validate_final_layer_only(manifest: dict, records: list[dict]) -> None:
    # La v2 supporta solo run final-layer-only.
    if manifest.get("all_layers"):
        raise ValueError("analysis_v2 currently supports final-layer-only runs.")
    for record in records:
        if record["metadata"].get("all_layers"):
            raise ValueError("analysis_v2 currently supports only final-layer-only prompt runs.")


def load_records(run_dir: Path) -> tuple[dict, list[dict]]:
    # Carica il run intero; i confronti restano baseline vs singolo prompt.
    manifest = load_json(run_dir / "run_manifest.json")
    records = []
    for prompt_run in manifest["prompt_runs"]:
        metadata = load_json(Path(prompt_run["metadata_path"]))
        records.append({
            "prompt_index": prompt_run["prompt_index"],
            "prompt_id": prompt_run["prompt_id"],
            "prompt_label": prompt_run["prompt_label"],
            "prompt_text": prompt_run["prompt_text"],
            "metadata_path": prompt_run["metadata_path"],
            "prompt_dir": prompt_run["prompt_dir"],
            "metadata": metadata,
            "word_lookup": build_word_lookup_cached(metadata),
            "normalized_words": [normalize_word(word) for word in metadata.get("generated_word_labels", metadata.get("generated_token_labels", []))],
        })
    records = sorted(records, key=lambda item: item["prompt_index"])
    validate_final_layer_only(manifest, records)
    return manifest, records


def compare_prompt_pair(baseline: dict, current: dict) -> tuple[list[dict], dict]:
    # Confronto per parola ricostruita con allineamento posizionale.
    baseline_steps = baseline["word_lookup"]
    current_steps = current["word_lookup"]
    max_shared_steps = min(len(baseline_steps), len(current_steps))
    step_rows = []
    missing_artifact_steps = 0

    for word_idx in range(max_shared_steps):
        word_a = baseline_steps.get(word_idx)
        word_b = current_steps.get(word_idx)
        if word_a is None or word_b is None:
            continue
        path_a = [Path(path) for path in word_a.get("source_heatmap_paths", []) if str(path)]
        path_b = [Path(path) for path in word_b.get("source_heatmap_paths", []) if str(path)]
        # Le parole senza artifact visivi validi vengono escluse e conteggiate.
        if not path_a or not path_b or any(not path.exists() for path in path_a) or any(not path.exists() for path in path_b):
            missing_artifact_steps += 1
            continue

        heatmap_a = combined_word_heatmap(word_a, baseline["metadata"]["image_path"])
        heatmap_b = combined_word_heatmap(word_b, current["metadata"]["image_path"])
        if heatmap_a is None or heatmap_b is None:
            missing_artifact_steps += 1
            continue
        if heatmap_a.shape != heatmap_b.shape:
            raise ValueError(f"Heatmap shape mismatch at word {word_idx}: {heatmap_a.shape} vs {heatmap_b.shape}")

        word_label_a = word_a.get("word_label", "")
        word_label_b = word_b.get("word_label", "")
        norm_a = normalize_word(word_label_a)
        norm_b = normalize_word(word_label_b)

        # La cosine e' lo score principale; la L1 normalizzata resta una misura ausiliaria.
        cosine = cosine_similarity(heatmap_a, heatmap_b)
        l1_similarity = normalized_l1_similarity(heatmap_a, heatmap_b)

        step_rows.append({
            "prompt_a": baseline["prompt_label"],
            "prompt_b": current["prompt_label"],
            "word_index_a": word_idx,
            "word_index_b": word_idx,
            "word_a": word_label_a,
            "word_b": word_label_b,
            "source_steps_a": ",".join(str(idx) for idx in word_a.get("source_step_indices", [])),
            "source_steps_b": ",".join(str(idx) for idx in word_b.get("source_step_indices", [])),
            "matching_rule": "word_index",
            "normalized_word_match": norm_a != "" and norm_a == norm_b,
            "heatmap_cosine_similarity": round(cosine, 6),
            "heatmap_l1_similarity": round(l1_similarity, 6),
            "heatmap_stability_score": round(cosine, 6),
            "heatmap_a_path": ";".join(str(path) for path in path_a),
            "heatmap_b_path": ";".join(str(path) for path in path_b),
            "note": "" if norm_a == norm_b else "word_label_mismatch",
        })

    excluded_baseline_words = max(0, len(baseline_steps) - max_shared_steps)
    excluded_current_words = max(0, len(current_steps) - max_shared_steps)
    exact_word_rows = [row for row in step_rows if row["normalized_word_match"]]
    cosine_values = [row["heatmap_cosine_similarity"] for row in step_rows]
    l1_values = [row["heatmap_l1_similarity"] for row in step_rows]

    aggregate = {
        "prompt_a": baseline["prompt_label"],
        "prompt_b": current["prompt_label"],
        "matching_rule": "word_index",
        "matched_words": len(step_rows),
        "excluded_baseline_words": excluded_baseline_words,
        "excluded_prompt_words": excluded_current_words,
        "missing_artifact_words": missing_artifact_steps,
        "exact_word_matches": len(exact_word_rows),
        "aggregate_heatmap_stability_score": round(float(np.mean(cosine_values)), 6) if cosine_values else None,
        "aggregate_heatmap_l1_similarity": round(float(np.mean(l1_values)), 6) if l1_values else None,
        "aggregate_exact_word_stability_score": (
            round(float(np.mean([row["heatmap_cosine_similarity"] for row in exact_word_rows])), 6)
            if exact_word_rows else None
        ),
        "stability_std": round(float(np.std(cosine_values)), 6) if cosine_values else None,
        "comparison_note": (
            "no_shared_words"
            if not step_rows
            else (
                "all_words_aligned"
                if len(exact_word_rows) == len(step_rows)
                else "word_match_with_label_mismatch"
            )
        ),
    }
    return step_rows, aggregate


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(manifest: dict, aggregate_rows: list[dict], step_rows: list[dict], path: Path) -> None:
    # Report sintetico su copertura e stabilita' per prompt.
    comparable = [row for row in aggregate_rows if row["aggregate_heatmap_stability_score"] is not None]
    stable_sorted = sorted(comparable, key=lambda row: row["aggregate_heatmap_stability_score"], reverse=True)
    total_excluded = sum(
        row["excluded_baseline_words"] + row["excluded_prompt_words"] + row["missing_artifact_words"]
        for row in aggregate_rows
    )
    total_exact = sum(row["exact_word_matches"] for row in aggregate_rows)
    total_matched = sum(row["matched_words"] for row in aggregate_rows)

    lines = [
        "# Prompt Sweep Heatmap Stability v2",
        "",
        f"- run_name: `{manifest['run_name']}`",
        f"- image_stem: `{manifest['image_stem']}`",
        f"- model_name: `{manifest['model_name']}`",
        f"- baseline_prompt: `{aggregate_rows[0]['prompt_a'] if aggregate_rows else ''}`",
        f"- compared_prompts: `{len(aggregate_rows)}`",
        f"- matched_word_comparisons: `{total_matched}`",
        f"- excluded_words: `{total_excluded}`",
        f"- exact_word_matches_on_aligned_words: `{total_exact}`",
        "",
        "## Aggregate baseline-vs-prompt stability",
        "",
        "| prompt_b | matched_words | missing_artifact_words | exact_word_matches | aggregate_heatmap_stability_score | aggregate_exact_word_stability_score | stability_std | note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in stable_sorted:
        lines.append(
            f"| {row['prompt_b']} | {row['matched_words']} | {row['missing_artifact_words']} | {row['exact_word_matches']} | "
            f"{row['aggregate_heatmap_stability_score']} | {row['aggregate_exact_word_stability_score']} | "
            f"{row['stability_std']} | {row['comparison_note']} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
    ])

    if stable_sorted:
        lines.append(
            f"- Most stable vs baseline: `{stable_sorted[0]['prompt_b']}` with score `{stable_sorted[0]['aggregate_heatmap_stability_score']}`."
        )
        lines.append(
            f"- Least stable vs baseline: `{stable_sorted[-1]['prompt_b']}` with score `{stable_sorted[-1]['aggregate_heatmap_stability_score']}`."
        )
    else:
        lines.append("- No comparable prompts were found.")

    lines.extend([
        f"- Word matching used: `word_index` only.",
        f"- Word identity was treated as a conservative annotation (`normalized_word_match`), not as the main matcher.",
        "",
        "## Limits",
        "",
        "- The current metric works only on final-layer-only runs.",
        "- Heatmap comparison is computed from the saved TAM overlay after subtracting the resized original image, which is an approximation of the colorized heatmap.",
        "- Word alignment is positional, so semantic equivalence across differently phrased responses is not solved yet.",
        "- The aggregate score is a first stability signal, not yet a full grounding-faithfulness metric.",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    # Analisi offline del primo livello di stabilita' tra heatmap.
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    manifest, records = load_records(run_dir)
    baseline = records[0]

    all_step_rows = []
    aggregate_rows = []
    for record in records[1:]:
        step_rows, aggregate = compare_prompt_pair(baseline, record)
        all_step_rows.extend(step_rows)
        aggregate_rows.append(aggregate)

    analysis_dir = run_dir / "analysis_v2"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    write_csv(all_step_rows, analysis_dir / "baseline_vs_prompt_steps.csv")
    write_csv(aggregate_rows, analysis_dir / "baseline_vs_prompt_aggregate.csv")
    write_json(
        {
            "manifest_summary": {
                "run_name": manifest["run_name"],
                "image_stem": manifest["image_stem"],
                "image_path": manifest["image_path"],
                "model_name": manifest["model_name"],
                "prompt_count": manifest["prompt_count"],
            },
            "matching_strategy": {
                "primary_rule": "word_index",
                "secondary_annotation": "normalized_word_match",
                "supported_run_type": "final_layer_only",
            },
            "metric_definition": {
                "heatmap_stability_score": "cosine similarity on estimated heatmap RGB after subtracting the resized raw image from the saved TAM overlay",
                "auxiliary_metric": "normalized_l1_similarity",
            },
            "aggregate_rows": aggregate_rows,
        },
        analysis_dir / "heatmap_stability_report.json",
    )
    write_markdown(manifest, aggregate_rows, all_step_rows, analysis_dir / "heatmap_stability_report.md")

    print("Saved:")
    print(analysis_dir / "baseline_vs_prompt_steps.csv")
    print(analysis_dir / "baseline_vs_prompt_aggregate.csv")
    print(analysis_dir / "heatmap_stability_report.json")
    print(analysis_dir / "heatmap_stability_report.md")


if __name__ == "__main__":
    main()
