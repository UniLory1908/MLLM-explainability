from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact cross-image summary from an overnight prompt sweep batch.",
    )
    parser.add_argument(
        "--batch-dir",
        required=True,
        help="Path to outputs/overnight_runs/<run_name>.",
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


def to_float(value: str | float | None) -> float | None:
    if value is None:
        return None
    raw = str(value).strip()
    if raw == "" or raw.lower() == "none":
        return None
    return float(raw)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def main() -> None:
    args = parse_args()
    batch_dir = Path(args.batch_dir).resolve()
    summary = load_json(batch_dir / "overnight_summary.json")

    per_image_rows = []
    prompt_detail_rows = []

    for run_row in summary["runs"]:
        run_dir = Path(run_row["run_dir"])
        prompt_runs = read_csv(run_dir / "prompt_runs.csv")
        v2_rows = {row["prompt_b"]: row for row in read_csv(run_dir / "analysis_v2" / "baseline_vs_prompt_aggregate.csv")}
        v3_rows = {
            row["prompt_b"]: row
            for row in read_csv(run_dir / "analysis_v3" / "baseline_vs_prompt_aggregate_filtered.csv")
            if row["filter_mode"] == "content_match"
        }
        v4_rows = {row["prompt_b"]: row for row in read_csv(run_dir / "analysis_v4" / "comparison_v2_v3_v4.csv")}
        mis_rows = {row["prompt_label"]: row for row in read_csv(run_dir / "analysis_misgrounding_v1" / "misgrounding_summary.csv")}

        per_image_rows.append({
            "img_id": run_row["img_id"],
            "image_stem": run_row["image_stem"],
            "image_label": run_row.get("image_label", ""),
            "image_dir_name": run_dir.parent.name,
            "selection_role": run_row["selection_role"],
            "runner_seconds": run_row["runner_seconds"],
            "analysis_v2_seconds": run_row["analysis_seconds"]["analysis_v2"],
            "analysis_v4_seconds": run_row["analysis_seconds"]["analysis_v4"],
            "total_seconds": run_row["total_seconds"],
            "run_dir": run_row["run_dir"],
        })

        for prompt_row in prompt_runs[1:]:
            label = prompt_row["prompt_label"]
            v2 = v2_rows.get(label, {})
            v3 = v3_rows.get(label, {})
            v4 = v4_rows.get(label, {})
            mis = mis_rows.get(label, {})

            prompt_detail_rows.append({
                "img_id": run_row["img_id"],
                "image_stem": run_row["image_stem"],
                "image_label": run_row.get("image_label", ""),
                "selection_role": run_row["selection_role"],
                "prompt_label": label,
                "prompt_text": prompt_row["prompt_text"],
                "response_text": prompt_row["response_text"],
                "num_rounds": prompt_row["num_rounds"],
                "v2_score": to_float(v2.get("aggregate_heatmap_stability_score")),
                "v2_exact_word_matches": v2.get("exact_word_matches", ""),
                "v3_score": to_float(v3.get("v3_filtered_score")),
                "v3_kept_comparisons": v3.get("kept_comparisons", ""),
                "v4_score": to_float(v4.get("v4_score")),
                "v4_used_comparisons": v4.get("v4_used_comparisons", ""),
                "misgrounding_candidate_score": to_float(mis.get("misgrounding_candidate_score")),
                "misgrounding_label": mis.get("label", ""),
            })

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in prompt_detail_rows:
        grouped[row["prompt_label"]].append(row)

    prompt_summary_rows = []
    for prompt_label in sorted(grouped):
        rows = grouped[prompt_label]
        prompt_summary_rows.append({
            "prompt_label": prompt_label,
            "images_covered": len(rows),
            "mean_num_rounds": mean([float(row["num_rounds"]) for row in rows]),
            "mean_v2_score": mean([value for value in (to_float(row["v2_score"]) for row in rows) if value is not None]),
            "mean_v3_score": mean([value for value in (to_float(row["v3_score"]) for row in rows) if value is not None]),
            "mean_v4_score": mean([value for value in (to_float(row["v4_score"]) for row in rows) if value is not None]),
            "mean_misgrounding_candidate_score": mean(
                [value for value in (to_float(row["misgrounding_candidate_score"]) for row in rows) if value is not None]
            ),
            "prompt_dominated_cases": sum(1 for row in rows if row["misgrounding_label"] == "prompt_dominated_candidate"),
            "insufficient_evidence_cases": sum(1 for row in rows if row["misgrounding_label"] == "insufficient_evidence"),
        })

    write_csv(per_image_rows, batch_dir / "per_image_run_summary.csv")
    write_csv(prompt_detail_rows, batch_dir / "cross_image_prompt_details.csv")
    write_csv(prompt_summary_rows, batch_dir / "cross_image_prompt_summary.csv")

    lines = [
        "# Cross-Image Prompt Summary",
        "",
        f"- batch_run: `{summary['run_name']}`",
        f"- prompt_set: `{summary['prompt_set_name']}`",
        f"- image_count: `{summary['image_count']}`",
        "",
        "## Per-image timing",
        "",
        "| img_id | image_label | role | total_seconds | run_dir |",
        "| --- | --- | --- | --- | --- |",
    ]

    for row in per_image_rows:
        lines.append(
            f"| {row['img_id']} | {row['image_label']} | {row['selection_role']} | {row['total_seconds']} | {row['run_dir']} |"
        )

    lines.extend([
        "",
        "## Cross-image prompt averages",
        "",
        "| prompt_label | images_covered | mean_num_rounds | mean_v2_score | mean_v3_score | mean_v4_score | mean_misgrounding_candidate_score | prompt_dominated_cases | insufficient_evidence_cases |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])

    for row in prompt_summary_rows:
        lines.append(
            f"| {row['prompt_label']} | {row['images_covered']} | {row['mean_num_rounds']} | {row['mean_v2_score']} | "
            f"{row['mean_v3_score']} | {row['mean_v4_score']} | {row['mean_misgrounding_candidate_score']} | "
            f"{row['prompt_dominated_cases']} | {row['insufficient_evidence_cases']} |"
        )

    (batch_dir / "cross_image_prompt_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
