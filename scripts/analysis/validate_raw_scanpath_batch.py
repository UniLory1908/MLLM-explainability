from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_ROOT = PROJECT_ROOT / "outputs" / "prompt_sensitivity"

EXPECTED_RUNS = [
    ("000000331352_bathroom_toilet", "overnight_prompt_sensitivity_v2f_rawscanpath_bathroom_toilet"),
    ("000000426253_microwave_bottle", "overnight_prompt_sensitivity_v2f_rawscanpath_microwave_bottle"),
    ("000000030213_kitchen_counter", "overnight_prompt_sensitivity_v2f_rawscanpath_kitchen_counter"),
    ("000000555009_desk_monitor", "overnight_prompt_sensitivity_v2f_rawscanpath_desk_monitor"),
    ("000000393226_street_traffic", "overnight_prompt_sensitivity_v2f_rawscanpath_street_traffic"),
    ("000000133645_bench_boat", "overnight_prompt_sensitivity_v2f_rawscanpath_bench_boat"),
]


def validate_metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    steps = [step for step in metadata.get("step_records", []) if isinstance(step, dict)]
    raw_paths = [Path(str(step.get("raw_map_path", ""))) for step in steps if step.get("raw_map_path")]
    raw_existing = [path for path in raw_paths if path.exists()]
    dominant_count = sum(1 for step in steps if step.get("dominant_hotspot"))
    raw_source_count = sum(1 for step in steps if step.get("hotspot_source") == "raw_tam_map")
    return {
        "metadata_path": str(path),
        "prompt_label": metadata.get("prompt_label"),
        "scanpath_source": metadata.get("scanpath", {}).get("source", "legacy"),
        "num_steps": len(steps),
        "raw_map_paths": len(raw_paths),
        "raw_map_existing": len(raw_existing),
        "dominant_hotspots": dominant_count,
        "raw_source_steps": raw_source_count,
        "ok": bool(
            steps
            and metadata.get("scanpath", {}).get("source") == "raw_tam_map"
            and len(raw_paths) == len(steps)
            and len(raw_existing) == len(steps)
            and dominant_count == len(steps)
            and raw_source_count == len(steps)
        ),
    }


def main() -> None:
    rows: list[dict] = []
    for image_dir, run_name in EXPECTED_RUNS:
        run_root = PROMPT_ROOT / image_dir / run_name
        metadata_paths = sorted(run_root.glob("*/metadata.json"))
        rows.append({
            "image_dir": image_dir,
            "run_name": run_name,
            "run_root": str(run_root),
            "prompt_count": len(metadata_paths),
            "metadata": [validate_metadata(path) for path in metadata_paths],
        })

    for row in rows:
        row["ok"] = bool(row["prompt_count"] == 10 and all(item["ok"] for item in row["metadata"]))

    summary = {
        "expected_prompt_count_per_image": 10,
        "image_count": len(rows),
        "ok": all(row["ok"] for row in rows),
        "runs": rows,
    }

    PROMPT_ROOT.mkdir(parents=True, exist_ok=True)
    out_json = PROMPT_ROOT / "raw_scanpath_v2f_validation.json"
    out_txt = PROMPT_ROOT / "raw_scanpath_v2f_validation.txt"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"ok: {summary['ok']}",
        f"image_count: {summary['image_count']}",
        "",
    ]
    for row in rows:
        lines.append(f"{row['image_dir']} | prompts={row['prompt_count']} | ok={row['ok']}")
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_txt)


if __name__ == "__main__":
    main()
