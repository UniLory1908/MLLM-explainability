from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1] if SCRIPT_DIR.name == "runs" else SCRIPT_DIR


def slugify(value: str, max_len: int = 80) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    compact = "_".join(chunk for chunk in cleaned.split("_") if chunk)
    return (compact or "run")[:max_len]


def image_dir_name(image_stem: str, image_label: str) -> str:
    return f"{image_stem}_{slugify(image_label)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the prompt sweep pipeline on multiple images and save a compact overnight summary.",
    )
    parser.add_argument(
        "--image-ids-file",
        required=True,
        help="JSON file with the list of COCO image ids to process.",
    )
    parser.add_argument(
        "--prompts-file",
        required=True,
        help="Prompt set JSON file passed to run_qwen_tam_prompt_sweep.py.",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Shared run name suffix used for every image.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Generation cap used for every prompt.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command: list[str], log_path: Path) -> float:
    # Esegue un comando e salva stdout e stderr in un log dedicato.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")
    return round(time.time() - start, 2)


def script_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath("scripts", *parts))


def main() -> None:
    args = parse_args()
    image_payload = load_json(Path(args.image_ids_file))
    prompt_payload = load_json(Path(args.prompts_file))
    selected_images = image_payload["images"]

    batch_root = PROJECT_ROOT / "outputs" / "overnight_runs" / args.run_name
    batch_root.mkdir(parents=True, exist_ok=True)

    pipeline_summary = {
        "run_name": args.run_name,
        "prompts_file": str(Path(args.prompts_file).resolve()),
        "prompt_set_name": prompt_payload.get("run_name", ""),
        "max_new_tokens": args.max_new_tokens,
        "image_count": len(selected_images),
        "images": selected_images,
        "runs": [],
    }

    for index, image_entry in enumerate(selected_images):
        img_id = int(image_entry["img_id"])
        image_stem = f"{img_id:012d}"
        image_label = str(image_entry.get("image_label") or image_entry.get("role") or image_stem)
        run_label = f"{args.run_name}_{image_label}"
        image_log_dir = batch_root / f"{index:02d}_{image_stem}_{slugify(image_label)}"
        image_log_dir.mkdir(parents=True, exist_ok=True)

        run_seconds = run_command(
            [
                sys.executable,
                script_path("runs", "run_qwen_tam_prompt_sweep.py"),
                "--img-id",
                str(img_id),
                "--prompts-file",
                args.prompts_file,
                "--run-name",
                run_label,
                "--image-label",
                image_label,
                "--final-layer-only",
                "--max-new-tokens",
                str(args.max_new_tokens),
            ],
            image_log_dir / "01_runner.log",
        )

        run_dir = PROJECT_ROOT / "outputs" / "prompt_sensitivity" / image_dir_name(image_stem, image_label) / slugify(run_label)
        analysis_steps = [
            ("analysis_v1", script_path("analysis", "analyze_prompt_sweep.py")),
            ("analysis_v2", script_path("analysis", "analyze_prompt_sweep_v2.py")),
            ("analysis_v3", script_path("analysis", "analyze_prompt_sweep_v3.py")),
            ("analysis_v4", script_path("analysis", "analyze_prompt_sweep_v4.py")),
            ("analysis_misgrounding_v1", script_path("analysis", "analyze_misgrounding_v1.py")),
        ]

        analysis_times = {}
        for order, (analysis_name, script_name) in enumerate(analysis_steps, start=2):
            analysis_times[analysis_name] = run_command(
                [
                    sys.executable,
                    script_name,
                    "--run-dir",
                    str(run_dir),
                ],
                image_log_dir / f"{order:02d}_{analysis_name}.log",
            )

        pipeline_summary["runs"].append(
            {
                "img_id": img_id,
                "image_stem": image_stem,
                "image_label": image_label,
                "image_dir_name": image_dir_name(image_stem, image_label),
                "selection_role": image_entry.get("role", ""),
                "selection_note": image_entry.get("note", ""),
                "run_dir": str(run_dir),
                "runner_seconds": run_seconds,
                "analysis_seconds": analysis_times,
                "total_seconds": round(run_seconds + sum(analysis_times.values()), 2),
            }
        )

        summary_path = batch_root / "overnight_summary.json"
        summary_path.write_text(json.dumps(pipeline_summary, indent=2), encoding="utf-8")

    total_seconds = round(sum(row["total_seconds"] for row in pipeline_summary["runs"]), 2)
    pipeline_summary["total_seconds"] = total_seconds
    pipeline_summary["total_minutes"] = round(total_seconds / 60.0, 2)
    (batch_root / "overnight_summary.json").write_text(json.dumps(pipeline_summary, indent=2), encoding="utf-8")

    lines = [
        "# Overnight Prompt Sweep Summary",
        "",
        f"- run_name: `{args.run_name}`",
        f"- prompts_file: `{Path(args.prompts_file).name}`",
        f"- max_new_tokens: `{args.max_new_tokens}`",
        f"- image_count: `{len(selected_images)}`",
        f"- total_seconds: `{total_seconds}`",
        f"- total_minutes: `{pipeline_summary['total_minutes']}`",
        "",
        "| img_id | image_label | role | runner_seconds | total_seconds | run_dir |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in pipeline_summary["runs"]:
        lines.append(
            f"| {row['img_id']} | {row['image_label']} | {row['selection_role']} | {row['runner_seconds']} | "
            f"{row['total_seconds']} | {row['run_dir']} |"
        )

    (batch_root / "overnight_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
