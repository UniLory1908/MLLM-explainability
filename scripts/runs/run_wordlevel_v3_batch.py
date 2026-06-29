from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_IMAGE_IDS = [30213, 331352, 426253, 393226, 555009]
DEFAULT_PROMPT_SET = PROJECT_ROOT / "prompt_sets" / "prompt_sensitivity_v3_wordlevel.json"
DEFAULT_RUN_PREFIX = "wordlevel_v3_rawscanpath"


def slugify(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value).strip().lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "value"


def load_image_registry() -> dict[str, dict]:
    path = PROJECT_ROOT / "configs" / "image_registry.json"
    return json.loads(path.read_text(encoding="utf-8")).get("images", {})


def load_coco_categories() -> list[str]:
    annotation_path = PROJECT_ROOT / "data" / "annotations" / "instances_val2017.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    return [str(cat["name"]) for cat in payload.get("categories", [])]


def build_mcq_options(obj_main: str, image_id: int, prompt_id: str, seed: int) -> str:
    categories = load_coco_categories()
    distractor_pool = [name for name in categories if name.lower() != obj_main.lower()]
    rng = random.Random(f"{seed}:{image_id}:{prompt_id}:{obj_main}")
    distractors = rng.sample(distractor_pool, 3)
    options = distractors + [obj_main]
    rng.shuffle(options)
    letters = ["A", "B", "C", "D"]
    return " ".join(f"{letter}) {option}" for letter, option in zip(letters, options))


def materialize_prompts(prompt_set_path: Path, image_id: int, obj_main: str, out_dir: Path, mcq_seed: int) -> Path:
    prompt_set = json.loads(prompt_set_path.read_text(encoding="utf-8"))
    prompts = []
    for prompt in prompt_set.get("prompts", []):
        prompt_id = str(prompt["id"])
        text = str(prompt["prompt"]).replace("{obj_main}", obj_main)
        system_prompt = str(prompt.get("system_prompt", "")).replace("{obj_main}", obj_main)
        if "{mcq_options}" in text:
            mcq_options = build_mcq_options(obj_main, image_id, prompt_id, mcq_seed)
            text = text.replace("{mcq_options}", mcq_options)
        prompt_row = {
            "id": prompt["id"],
            "label": prompt["label"],
            "prompt": text,
        }
        if system_prompt:
            prompt_row["system_prompt"] = system_prompt
        for key in ("do_sample", "temperature", "top_p"):
            if key in prompt:
                prompt_row[key] = prompt[key]
        prompts.append(prompt_row)

    payload = {
        "run_name": prompt_set.get("run_name", "prompt_sensitivity_v3_wordlevel"),
        "image_id": image_id,
        "obj_main": obj_main,
        "mcq_seed": mcq_seed,
        "prompts": prompts,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"prompts_{image_id}_{slugify(obj_main)}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run word-level prompt set v3 on the selected COCO images.",
    )
    parser.add_argument("--prompt-set", default=str(DEFAULT_PROMPT_SET))
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "float32", "bfloat16"], default="auto")
    parser.add_argument("--all-layers", action="store_true", help="Save TAM heatmaps for all layers.")
    parser.add_argument("--layers", help="Optional comma-separated layer list passed to the prompt sweep runner.")
    parser.add_argument("--mcq-seed", type=int, default=20260515, help="Seed used when materializing MCQ distractors.")
    parser.add_argument("--image-id", type=int, action="append", help="Repeat to run a subset of images.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching Qwen/TAM.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prompt_set_path = Path(args.prompt_set).resolve()
    image_ids = args.image_id or DEFAULT_IMAGE_IDS
    registry = load_image_registry()
    materialized_dir = PROJECT_ROOT / "outputs" / "_materialized_prompt_sets" / args.run_prefix

    for image_id in image_ids:
        info = registry.get(str(image_id), {})
        label = info.get("label") or str(image_id)
        obj_main = info.get("obj_main")
        if not obj_main:
            raise ValueError(f"Missing obj_main for image {image_id} in configs/image_registry.json")

        prompts_file = materialize_prompts(prompt_set_path, image_id, str(obj_main), materialized_dir, args.mcq_seed)
        run_name = f"{args.run_prefix}_{label}"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "runs" / "run_qwen_tam_prompt_sweep.py"),
            "--img-id",
            str(image_id),
            "--image-label",
            str(label),
            "--run-name",
            run_name,
            "--prompts-file",
            str(prompts_file),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--device",
            args.device,
            "--torch-dtype",
            args.torch_dtype,
        ]
        if args.layers:
            cmd.extend(["--layers", args.layers])
        if not args.all_layers:
            cmd.append("--final-layer-only")

        print("\n" + " ".join(f'"{part}"' if " " in part else part for part in cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
