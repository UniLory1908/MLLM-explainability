from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1] if SCRIPT_DIR.name == "runs" else SCRIPT_DIR
LL_TAM_DIR = PROJECT_ROOT / "external" / "tam-logit-lenses" / "ll_tam"
sys.path.insert(0, str(LL_TAM_DIR))

# Funzione demo del repository TAM per Qwen2-VL.
from demo import tam_demo_for_qwen2_vl  # noqa: E402

# Questo file non sostituisce il run della fase 0.
# Serve per salvare una vista estesa layer-by-layer e step-by-step.

IMG_ID = 331352
PROMPT_DEFAULT = "What is the main object in the image? Answer with one word only."
PROMPT_OVERRIDES = {
    403385: "What bathroom object is most visible in the image? Answer with one word only.",
}

# Se True usa tutti i layer disponibili.
RUN_ALL_LAYERS = True

# Se RUN_ALL_LAYERS = False, permette di selezionare un subset di layer.
LAYERS_TO_RUN = None

# Numero di colonne usato nelle griglie finali.
GRID_COLS = 8


def build_output_dirs(img_id: int) -> tuple[Path, Path, Path, Path]:
    # Salva il run in una cartella separata dai risultati della fase 0.
    run_root = PROJECT_ROOT / "outputs" / "intermediate_tam" / str(img_id)
    vis_dir = run_root / "vis_results"
    grids_dir = run_root / "token_grids"
    word_views_dir = run_root / "word_views"
    run_root.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    grids_dir.mkdir(parents=True, exist_ok=True)
    word_views_dir.mkdir(parents=True, exist_ok=True)
    return run_root, vis_dir, grids_dir, word_views_dir


def get_image_info(img_id: int) -> dict:
    # Recupera il file immagine corretto a partire da img_id.
    annotation_file = PROJECT_ROOT / "data" / "annotations" / "instances_val2017.json"
    coco = COCO(str(annotation_file))
    image_info = coco.loadImgs([img_id])[0]
    image_path = PROJECT_ROOT / "data" / "val2017" / image_info["file_name"]
    return {
        "img_id": img_id,
        "file_name": image_info["file_name"],
        "image_path": str(image_path),
    }


def parse_step_folders(grids_dir: Path, image_stem: str) -> list[dict]:
    # Legge le cartelle gia' create da TAM per i singoli token.
    token_root = grids_dir / image_stem
    step_rows = []
    if not token_root.exists():
        return step_rows

    for step_dir in sorted(token_root.iterdir()):
        if not step_dir.is_dir() or not step_dir.name.startswith("step_"):
            continue
        parts = step_dir.name.split("_", 2)
        if len(parts) < 3:
            continue
        step_rows.append({
            "step_idx": int(parts[1]),
            "token_label": parts[2],
            "folder_name": step_dir.name,
        })
    return step_rows


def build_token_groups(step_rows: list[dict]) -> list[dict]:
    # Costruisce una vista parola-level con una euristica semplice sui token consecutivi.
    groups = []
    current = []

    def flush_current() -> None:
        nonlocal current
        if not current:
            return
        combined_label = "".join(row["token_label"] for row in current)
        groups.append({
            "label": combined_label,
            "step_indices": [row["step_idx"] for row in current],
            "source_tokens": [row["token_label"] for row in current],
        })
        current = []

    for row in step_rows:
        token_label = row["token_label"]
        is_special = token_label.startswith("im_") or token_label in {"bos", "eos"}
        is_alpha = token_label.isalpha()

        if is_special:
            flush_current()
            continue

        if not current:
            current = [row]
            continue

        if is_alpha and current[-1]["token_label"].isalpha():
            current.append(row)
        else:
            flush_current()
            current = [row]

    flush_current()
    return groups


def combine_images_max(image_paths: list[Path]) -> Image.Image | None:
    # Combina piu' heatmap con una regola max pixelwise.
    images = []
    for image_path in image_paths:
        if image_path.exists():
            images.append(Image.open(image_path).convert("RGB"))

    if not images:
        return None

    width = max(img.width for img in images)
    height = max(img.height for img in images)
    arrays = []
    for img in images:
        if img.size != (width, height):
            img = img.resize((width, height), Image.BILINEAR)
        arrays.append(np.array(img, dtype=np.uint8))

    combined = np.maximum.reduce(arrays)
    return Image.fromarray(combined)


def save_layer_grid(layer_images: list[tuple[str, Path]], grid_path: Path, title: str) -> None:
    # Salva una griglia finale per la vista combinata word-level.
    valid_images = []
    for layer_name, image_path in layer_images:
        if image_path.exists():
            valid_images.append((layer_name, Image.open(image_path).convert("RGB")))

    if not valid_images:
        return

    cols = GRID_COLS
    rows = int(np.ceil(len(valid_images) / cols))
    tile_w = max(img.width for _, img in valid_images)
    tile_h = max(img.height for _, img in valid_images)
    pad = 8
    label_h = 22
    header_h = 30

    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = rows * (tile_h + label_h) + (rows + 1) * pad + header_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((pad, 7), title, fill=(255, 255, 255), font=font)

    for idx, (layer_name, img) in enumerate(valid_images):
        row = idx // cols
        col = idx % cols
        x0 = pad + col * (tile_w + pad)
        y0 = header_h + pad + row * (tile_h + label_h + pad)
        if img.size != (tile_w, tile_h):
            img = img.resize((tile_w, tile_h), Image.BILINEAR)
        draw.text((x0 + 4, y0 + 4), layer_name, fill=(255, 255, 255), font=font)
        canvas.paste(img, (x0, y0 + label_h))

    grid_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(grid_path)


def build_word_level_views(vis_dir: Path, grids_dir: Path, word_views_dir: Path, image_stem: str) -> list[dict]:
    # Costruisce una vista word-level a partire dalle heatmap token-level gia' salvate.
    step_rows = parse_step_folders(grids_dir, image_stem)
    token_groups = build_token_groups(step_rows)
    image_vis_root = vis_dir / image_stem
    word_rows = []

    for group_idx, group in enumerate(token_groups):
        group_name = f"group_{group_idx:02d}_{group['label']}"
        group_dir = word_views_dir / image_stem / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        layer_outputs = []

        for layer_dir in sorted(image_vis_root.iterdir()):
            if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
                continue
            step_paths = [layer_dir / f"step_{step_idx:04d}.jpg" for step_idx in group["step_indices"]]
            combined_image = combine_images_max(step_paths)
            if combined_image is None:
                continue
            out_path = group_dir / f"{layer_dir.name}.jpg"
            combined_image.save(out_path)
            layer_outputs.append((layer_dir.name, out_path))

        title = f"word view: {group['label']} | from: {' + '.join(group['source_tokens'])}"
        save_layer_grid(layer_outputs, group_dir / "grid_layers.jpg", title)
        word_rows.append({
            "group_name": group_name,
            "label": group["label"],
            "source_tokens": group["source_tokens"],
            "step_indices": group["step_indices"],
            "grid_path": str(group_dir / "grid_layers.jpg"),
        })

    return word_rows


def main() -> None:
    # Recupera immagine e prompt del run.
    image_info = get_image_info(IMG_ID)
    image_path = image_info["image_path"]
    image_stem = Path(image_path).stem
    prompt_text = PROMPT_OVERRIDES.get(IMG_ID, PROMPT_DEFAULT)
    run_root, vis_dir, grids_dir, word_views_dir = build_output_dirs(IMG_ID)

    # Stampa un riepilogo minimo del run.
    print(f"img_id: {IMG_ID}")
    print(f"image: {image_path}")
    print(f"prompt: {prompt_text}")
    print("run all layers:", RUN_ALL_LAYERS)
    if not RUN_ALL_LAYERS:
        print("layers:", LAYERS_TO_RUN)

    # Usa direttamente la funzione demo del repository TAM.
    tam_demo_for_qwen2_vl(
        image_path=image_path,
        prompt_text=prompt_text,
        save_dir=str(vis_dir),
        grids_dir=str(grids_dir),
        all_layers=RUN_ALL_LAYERS,
        layers=LAYERS_TO_RUN,
        grid_cols=GRID_COLS,
    )

    # Dopo il run base costruisce anche una vista word-level.
    word_rows = build_word_level_views(
        vis_dir=vis_dir,
        grids_dir=grids_dir,
        word_views_dir=word_views_dir,
        image_stem=image_stem,
    )

    # Salva un file json con le scelte principali del run.
    metadata = {
        "img_id": IMG_ID,
        "image_path": image_path,
        "prompt": prompt_text,
        "run_all_layers": RUN_ALL_LAYERS,
        "layers_to_run": LAYERS_TO_RUN,
        "vis_dir": str(vis_dir),
        "grids_dir": str(grids_dir),
        "word_views_dir": str(word_views_dir),
        "word_groups": word_rows,
    }
    (run_root / "run_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(run_root / "run_info.json")
    print(vis_dir)
    print(grids_dir)
    print(word_views_dir)


if __name__ == "__main__":
    main()
