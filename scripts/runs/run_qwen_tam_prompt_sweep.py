from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1] if SCRIPT_DIR.name == "runs" else SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))
LL_TAM_DIR = PROJECT_ROOT / "external" / "tam-logit-lenses" / "ll_tam"
sys.path.insert(0, str(LL_TAM_DIR))

from demo import (  # noqa: E402
    _build_logitlens_logits,
    _build_per_token_grids,
    _decode_tokens,
    _get_final_norm,
    _num_rounds,
    _safe_folder_name,
)
from qwen_utils import process_vision_info  # noqa: E402
from tam import TAM  # noqa: E402
from scripts.common.prompt_word_utils import (  # noqa: E402
    build_word_groups,
    estimate_heatmap_rgb,
    load_saliency_map,
    resize_saliency_map,
    saliency_from_array,
)

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
# Tengo qui un set minimo di fallback.
# Serve solo per avere una prova pronta anche senza file JSON esterni.
DEFAULT_PROMPTS = [
    {
        "id": "baseline",
        "label": "baseline",
        "prompt": "Describe the image.",
    },
    {
        "id": "visible_only",
        "label": "visible_only",
        "prompt": "Describe ONLY what is visible in the image.",
    },
]


SCANPATH_THRESHOLD_PERCENTILE = 95.0
SCANPATH_MIN_HOTSPOT_AREA = 64
SCANPATH_TOPK_HOTSPOTS = 3
SCANPATH_MAX_LINK_DISTANCE_RATIO = 0.18


def slugify(value: str, max_len: int = 80) -> str:
    # Mi serve un nome cartella stabile e leggibile a partire da label o prompt.
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return (normalized or "prompt")[:max_len]


def step_artifact_stem(token_label: str, step_idx: int) -> str:
    # Riuso la convenzione TAM gia' presente: step + indice + token leggibile.
    # Cosi' i file non restano solo numerici.
    return _safe_folder_name(token_label or "tok", step_idx)


def load_image_registry() -> dict[str, dict]:
    registry_path = PROJECT_ROOT / "configs" / "image_registry.json"
    if not registry_path.exists():
        return {}
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    return payload.get("images", {})


def resolve_image_label(img_id: int | None, image_path: str, explicit_label: str | None) -> str:
    if explicit_label:
        return slugify(explicit_label)
    registry = load_image_registry()
    if img_id is not None:
        entry = registry.get(str(img_id))
        if entry and entry.get("label"):
            return slugify(str(entry["label"]))
    return slugify(Path(image_path).stem)


def parse_layers(raw_value: str | None) -> list[int] | None:
    # Le layer list arrivano da CLI come stringa semplice tipo "0,4,8".
    # Qui le porto in una lista di interi senza introdurre altre dipendenze.
    if not raw_value:
        return None
    layers = []
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        layers.append(int(chunk))
    return layers or None


def build_messages(image_path: str, prompt_text: str) -> list[dict]:
    # Per ogni prompt ricostruisco da zero il messaggio multimodale.
    # Questo tiene isolati i run e rende il confronto cross-prompt pulito.
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text},
        ],
    }]


def resolve_image(args: argparse.Namespace) -> tuple[str, int | None]:
    # Supporto due modalita' semplici:
    # - path immagine diretto
    # - img_id COCO risolto dentro data/
    if args.image_path:
        return str(Path(args.image_path).resolve()), None

    if args.img_id is None:
        raise ValueError("Specify either --image-path or --img-id.")

    annotation_file = PROJECT_ROOT / "data" / "annotations" / "instances_val2017.json"
    coco = COCO(str(annotation_file))
    image_info = coco.loadImgs([args.img_id])[0]
    image_path = PROJECT_ROOT / "data" / "val2017" / image_info["file_name"]
    return str(image_path), int(args.img_id)


def normalize_prompt_entry(entry: object, index: int) -> dict:
    # Accetto sia stringhe nude sia oggetti piu' ricchi dal JSON.
    # L'idea e' mantenere il formato facile da estendere ma leggero da usare.
    if isinstance(entry, str):
        prompt_text = entry.strip()
        prompt_id = f"prompt_{index:02d}"
        label = slugify(prompt_text)
    elif isinstance(entry, dict):
        prompt_text = str(entry["prompt"]).strip()
        prompt_id = str(entry.get("id") or f"prompt_{index:02d}")
        label = str(entry.get("label") or slugify(prompt_text))
    else:
        raise ValueError(f"Unsupported prompt entry at index {index}: {entry!r}")

    if not prompt_text:
        raise ValueError(f"Empty prompt at index {index}.")

    return {
        "id": prompt_id,
        "label": label,
        "prompt": prompt_text,
    }


def load_prompts(args: argparse.Namespace) -> tuple[list[dict], str | None]:
    # Ordine di priorita':
    # 1. prompt passati da CLI
    # 2. file JSON
    # 3. fallback minimo locale
    if args.prompt:
        prompts = [normalize_prompt_entry(prompt, idx) for idx, prompt in enumerate(args.prompt)]
        return prompts, None

    if args.prompts_file:
        payload = json.loads(Path(args.prompts_file).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            run_name = payload.get("run_name")
            entries = payload.get("prompts", [])
        else:
            run_name = None
            entries = payload
        prompts = [normalize_prompt_entry(entry, idx) for idx, entry in enumerate(entries)]
        return prompts, run_name

    prompts = [normalize_prompt_entry(entry, idx) for idx, entry in enumerate(DEFAULT_PROMPTS)]
    return prompts, "default_prompt_sweep"


def save_summary_csv(summary_rows: list[dict], output_path: Path) -> None:
    # Questo CSV e' il punto di accesso piu' comodo per una lettura rapida del run.
    # I dettagli piu' fini restano nei metadata per-prompt.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt_index",
        "prompt_id",
        "prompt_label",
        "prompt_text",
        "response_text",
        "num_rounds",
        "elapsed_seconds",
        "prompt_dir",
        "metadata_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def _extract_hotspots_from_saliency(
    saliency_map: np.ndarray,
    threshold_percentile: float,
    min_area: int,
    top_k: int,
) -> list[dict]:
    # La scanpath va calcolata sulla mappa TAM scalare, non sulla JPG colorata.
    saliency = saliency_from_array(saliency_map)
    positive = saliency[saliency > 0]
    if positive.size == 0:
        return []

    threshold = float(np.percentile(positive, threshold_percentile))
    mask = saliency >= threshold
    if not np.any(mask):
        return []

    height, width = saliency.shape
    visited = np.zeros_like(mask, dtype=bool)
    hotspots: list[dict] = []

    for y0 in range(height):
        for x0 in range(width):
            if not mask[y0, x0] or visited[y0, x0]:
                continue

            queue = [(y0, x0)]
            visited[y0, x0] = True
            component_pixels: list[tuple[int, int]] = []

            while queue:
                y, x = queue.pop()
                component_pixels.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yn = y + dy
                    xn = x + dx
                    if yn < 0 or yn >= height or xn < 0 or xn >= width:
                        continue
                    if visited[yn, xn] or not mask[yn, xn]:
                        continue
                    visited[yn, xn] = True
                    queue.append((yn, xn))

            area = len(component_pixels)
            if area < min_area:
                continue

            ys = np.array([p[0] for p in component_pixels], dtype=np.int32)
            xs = np.array([p[1] for p in component_pixels], dtype=np.int32)
            weights = saliency[ys, xs]
            strength = float(weights.sum())
            weight_sum = float(weights.sum())
            if weight_sum <= 0.0:
                continue

            cx = float((xs * weights).sum() / weight_sum)
            cy = float((ys * weights).sum() / weight_sum)
            hotspots.append({
                "centroid_x": round(cx, 3),
                "centroid_y": round(cy, 3),
                "strength": round(strength, 3),
                "mean_value": round(float(weights.mean()), 3),
                "peak_value": round(float(weights.max()), 3),
                "area": int(area),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "threshold_value": round(threshold, 3),
            })

    hotspots.sort(key=lambda h: (h["strength"], h["area"]), reverse=True)
    return hotspots[:top_k]


def _extract_hotspots_from_heatmap(
    heatmap_rgb: np.ndarray,
    threshold_percentile: float,
    min_area: int,
    top_k: int,
) -> list[dict]:
    return _extract_hotspots_from_saliency(
        saliency_map=heatmap_rgb,
        threshold_percentile=threshold_percentile,
        min_area=min_area,
        top_k=top_k,
    )


def save_raw_tam_map(
    raw_map: np.ndarray,
    raw_map_path: Path,
    reference_heatmap_path: Path,
    fallback_image_path: str,
    fallback_size: tuple[int, int] | None = None,
    fallback_width: int = 500,
) -> tuple[str, list[int], list[int]]:
    # TAM restituisce la mappa sulla griglia dei visual token.
    # La salvo anche ridimensionata come la JPG, cosi' i centroidi hanno coordinate coerenti con il viewer.
    if raw_map is None:
        raise ValueError(f"TAM did not return a raw map for {reference_heatmap_path}")
    scalar = saliency_from_array(raw_map)
    original_shape = [int(scalar.shape[0]), int(scalar.shape[1])]
    from PIL import Image

    if reference_heatmap_path.exists():
        width, height = Image.open(reference_heatmap_path).size
        scalar = resize_saliency_map(scalar, (width, height))
    elif fallback_size is not None:
        scalar = resize_saliency_map(scalar, fallback_size)
    elif fallback_image_path and Path(fallback_image_path).exists():
        image = Image.open(fallback_image_path)
        width = int(fallback_width)
        height = max(1, int(float(image.height) / float(image.width) * width))
        scalar = resize_saliency_map(scalar, (width, height))
    resized_shape = [int(scalar.shape[0]), int(scalar.shape[1])]
    raw_map_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(raw_map_path, scalar.astype(np.float32))
    return str(raw_map_path), original_shape, resized_shape


def build_step_hotspots_and_scanpath(
    step_records: list[dict],
    image_path: str,
    threshold_percentile: float,
    min_area: int,
    top_k: int,
    max_link_distance_ratio: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    # Estrae hotspot per step e collega i centroidi nel tempo per ottenere uno scanpath.
    enriched_steps: list[dict] = []
    tracks: list[dict] = []
    track_states: dict[int, dict] = {}
    next_track_id = 0

    base_hw = None
    diag = None
    for step in step_records:
        raw_map_path = step.get("raw_map_path")
        hotspot_source = "legacy_overlay_reconstruction"
        if raw_map_path and Path(str(raw_map_path)).exists():
            saliency = load_saliency_map(raw_map_path)
            hotspot_source = "raw_tam_map"
        else:
            heatmap_path = step.get("heatmap_path")
            if not heatmap_path or not Path(str(heatmap_path)).exists():
                enriched_steps.append({**step, "hotspots": [], "dominant_hotspot": None})
                continue
            saliency = estimate_heatmap_rgb(heatmap_path, image_path)
        if base_hw is None:
            h, w = saliency.shape[:2]
            base_hw = (h, w)
            diag = float((h ** 2 + w ** 2) ** 0.5)

        hotspots = _extract_hotspots_from_saliency(
            saliency_map=saliency,
            threshold_percentile=threshold_percentile,
            min_area=min_area,
            top_k=top_k,
        )

        step_idx = int(step.get("step_idx", len(enriched_steps)))
        dominant_hotspot = hotspots[0] if hotspots else None
        enriched_steps.append({
            **step,
            "hotspots": hotspots,
            "dominant_hotspot": dominant_hotspot,
            "hotspot_source": hotspot_source,
        })

        if not hotspots:
            continue

        link_radius = (diag or 1.0) * max_link_distance_ratio
        used_hotspots = set()
        for track_id, state in list(track_states.items()):
            best_idx = None
            best_dist = None
            for idx, hotspot in enumerate(hotspots):
                if idx in used_hotspots:
                    continue
                dx = float(hotspot["centroid_x"]) - float(state["x"])
                dy = float(hotspot["centroid_y"]) - float(state["y"])
                dist = float((dx ** 2 + dy ** 2) ** 0.5)
                if dist > link_radius:
                    continue
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is None:
                continue
            hotspot = hotspots[best_idx]
            used_hotspots.add(best_idx)
            state["x"] = float(hotspot["centroid_x"])
            state["y"] = float(hotspot["centroid_y"])
            state["last_step"] = step_idx
            state["points"].append({
                "step_idx": step_idx,
                "centroid_x": hotspot["centroid_x"],
                "centroid_y": hotspot["centroid_y"],
                "strength": hotspot["strength"],
                "area": hotspot["area"],
            })

        for idx, hotspot in enumerate(hotspots):
            if idx in used_hotspots:
                continue
            track_states[next_track_id] = {
                "x": float(hotspot["centroid_x"]),
                "y": float(hotspot["centroid_y"]),
                "last_step": step_idx,
                "points": [{
                    "step_idx": step_idx,
                    "centroid_x": hotspot["centroid_x"],
                    "centroid_y": hotspot["centroid_y"],
                    "strength": hotspot["strength"],
                    "area": hotspot["area"],
                }],
            }
            next_track_id += 1

    for track_id, state in track_states.items():
        points = state["points"]
        tracks.append({
            "track_id": int(track_id),
            "num_points": len(points),
            "start_step": int(points[0]["step_idx"]),
            "end_step": int(points[-1]["step_idx"]),
            "mean_strength": round(float(np.mean([p["strength"] for p in points])), 3),
            "points": points,
        })

    tracks.sort(key=lambda t: (t["num_points"], t["mean_strength"]), reverse=True)
    dominant_scanpath = tracks[0]["points"] if tracks else []
    return enriched_steps, tracks, dominant_scanpath


def run_single_prompt(
    model,
    processor,
    image_path: str,
    prompt_entry: dict,
    prompt_index: int,
    run_root: Path,
    all_layers: bool,
    requested_layers: list[int] | None,
    max_new_tokens: int,
    grid_cols: int,
    img_id: int | None,
    scanpath_threshold_percentile: float,
    scanpath_min_hotspot_area: int,
    scanpath_topk_hotspots: int,
    scanpath_max_link_distance_ratio: float,
) -> dict:
    # Qui faccio il run di un solo prompt.
    # Il modello resta caricato fuori da questa funzione, mentre tutto il resto
    # viene ricostruito fresh per tenere i prompt indipendenti.
    prompt_slug = slugify(prompt_entry["label"])
    prompt_dir = run_root / f"{prompt_index:02d}_{prompt_slug}"
    vis_dir = prompt_dir / "vis_results"
    grids_dir = prompt_dir / "token_grids"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    grids_dir.mkdir(parents=True, exist_ok=True)

    messages = build_messages(image_path, prompt_entry["prompt"])
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Misuro il tempo per avere un confronto semplice anche lato costo computazionale.
    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    elapsed_seconds = round(time.time() - start, 2)

    generated_ids = outputs.sequences
    prompt_len = inputs["input_ids"].shape[1]
    n_layers = len(outputs.hidden_states[0])
    num_rounds, _ = _num_rounds(outputs, prompt_len)
    token_labels = _decode_tokens(outputs, prompt_len, processor)
    token_pieces = [
        str(processor.tokenizer.convert_ids_to_tokens(int(token_id)))
        for token_id in generated_ids[0][prompt_len:].tolist()
    ]
    response_text = processor.tokenizer.decode(
        generated_ids[0][prompt_len:].tolist(),
        skip_special_tokens=True,
    ).strip()

    # Questi identificatori sono quelli gia' usati nel flusso TAM per Qwen2-VL.
    # Non li cambio qui: il prompt sweep deve riusare lo stesso comportamento del runner.
    special_ids = {
        "img_id": [151652, 151653],
        "prompt_id": [151653, [151645, 198, 151644, 77091]],
        "answer_id": [[198, 151644, 77091, 198], -1],
    }
    vision_shape = (
        int(inputs["image_grid_thw"][0, 1]) // 2,
        int(inputs["image_grid_thw"][0, 2]) // 2,
    )
    stem = Path(image_path).stem
    image_dir = vis_dir / stem
    raw_maps_dir = prompt_dir / "raw_maps" / stem
    image_dir.mkdir(parents=True, exist_ok=True)
    raw_maps_dir.mkdir(parents=True, exist_ok=True)

    # Se non passo un subset di layer, considero tutti quelli disponibili.
    run_layers = requested_layers if requested_layers is not None else list(range(n_layers))
    step_records: list[dict] = []

    print(
        f"[prompt {prompt_index:02d}] '{prompt_entry['label']}'"
        f" -> '{response_text}' | steps={num_rounds} | secs={elapsed_seconds}"
    )

    if not all_layers:
        # Modalita' piu' leggera:
        # salvo una heatmap per step usando l'ultimo layer, utile per sweep veloci.
        logits = _build_logitlens_logits(outputs, model, n_layers - 1, n_layers)
        raw_map_records = []
        display_size: tuple[int, int] | None = None
        for step_idx in range(num_rounds):
            token_label = token_labels[step_idx] if step_idx < len(token_labels) else "tok"
            token_piece = token_pieces[step_idx] if step_idx < len(token_pieces) else token_label
            step_label = step_artifact_stem(token_label, step_idx)
            save_path = image_dir / f"{step_label}.jpg"
            raw_map = TAM(
                generated_ids[0].cpu().tolist(),
                vision_shape,
                logits,
                special_ids,
                image_inputs,
                processor,
                str(save_path),
                step_idx,
                raw_map_records,
                False,
            )
            raw_map_path, raw_map_shape, resized_raw_map_shape = save_raw_tam_map(
                raw_map,
                raw_maps_dir / f"{step_label}.npy",
                save_path,
                image_path,
                display_size,
            )
            display_size = (resized_raw_map_shape[1], resized_raw_map_shape[0])
            step_records.append({
                "step_idx": step_idx,
                "token_label": token_label,
                "token_piece": token_piece,
                "step_label": step_label,
                "heatmap_path": str(save_path),
                "raw_map_path": raw_map_path,
                "raw_map_shape": raw_map_shape,
                "resized_raw_map_shape": resized_raw_map_shape,
            })
    else:
        # Modalita' piu' ricca:
        # salvo heatmap per ogni step e per ogni layer richiesto, poi costruisco le grid.
        layer_step_paths: dict[int, dict[int, Path]] = {}
        layer_raw_map_paths: dict[int, dict[int, str]] = {}
        layer_raw_map_shapes: dict[int, dict[int, list[int]]] = {}
        layer_resized_raw_map_shapes: dict[int, dict[int, list[int]]] = {}
        for layer_idx in run_layers:
            layer_dir = image_dir / f"layer_{layer_idx:03d}"
            layer_raw_dir = raw_maps_dir / f"layer_{layer_idx:03d}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            layer_raw_dir.mkdir(parents=True, exist_ok=True)
            logits = _build_logitlens_logits(outputs, model, layer_idx, n_layers)
            img_scores_list = []
            display_size: tuple[int, int] | None = None
            layer_step_paths[layer_idx] = {}
            layer_raw_map_paths[layer_idx] = {}
            layer_raw_map_shapes[layer_idx] = {}
            layer_resized_raw_map_shapes[layer_idx] = {}
            for step_idx in range(num_rounds):
                token_label = token_labels[step_idx] if step_idx < len(token_labels) else "tok"
                step_label = step_artifact_stem(token_label, step_idx)
                save_path = layer_dir / f"{step_label}.jpg"
                raw_map = TAM(
                    generated_ids[0].cpu().tolist(),
                    vision_shape,
                    logits,
                    special_ids,
                    image_inputs,
                    processor,
                    str(save_path),
                    step_idx,
                    img_scores_list,
                    False,
                )
                layer_step_paths[layer_idx][step_idx] = save_path
                raw_map_path, raw_map_shape, resized_raw_map_shape = save_raw_tam_map(
                    raw_map,
                    layer_raw_dir / f"{step_label}.npy",
                    save_path,
                    image_path,
                    display_size,
                )
                display_size = (resized_raw_map_shape[1], resized_raw_map_shape[0])
                layer_raw_map_paths[layer_idx][step_idx] = raw_map_path
                layer_raw_map_shapes[layer_idx][step_idx] = raw_map_shape
                layer_resized_raw_map_shapes[layer_idx][step_idx] = resized_raw_map_shape

        _build_per_token_grids(
            stem,
            token_labels,
            num_rounds,
            run_layers,
            layer_step_paths,
            grids_dir,
            cols=grid_cols,
        )

        for step_idx in range(num_rounds):
            token_label = token_labels[step_idx] if step_idx < len(token_labels) else "tok"
            token_piece = token_pieces[step_idx] if step_idx < len(token_pieces) else token_label
            step_label = step_artifact_stem(token_label, step_idx)
            scanpath_layer = max(run_layers)
            step_records.append({
                "step_idx": step_idx,
                "token_label": token_label,
                "token_piece": token_piece,
                "step_label": step_label,
                "layer_heatmaps": {
                    str(layer_idx): str(layer_step_paths[layer_idx][step_idx])
                    for layer_idx in run_layers
                },
                "layer_raw_maps": {
                    str(layer_idx): str(layer_raw_map_paths[layer_idx][step_idx])
                    for layer_idx in run_layers
                },
                "token_grid_dir": str(
                    grids_dir
                    / stem
                    / step_label
                ),
                "heatmap_path": str(layer_step_paths[scanpath_layer][step_idx]),
                "raw_map_path": str(layer_raw_map_paths[scanpath_layer][step_idx]),
                "raw_map_shape": layer_raw_map_shapes[scanpath_layer][step_idx],
                "resized_raw_map_shape": layer_resized_raw_map_shapes[scanpath_layer][step_idx],
            })

    step_records, scanpath_tracks, dominant_scanpath = build_step_hotspots_and_scanpath(
        step_records=step_records,
        image_path=image_path,
        threshold_percentile=scanpath_threshold_percentile,
        min_area=scanpath_min_hotspot_area,
        top_k=scanpath_topk_hotspots,
        max_link_distance_ratio=scanpath_max_link_distance_ratio,
    )

    # Salvo tutto cio' che serve per confronti successivi senza dover rieseguire il modello.
    # Qui tengo sia la parte testuale sia i path agli artifact visivi.
    metadata = {
        "prompt_index": prompt_index,
        "prompt_id": prompt_entry["id"],
        "prompt_label": prompt_entry["label"],
        "prompt_text": prompt_entry["prompt"],
        "image_path": image_path,
        "image_stem": stem,
        "img_id": img_id,
        "model_name": MODEL_NAME,
        "max_new_tokens": max_new_tokens,
        "all_layers": all_layers,
        "layers": run_layers,
        "response_text": response_text,
        "generated_token_ids": generated_ids[0][prompt_len:].tolist(),
        "generated_token_labels": token_labels[:num_rounds],
        "generated_token_pieces": token_pieces[:num_rounds],
        "num_rounds": num_rounds,
        "elapsed_seconds": elapsed_seconds,
        "vis_dir": str(vis_dir),
        "grids_dir": str(grids_dir) if all_layers else "",
        "step_records": step_records,
        "scanpath": {
            "source": "raw_tam_map",
            "threshold_percentile": scanpath_threshold_percentile,
            "min_hotspot_area": scanpath_min_hotspot_area,
            "topk_hotspots_per_step": scanpath_topk_hotspots,
            "max_link_distance_ratio": scanpath_max_link_distance_ratio,
            "tracks": scanpath_tracks,
            "dominant_scanpath": dominant_scanpath,
        },
    }
    metadata["word_records"] = build_word_groups(step_records, token_pieces[:num_rounds])
    metadata["generated_word_labels"] = [word["word_label"] for word in metadata["word_records"]]
    metadata_path = prompt_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "prompt_index": prompt_index,
        "prompt_id": prompt_entry["id"],
        "prompt_label": prompt_entry["label"],
        "prompt_text": prompt_entry["prompt"],
        "response_text": response_text,
        "num_rounds": num_rounds,
        "elapsed_seconds": elapsed_seconds,
        "prompt_dir": str(prompt_dir),
        "metadata_path": str(metadata_path),
    }


def build_parser() -> argparse.ArgumentParser:
    # La CLI resta volutamente piccola:
    # abbastanza flessibile per gli esperimenti, ma senza trasformarla in un framework.
    parser = argparse.ArgumentParser(
        description="Run Qwen2-VL + TAM on the same image with multiple prompts.",
    )
    parser.add_argument("--img-id", type=int, help="COCO image id to resolve via data/annotations.")
    parser.add_argument("--image-path", help="Direct path to an image file.")
    parser.add_argument(
        "--prompt",
        action="append",
        help="Prompt text. Repeat the flag to sweep multiple prompts.",
    )
    parser.add_argument(
        "--prompts-file",
        help="JSON file with prompt entries. Supports either a list or {run_name, prompts}.",
    )
    parser.add_argument("--run-name", help="Optional custom run name.")
    parser.add_argument("--image-label", help="Short readable image label used in output folders and manifests.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--grid-cols", type=int, default=8)
    parser.add_argument("--layers", help="Comma-separated layer list, e.g. 0,4,8,12.")
    parser.add_argument(
        "--scanpath-threshold-percentile",
        type=float,
        default=SCANPATH_THRESHOLD_PERCENTILE,
        help="Percentile threshold used to detect hotspots from each TAM heatmap.",
    )
    parser.add_argument(
        "--scanpath-min-hotspot-area",
        type=int,
        default=SCANPATH_MIN_HOTSPOT_AREA,
        help="Minimum connected-component area (pixels) kept as hotspot.",
    )
    parser.add_argument(
        "--scanpath-topk-hotspots",
        type=int,
        default=SCANPATH_TOPK_HOTSPOTS,
        help="Maximum number of hotspots retained for each step.",
    )
    parser.add_argument(
        "--scanpath-max-link-distance-ratio",
        type=float,
        default=SCANPATH_MAX_LINK_DISTANCE_RATIO,
        help="Tracking radius as ratio of image diagonal for linking hotspots between steps.",
    )
    parser.add_argument(
        "--final-layer-only",
        action="store_true",
        help="Save only final-layer TAM heatmaps instead of all layers + token grids.",
    )
    return parser


def main() -> None:
    # Questo entrypoint e' separato dal run fase 0.
    # Serve per esperimenti controllati sul prompt mantenendo fissi immagine e modello.
    parser = build_parser()
    args = parser.parse_args()

    image_path, img_id = resolve_image(args)
    prompts, file_run_name = load_prompts(args)
    run_name = args.run_name or file_run_name or f"prompt_sweep_{Path(image_path).stem}"
    image_label = resolve_image_label(img_id, image_path, args.image_label)
    all_layers = not args.final_layer_only
    requested_layers = parse_layers(args.layers)

    # Ogni sweep finisce in una cartella dedicata.
    # Cosi' posso lanciare piu' esperimenti sulla stessa immagine senza sovrascrivere nulla.
    image_dir_name = f"{Path(image_path).stem}_{image_label}"
    run_root = PROJECT_ROOT / "outputs" / "prompt_sensitivity" / image_dir_name / slugify(run_name)
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"image: {image_path}")
    print(f"image label: {image_label}")
    print(f"run root: {run_root}")
    print(f"prompts: {len(prompts)}")
    print(f"all layers: {all_layers}")
    if requested_layers is not None:
        print(f"layers: {requested_layers}")
    print(f"scanpath threshold percentile: {args.scanpath_threshold_percentile}")
    print(f"scanpath min hotspot area: {args.scanpath_min_hotspot_area}")
    print(f"scanpath top-k hotspots: {args.scanpath_topk_hotspots}")
    print(f"scanpath max link distance ratio: {args.scanpath_max_link_distance_ratio}")

    # Carico modello e processor una sola volta.
    # Il guadagno principale dello sweep e' proprio evitare ricariche inutili tra prompt.
    print(f"Loading model: {MODEL_NAME}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    norm = _get_final_norm(model)
    if norm is None:
        print("[WARN] final norm not found")
    else:
        print(f"[OK] final norm: {type(norm).__name__}")

    # Eseguo i prompt in sequenza ma con stato conversazionale isolato.
    # Il confronto scientifico ha senso solo se ogni prompt parte pulito.
    summary_rows = []
    for prompt_index, prompt_entry in enumerate(prompts):
        summary_rows.append(
            run_single_prompt(
                model=model,
                processor=processor,
                image_path=image_path,
                prompt_entry=prompt_entry,
                prompt_index=prompt_index,
                run_root=run_root,
                all_layers=all_layers,
                requested_layers=requested_layers,
                max_new_tokens=args.max_new_tokens,
                grid_cols=args.grid_cols,
                img_id=img_id,
                scanpath_threshold_percentile=args.scanpath_threshold_percentile,
                scanpath_min_hotspot_area=args.scanpath_min_hotspot_area,
                scanpath_topk_hotspots=args.scanpath_topk_hotspots,
                scanpath_max_link_distance_ratio=args.scanpath_max_link_distance_ratio,
            )
        )

    save_summary_csv(summary_rows, run_root / "prompt_runs.csv")
    # Il manifest finale tiene insieme il run completo.
    # E' il file piu' comodo da dare in input agli script di analisi.
    manifest = {
        "run_name": run_name,
        "image_path": image_path,
        "image_stem": Path(image_path).stem,
        "image_label": image_label,
        "image_dir_name": image_dir_name,
        "img_id": img_id,
        "model_name": MODEL_NAME,
        "max_new_tokens": args.max_new_tokens,
        "all_layers": all_layers,
        "layers": requested_layers,
        "prompt_count": len(prompts),
        "summary_csv": str(run_root / "prompt_runs.csv"),
        "prompt_runs": summary_rows,
    }
    (run_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(run_root / "run_manifest.json")
    print(run_root / "prompt_runs.csv")


if __name__ == "__main__":
    main()
