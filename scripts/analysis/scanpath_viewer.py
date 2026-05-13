from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.prompt_word_utils import (  # noqa: E402
    build_word_groups,
    estimate_heatmap_rgb,
    load_saliency_map,
    saliency_from_array,
    saliency_to_uint8,
)


class ViewerInputError(ValueError):
    pass


class MetadataContractError(ValueError):
    pass


SCANPATH_VIEW_MODES = {"token_raw", "token_clean", "word", "word_fixations"}


def load_font(size: int = 18):
    # Font semplice di sistema. Se non c'e', PIL usa quello base.
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rgb_array(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def resize_rgb(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize(size, Image.BILINEAR), dtype=np.uint8)


def blend_on_original(
    heatmap_rgb: np.ndarray,
    original_image_path: str | Path | None,
    heatmap_alpha: float = 0.42,
) -> np.ndarray:
    # La vista presentabile usa l'immagine originale come base.
    # La heatmap resta visibile, ma non copre completamente la scena.
    if not original_image_path:
        return heatmap_rgb.astype(np.uint8)
    image_path = Path(original_image_path)
    if not image_path.exists():
        return heatmap_rgb.astype(np.uint8)
    height, width = heatmap_rgb.shape[:2]
    original = resize_rgb(image_path, (width, height)).astype(np.float32)
    heatmap = heatmap_rgb.astype(np.float32)
    return np.clip((1.0 - heatmap_alpha) * original + heatmap_alpha * heatmap, 0, 255).astype(np.uint8)


def heatmap_for_step(step: dict, original_image_path: str | Path | None, recover_from_overlay: bool) -> np.ndarray:
    raw_map_path = step.get("raw_map_path")
    if raw_map_path and Path(str(raw_map_path)).exists():
        return saliency_to_rgb(load_saliency_map(raw_map_path))

    heatmap_path = step.get("heatmap_path")
    if not heatmap_path:
        raise ViewerInputError("step senza heatmap_path")
    if recover_from_overlay and original_image_path:
        return estimate_heatmap_rgb(heatmap_path, original_image_path).astype(np.uint8)
    return rgb_array(heatmap_path)


def safe_name(value: str, default: str = "tok", max_len: int = 60) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9._-]+', "_", str(value)).strip("._-")
    return (cleaned or default)[:max_len]


def saliency_to_rgb(saliency: np.ndarray) -> np.ndarray:
    normalized = saliency_to_uint8(saliency)
    rgba = plt.get_cmap("jet")(normalized.astype(np.float32) / 255.0)
    return np.clip(rgba[:, :, :3] * 255.0, 0, 255).astype(np.uint8)


def extract_hotspots_from_heatmap(
    heatmap_rgb: np.ndarray,
    threshold_percentile: float = 95.0,
    min_area: int = 64,
    top_k: int = 3,
) -> list[dict]:
    # Versione locale e semplice dell'estrazione hotspot.
    # Nei run nuovi lavora su raw TAM map; nei run legacy accetta ancora RGB.
    saliency = saliency_from_array(heatmap_rgb)
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
            pixels: list[tuple[int, int]] = []

            while queue:
                y, x = queue.pop()
                pixels.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yn = y + dy
                    xn = x + dx
                    if yn < 0 or yn >= height or xn < 0 or xn >= width:
                        continue
                    if visited[yn, xn] or not mask[yn, xn]:
                        continue
                    visited[yn, xn] = True
                    queue.append((yn, xn))

            area = len(pixels)
            if area < min_area:
                continue

            ys = np.array([p[0] for p in pixels], dtype=np.int32)
            xs = np.array([p[1] for p in pixels], dtype=np.int32)
            weights = saliency[ys, xs]
            weight_sum = float(weights.sum())
            if weight_sum <= 0.0:
                continue

            hotspots.append({
                "centroid_x": round(float((xs * weights).sum() / weight_sum), 3),
                "centroid_y": round(float((ys * weights).sum() / weight_sum), 3),
                "strength": round(float(weights.sum()), 3),
                "mean_value": round(float(weights.mean()), 3),
                "peak_value": round(float(weights.max()), 3),
                "area": int(area),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "threshold_value": round(threshold, 3),
            })

    hotspots.sort(key=lambda h: (h["strength"], h["area"]), reverse=True)
    return hotspots[:top_k]


def build_word_step_records(metadata: dict) -> list[dict]:
    # Converte gli step token-level in step word-level.
    # Le heatmap dei pezzi della stessa parola vengono combinate con massimo per pixel.
    step_records = metadata.get("step_records", [])
    token_pieces = metadata.get("generated_token_pieces")
    if not isinstance(token_pieces, list):
        token_pieces = None
    word_records = metadata.get("word_records")
    if not isinstance(word_records, list) or not word_records:
        word_records = build_word_groups(step_records, token_pieces)

    original_image_path = metadata.get("image_path")
    word_steps: list[dict] = []
    for word in word_records:
        raw_map_paths = [Path(path) for path in word.get("source_raw_map_paths", []) if str(path)]
        valid_raw_paths = [path for path in raw_map_paths if path.exists()]
        heatmap_paths = [Path(path) for path in word.get("source_heatmap_paths", []) if str(path)]
        valid_paths = [path for path in heatmap_paths if path.exists()]
        if not valid_raw_paths and not valid_paths:
            continue

        if valid_raw_paths:
            maps = [load_saliency_map(path) for path in valid_raw_paths]
            combined_saliency = np.maximum.reduce(maps)
            combined = saliency_to_rgb(combined_saliency)
            hotspots = extract_hotspots_from_heatmap(combined_saliency)
            hotspot_source = "raw_tam_map"
        elif original_image_path and Path(str(original_image_path)).exists():
            maps = [estimate_heatmap_rgb(path, original_image_path) for path in valid_paths]
            combined = np.maximum.reduce(maps).astype(np.uint8)
            hotspots = extract_hotspots_from_heatmap(combined)
            hotspot_source = "legacy_overlay_reconstruction"
        else:
            maps = [rgb_array(path) for path in valid_paths]
            combined = np.maximum.reduce(maps).astype(np.uint8)
            hotspots = extract_hotspots_from_heatmap(combined)
            hotspot_source = "legacy_rgb_heatmap"
        dominant_hotspot = hotspots[0] if hotspots else None
        word_steps.append({
            "step_idx": int(word.get("word_index", len(word_steps))),
            "token_label": str(word.get("word_label", "")),
            "word_label": str(word.get("word_label", "")),
            "source_step_indices": word.get("source_step_indices", []),
            "heatmap_rgb": combined,
            "hotspots": hotspots,
            "dominant_hotspot": dominant_hotspot,
            "hotspot_source": hotspot_source,
        })
    return word_steps


def build_token_step_records(metadata: dict) -> list[dict]:
    # I run vecchi hanno solo heatmap_path.
    # Per poterli visualizzare comunque, ricostruisco hotspot e punto dominante dalla heatmap.
    original_image_path = metadata.get("image_path")
    enriched_steps: list[dict] = []
    for idx, step in enumerate(metadata.get("step_records", [])):
        if not isinstance(step, dict):
            continue
        heatmap_path = step.get("heatmap_path")
        if not heatmap_path or not Path(str(heatmap_path)).exists():
            enriched_steps.append(step)
            continue

        if "hotspots" in step and "dominant_hotspot" in step:
            enriched_steps.append(step)
            continue

        raw_map_path = step.get("raw_map_path")
        if raw_map_path and Path(str(raw_map_path)).exists():
            saliency = load_saliency_map(raw_map_path)
            hotspots = extract_hotspots_from_heatmap(saliency)
            hotspot_source = "raw_tam_map"
        elif original_image_path and Path(str(original_image_path)).exists():
            heatmap_rgb = estimate_heatmap_rgb(heatmap_path, original_image_path).astype(np.uint8)
            hotspots = extract_hotspots_from_heatmap(heatmap_rgb)
            hotspot_source = "legacy_overlay_reconstruction"
        else:
            heatmap_rgb = rgb_array(heatmap_path)
            hotspots = extract_hotspots_from_heatmap(heatmap_rgb)
            hotspot_source = "legacy_rgb_heatmap"
        enriched_steps.append({
            **step,
            "step_idx": int(step.get("step_idx", idx)),
            "hotspots": hotspots,
            "dominant_hotspot": hotspots[0] if hotspots else None,
            "hotspot_source": hotspot_source,
        })
    return enriched_steps


def aggregate_fixations(
    step_records: list[dict],
    merge_distance_px: float = 42.0,
) -> list[dict]:
    # Raggruppa punti consecutivi molto vicini.
    # Questo rende la vista piu' simile a fissazioni visive persistenti.
    fixations: list[dict] = []
    current: dict | None = None

    def flush_current() -> None:
        nonlocal current
        if not current:
            return
        points = current["points"]
        strengths = [float(p.get("strength", 1.0)) for p in points]
        weight_sum = sum(strengths) or 1.0
        current["centroid_x"] = round(sum(float(p["centroid_x"]) * w for p, w in zip(points, strengths)) / weight_sum, 3)
        current["centroid_y"] = round(sum(float(p["centroid_y"]) * w for p, w in zip(points, strengths)) / weight_sum, 3)
        current["duration_steps"] = len(points)
        current["label"] = " ".join(current["labels"])[:80]
        fixations.append(current)
        current = None

    for step in step_records:
        hotspot = step.get("dominant_hotspot")
        if not hotspot:
            flush_current()
            continue
        point = {
            "step_idx": int(step.get("step_idx", len(fixations))),
            "centroid_x": float(hotspot["centroid_x"]),
            "centroid_y": float(hotspot["centroid_y"]),
            "strength": float(hotspot.get("strength", 1.0)),
        }
        label = str(step.get("word_label") or step.get("token_label") or "")

        if current is None:
            current = {"start_step": point["step_idx"], "end_step": point["step_idx"], "points": [point], "labels": [label]}
            continue

        dx = point["centroid_x"] - float(current["points"][-1]["centroid_x"])
        dy = point["centroid_y"] - float(current["points"][-1]["centroid_y"])
        if float((dx ** 2 + dy ** 2) ** 0.5) <= merge_distance_px:
            current["points"].append(point)
            current["labels"].append(label)
            current["end_step"] = point["step_idx"]
        else:
            flush_current()
            current = {"start_step": point["step_idx"], "end_step": point["step_idx"], "points": [point], "labels": [label]}

    flush_current()
    for idx, fixation in enumerate(fixations, start=1):
        fixation["fixation_id"] = idx
    return fixations


def detect_metadata_capabilities(metadata: dict) -> dict:
    step_records = metadata.get("step_records")
    if not isinstance(step_records, list) or not step_records:
        return {
            "has_step_records": False,
            "has_heatmap_paths": False,
            "scanpath_ready": False,
            "reason": "missing_or_empty_step_records",
        }

    has_heatmap_paths = any(bool(step.get("heatmap_path")) for step in step_records if isinstance(step, dict))
    has_hotspots = all(isinstance(step, dict) and "hotspots" in step for step in step_records)
    has_dominant = all(isinstance(step, dict) and "dominant_hotspot" in step for step in step_records)
    scanpath = metadata.get("scanpath")
    has_tracks = isinstance(scanpath, dict) and isinstance(scanpath.get("tracks"), list)

    scanpath_ready = bool(has_heatmap_paths and has_hotspots and has_dominant and has_tracks)
    if scanpath_ready:
        reason = "scanpath_ready"
    elif has_heatmap_paths:
        reason = "heatmap_only"
    else:
        reason = "missing_heatmap_paths"

    return {
        "has_step_records": True,
        "has_heatmap_paths": bool(has_heatmap_paths),
        "has_hotspots": bool(has_hotspots),
        "has_dominant_hotspot": bool(has_dominant),
        "has_scanpath_tracks": bool(has_tracks),
        "scanpath_ready": bool(scanpath_ready),
        "reason": reason,
    }


def validate_cli_args(args: argparse.Namespace) -> None:
    metadata_path = Path(args.metadata)
    if not metadata_path.exists() or not metadata_path.is_file():
        raise ViewerInputError(f"--metadata non trovato: {metadata_path}")
    if args.view_mode not in SCANPATH_VIEW_MODES:
        raise ViewerInputError(f"--view-mode deve essere uno tra: {', '.join(sorted(SCANPATH_VIEW_MODES))}")
    if args.base_image not in {"heatmap", "original"}:
        raise ViewerInputError("--base-image deve essere heatmap oppure original")
    if args.sheet_cols < 1:
        raise ViewerInputError("--sheet-cols deve essere >= 1")
    if args.gif_ms < 1:
        raise ViewerInputError("--gif-ms deve essere >= 1")
    if args.max_steps is not None and args.max_steps < 1:
        raise ViewerInputError("--max-steps deve essere >= 1 se specificato")
    if args.dominant_tail < 2:
        raise ViewerInputError("--dominant-tail deve essere >= 2")
    if args.fixation_merge_distance_px < 1:
        raise ViewerInputError("--fixation-merge-distance-px deve essere >= 1")
    if args.path_tail < 0:
        raise ViewerInputError("--path-tail deve essere >= 0")


def validate_metadata_contract(metadata: dict, allow_heatmap_only: bool) -> tuple[str, dict]:
    caps = detect_metadata_capabilities(metadata)
    if not caps["has_step_records"]:
        raise MetadataContractError("Metadata non valido: step_records mancante o vuoto.")

    if caps["scanpath_ready"]:
        return "scanpath", caps

    if allow_heatmap_only and caps["has_heatmap_paths"]:
        return "heatmap_only", caps

    raise MetadataContractError(
        "Metadata non compatibile con la vista scanpath. Servono: step_records[].heatmap_path, "
        "step_records[].hotspots, step_records[].dominant_hotspot, scanpath.tracks. "
        "Riesegui il runner aggiornato oppure usa --allow-heatmap-only per i metadata legacy."
    )


def draw_hotspots_and_tracks(
    heatmap: np.ndarray,
    hotspots: list[dict],
    active_tracks: list[dict],
    dominant_path: list[dict],
    token_label: str = "",
    show_secondary_tracks: bool = False,
    dominant_tail: int = 24,
) -> Image.Image:
    image = Image.fromarray(heatmap, mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    label_font = load_font(18)

    for hotspot in hotspots:
        x = float(hotspot["centroid_x"])
        y = float(hotspot["centroid_y"])
        area = max(1.0, float(hotspot.get("area", 1)))
        radius = float(np.clip(np.sqrt(area) * 0.8, 4.0, 20.0))
        bbox = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bbox, outline=(255, 255, 0, 255), width=2)
        # Punto centrale dell'hotspot.
        dot_r = 3.2
        draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r], fill=(255, 255, 255, 255))

    if show_secondary_tracks:
        for track in active_tracks:
            points = track.get("points", [])
            if len(points) < 2:
                continue
            xy = [(float(p["centroid_x"]), float(p["centroid_y"])) for p in points]
            draw.line(xy, fill=(70, 220, 255, 110), width=1)

    # Mostro solo l'ultimo spostamento del percorso dominante.
    if len(dominant_path) >= 2:
        prev_point = dominant_path[-2]
        curr_point = dominant_path[-1]
        dominant_xy = [
            (float(prev_point["centroid_x"]), float(prev_point["centroid_y"])),
            (float(curr_point["centroid_x"]), float(curr_point["centroid_y"])),
        ]
        draw.line(dominant_xy, fill=(255, 255, 255, 225), width=3)
        for idx, p in enumerate([prev_point, curr_point]):
            x = float(p["centroid_x"])
            y = float(p["centroid_y"])
            r = 4.5 if idx == 1 else 3.0
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 255), outline=(0, 0, 0, 220), width=1)
    elif len(dominant_path) == 1:
        p = dominant_path[-1]
        x = float(p["centroid_x"])
        y = float(p["centroid_y"])
        r = 4.5
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 255), outline=(0, 0, 0, 220), width=1)

    if token_label:
        label = f"token: {token_label}"
        bbox = draw.textbbox((12, 10), label, font=label_font)
        pad_x = 8
        pad_y = 6
        bg = [
            bbox[0] - pad_x,
            bbox[1] - pad_y,
            bbox[2] + pad_x,
            bbox[3] + pad_y,
        ]
        draw.rounded_rectangle(bg, radius=6, fill=(0, 0, 0, 170), outline=(255, 255, 255, 130), width=1)
        draw.text((12, 10), label, fill=(255, 255, 255, 255), font=label_font)

    return image


def active_tracks_until_step(tracks: list[dict], step_idx: int) -> list[dict]:
    active = []
    for track in tracks:
        prefix = [p for p in track.get("points", []) if int(p.get("step_idx", -1)) <= step_idx]
        if not prefix:
            continue
        active.append({"track_id": track.get("track_id", -1), "points": prefix})
    return active


def dominant_path_until_step(step_records: list[dict], step_idx: int) -> list[dict]:
    # Percorso costruito con l'hotspot dominante di ogni step.
    points = []
    for step in step_records:
        idx = int(step.get("step_idx", -1))
        if idx > step_idx:
            continue
        hotspot = step.get("dominant_hotspot")
        if not hotspot:
            continue
        points.append({
            "step_idx": idx,
            "centroid_x": float(hotspot["centroid_x"]),
            "centroid_y": float(hotspot["centroid_y"]),
        })
    return points


def path_points_until_step(step_records: list[dict], step_idx: int) -> list[dict]:
    points = []
    for step in step_records:
        idx = int(step.get("step_idx", -1))
        if idx > step_idx:
            continue
        hotspot = step.get("dominant_hotspot")
        if not hotspot:
            continue
        points.append({
            "step_idx": idx,
            "centroid_x": float(hotspot["centroid_x"]),
            "centroid_y": float(hotspot["centroid_y"]),
            "strength": float(hotspot.get("strength", 1.0)),
            "label": str(step.get("word_label") or step.get("token_label") or ""),
        })
    return points


def fixation_points_until_step(fixations: list[dict], step_idx: int) -> list[dict]:
    points = []
    for fixation in fixations:
        if int(fixation.get("start_step", -1)) > step_idx:
            continue
        points.append({
            "step_idx": int(fixation.get("fixation_id", len(points) + 1)),
            "centroid_x": float(fixation["centroid_x"]),
            "centroid_y": float(fixation["centroid_y"]),
            "strength": float(fixation.get("duration_steps", 1)),
            "label": str(fixation.get("label", "")),
        })
    return points


def draw_clean_path_frame(
    base_rgb: np.ndarray,
    current_step: dict,
    path_points: list[dict],
    show_secondary_hotspots: bool = False,
    show_jump_breaks: bool = True,
    jump_distance_px: float = 120.0,
    show_numbers: bool = False,
    max_numbers: int = 14,
    path_tail: int = 0,
) -> Image.Image:
    image = Image.fromarray(base_rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    label_font = load_font(18)
    number_font = load_font(16)

    if show_secondary_hotspots:
        for hotspot in current_step.get("hotspots", []) or []:
            x = float(hotspot["centroid_x"])
            y = float(hotspot["centroid_y"])
            area = max(1.0, float(hotspot.get("area", 1)))
            radius = float(np.clip(np.sqrt(area) * 0.6, 3.0, 14.0))
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=(255, 230, 70, 100), width=1)

    visible_points = path_points[-path_tail:] if path_tail > 0 else path_points

    # Disegno solo la coda scelta del percorso.
    # In questo modo le GIF lunghe non diventano una rete di linee.
    for idx in range(1, len(visible_points)):
        prev_point = visible_points[idx - 1]
        curr_point = visible_points[idx]
        x0, y0 = float(prev_point["centroid_x"]), float(prev_point["centroid_y"])
        x1, y1 = float(curr_point["centroid_x"]), float(curr_point["centroid_y"])
        dist = float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
        recent_boost = idx / max(1, len(visible_points) - 1)
        alpha = int(70 + 150 * recent_boost)
        if show_jump_breaks and dist > jump_distance_px:
            # Un salto lungo resta visibile ma non viene letto come continuita' fluida.
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            draw.line([(x0, y0), (mid_x, mid_y)], fill=(255, 255, 255, alpha), width=2)
            draw.line([(mid_x, mid_y), (x1, y1)], fill=(255, 110, 80, alpha), width=2)
        else:
            draw.line([(x0, y0), (x1, y1)], fill=(255, 255, 255, alpha), width=2)

    hidden_count = max(0, len(path_points) - len(visible_points))
    for idx, point in enumerate(visible_points):
        x = float(point["centroid_x"])
        y = float(point["centroid_y"])
        is_current = idx == len(visible_points) - 1
        age_ratio = (idx + 1) / max(1, len(visible_points))
        radius = 6.5 if is_current else 3.5 + 2.0 * age_ratio
        fill = (255, 255, 255, 255) if is_current else (80, 210, 255, int(70 + 120 * age_ratio))
        outline = (0, 0, 0, 230) if is_current else (0, 0, 0, 130)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill, outline=outline, width=1)

        if show_numbers and idx < max_numbers:
            label = str(hidden_count + idx + 1)
            bbox = draw.textbbox((x + 7, y - 9), label, font=number_font)
            draw.rectangle([bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2], fill=(0, 0, 0, 150))
            draw.text((x + 7, y - 9), label, fill=(255, 255, 255, 255), font=number_font)

    label = str(current_step.get("word_label") or current_step.get("token_label") or "")
    if label:
        text = f"step: {int(current_step.get('step_idx', 0))}   {label}"
        bbox = draw.textbbox((12, 10), text, font=label_font)
        draw.rounded_rectangle(
            [bbox[0] - 8, bbox[1] - 6, bbox[2] + 8, bbox[3] + 6],
            radius=6,
            fill=(0, 0, 0, 170),
            outline=(255, 255, 255, 120),
            width=1,
        )
        draw.text((12, 10), text, fill=(255, 255, 255, 255), font=label_font)

    return image


def draw_fixation_summary(
    base_rgb: np.ndarray,
    fixations: list[dict],
    show_numbers: bool = True,
    jump_distance_px: float = 140.0,
) -> Image.Image:
    image = Image.fromarray(base_rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    number_font = load_font(18)

    for idx in range(1, len(fixations)):
        prev_point = fixations[idx - 1]
        curr_point = fixations[idx]
        x0, y0 = float(prev_point["centroid_x"]), float(prev_point["centroid_y"])
        x1, y1 = float(curr_point["centroid_x"]), float(curr_point["centroid_y"])
        dist = float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
        color = (255, 120, 80, 210) if dist > jump_distance_px else (255, 255, 255, 210)
        draw.line([(x0, y0), (x1, y1)], fill=color, width=3)

    for idx, fixation in enumerate(fixations, start=1):
        x = float(fixation["centroid_x"])
        y = float(fixation["centroid_y"])
        duration = int(fixation.get("duration_steps", 1))
        radius = float(np.clip(5.0 + duration * 2.2, 7.0, 22.0))
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(80, 210, 255, 190), outline=(0, 0, 0, 230), width=2)
        if show_numbers:
            label = str(idx)
            bbox = draw.textbbox((x - 5, y - 9), label, font=number_font)
            draw.text((x - 5, y - 9), label, fill=(0, 0, 0, 255), font=number_font)

    return image


def build_scanpath_frames_with_stats(
    metadata: dict,
    output_dir: Path,
    max_steps: int | None = None,
    show_secondary_tracks: bool = False,
    dominant_tail: int = 24,
    view_mode: str = "token_raw",
    base_image: str = "heatmap",
    show_secondary_hotspots: bool = False,
    show_jump_breaks: bool = True,
    show_fixation_numbers: bool = False,
    fixation_merge_distance_px: float = 42.0,
    path_tail: int = 0,
) -> tuple[list[Path], dict]:
    ensure_dir(output_dir)
    if view_mode not in SCANPATH_VIEW_MODES:
        raise ViewerInputError(f"view_mode non valido: {view_mode}")

    original_image_path = metadata.get("image_path")
    word_mode = view_mode in {"word", "word_fixations"}
    if word_mode:
        step_records = build_word_step_records(metadata)
    else:
        step_records = build_token_step_records(metadata)

    tracks = metadata.get("scanpath", {}).get("tracks", [])
    frames: list[Path] = []
    skipped_missing_heatmap_path = 0
    skipped_missing_heatmap_file = 0
    if max_steps is not None:
        step_records = step_records[: max(0, int(max_steps))]

    fixations = aggregate_fixations(step_records, merge_distance_px=fixation_merge_distance_px) if view_mode == "word_fixations" else []

    for step in step_records:
        step_idx = int(step.get("step_idx", len(frames)))
        heatmap_path = step.get("heatmap_path")
        raw_map_path = step.get("raw_map_path")
        has_raw_map = bool(raw_map_path and Path(str(raw_map_path)).exists())
        has_inline_heatmap = isinstance(step.get("heatmap_rgb"), np.ndarray)
        if not heatmap_path and not has_inline_heatmap and not has_raw_map:
            skipped_missing_heatmap_path += 1
            continue
        if heatmap_path and not Path(heatmap_path).exists() and not has_raw_map:
            skipped_missing_heatmap_file += 1
            continue

        if has_inline_heatmap:
            heatmap = step["heatmap_rgb"]
        else:
            heatmap = heatmap_for_step(
                step,
                original_image_path=original_image_path,
                recover_from_overlay=base_image == "original",
            )

        base_rgb = blend_on_original(heatmap, original_image_path, heatmap_alpha=0.42) if base_image == "original" else heatmap
        hotspots = step.get("hotspots", []) or []
        active_tracks = active_tracks_until_step(tracks, step_idx)
        dominant_path = dominant_path_until_step(step_records, step_idx)

        if view_mode == "token_raw":
            frame = draw_hotspots_and_tracks(
                base_rgb,
                hotspots,
                active_tracks,
                dominant_path,
                token_label=str(step.get("token_label", "")),
                show_secondary_tracks=show_secondary_tracks,
                dominant_tail=dominant_tail,
            )
        else:
            if view_mode == "word_fixations":
                display_points = fixation_points_until_step(fixations, step_idx)
            else:
                display_points = path_points_until_step(step_records, step_idx)
            frame = draw_clean_path_frame(
                base_rgb=base_rgb,
                current_step=step,
                path_points=display_points,
                show_secondary_hotspots=show_secondary_hotspots,
                show_jump_breaks=show_jump_breaks,
                show_numbers=show_fixation_numbers and view_mode != "word_fixations",
                path_tail=path_tail,
            )

        token = safe_name(step.get("word_label") or step.get("token_label", "tok"))
        out_path = output_dir / f"step_{step_idx:04d}_{token}.png"
        frame.save(out_path)
        frames.append(out_path)

    summary_path = output_dir / "scanpath_summary.jpg"
    if view_mode == "word_fixations" and fixations:
        first_heatmap = None
        for step in step_records:
            if isinstance(step.get("heatmap_rgb"), np.ndarray):
                first_heatmap = step["heatmap_rgb"]
                break
        if first_heatmap is not None:
            base_rgb = blend_on_original(first_heatmap, original_image_path, heatmap_alpha=0.25) if base_image == "original" else first_heatmap
            draw_fixation_summary(base_rgb, fixations, show_numbers=True).save(summary_path)

    stats = {
        "view_mode": view_mode,
        "base_image": base_image,
        "steps_considered": len(step_records),
        "frames_generated": len(frames),
        "fixations_generated": len(fixations),
        "path_tail": path_tail,
        "skipped_missing_heatmap_path": skipped_missing_heatmap_path,
        "skipped_missing_heatmap_file": skipped_missing_heatmap_file,
        "summary_path": str(summary_path) if summary_path.exists() else "",
    }
    return frames, stats


def build_scanpath_frames(
    metadata: dict,
    output_dir: Path,
    max_steps: int | None = None,
    show_secondary_tracks: bool = False,
    dominant_tail: int = 24,
    view_mode: str = "token_raw",
    base_image: str = "heatmap",
    path_tail: int = 0,
) -> list[Path]:
    frames, _ = build_scanpath_frames_with_stats(
        metadata=metadata,
        output_dir=output_dir,
        max_steps=max_steps,
        show_secondary_tracks=show_secondary_tracks,
        dominant_tail=dominant_tail,
        view_mode=view_mode,
        base_image=base_image,
        path_tail=path_tail,
    )
    return frames


def save_gif(frame_paths: list[Path], gif_path: Path, duration_ms: int = 220) -> None:
    if not frame_paths:
        return
    frames = [Image.open(path).convert("RGB") for path in frame_paths]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def save_contact_sheet(frame_paths: list[Path], output_path: Path, cols: int = 8) -> None:
    if not frame_paths:
        return
    if cols < 1:
        raise ViewerInputError("save_contact_sheet: cols deve essere >= 1")
    label_font = load_font(20)
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    w = max(img.width for img in images)
    h = max(img.height for img in images)
    rows = int(np.ceil(len(images) / max(1, cols)))
    pad = 6
    label_h = 34
    tile_h = h + label_h
    canvas = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * tile_h + (rows + 1) * pad), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = pad + col * (w + pad)
        y = pad + row * (tile_h + pad)
        if img.size != (w, h):
            img = img.resize((w, h), Image.BILINEAR)
        label = frame_paths[idx].stem.replace("step_", "step ").replace("_", " ")
        label = label[:48]
        draw.rectangle([x, y, x + w, y + label_h], fill=(18, 18, 18))
        draw.text((x + 8, y + 6), label, fill=(255, 255, 255), font=label_font)
        canvas.paste(img, (x, y + label_h))
    canvas.save(output_path)


def quick_preview(frame_paths: list[Path], n: int = 12) -> None:
    if not frame_paths:
        return
    sample = frame_paths[:n]
    cols = min(4, len(sample))
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes[len(sample):]:
        ax.axis("off")
    for ax, path in zip(axes, sample):
        ax.imshow(rgb_array(path))
        ax.set_title(path.stem[:28], fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render dei frame scanpath a partire da metadata.json")
    parser.add_argument("--metadata", required=True, help="Percorso di metadata.json")
    parser.add_argument("--out-dir", help="Cartella di output. Default: <prompt_dir>/scanpath_views")
    parser.add_argument("--max-steps", type=int, help="Numero massimo di step da renderizzare.")
    parser.add_argument("--gif-ms", type=int, default=220, help="Durata dei frame GIF in millisecondi.")
    parser.add_argument("--sheet-cols", type=int, default=8, help="Numero di colonne della contact sheet.")
    parser.add_argument(
        "--view-mode",
        choices=sorted(SCANPATH_VIEW_MODES),
        default="token_raw",
        help="Vista da generare: tecnica token, token pulita, word-level o word-level con fissazioni.",
    )
    parser.add_argument(
        "--base-image",
        choices=["heatmap", "original"],
        default="heatmap",
        help="Sfondo della visualizzazione. original rende la scena piu' leggibile.",
    )
    parser.add_argument("--allow-heatmap-only", action="store_true", help="Permette metadata legacy senza campi scanpath.")
    parser.add_argument("--show-secondary-tracks", action="store_true", help="Mostra anche le tracce secondarie.")
    parser.add_argument("--show-secondary-hotspots", action="store_true", help="Mostra hotspot secondari nella vista pulita.")
    parser.add_argument("--hide-jump-breaks", action="store_true", help="Non evidenzia i salti lunghi nella vista pulita.")
    parser.add_argument("--show-fixation-numbers", action="store_true", help="Mostra piccoli numeri sui punti principali.")
    parser.add_argument("--fixation-merge-distance-px", type=float, default=42.0, help="Distanza massima per unire fissazioni vicine.")
    parser.add_argument("--path-tail", type=int, default=0, help="Numero di punti recenti da mostrare nella vista pulita. 0 = percorso completo.")
    parser.add_argument("--dominant-tail", type=int, default=24, help="Parametro mantenuto per compatibilita' con le chiamate esistenti.")
    parser.add_argument("--no-preview", action="store_true", help="Disattiva l'anteprima matplotlib.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        validate_cli_args(args)
        metadata_path = Path(args.metadata).resolve()
        metadata = load_json(metadata_path)
        # Le viste nuove possono lavorare anche sui run legacy:
        # se ci sono le heatmap, hotspot e fissazioni vengono ricostruiti offline.
        mode, caps = validate_metadata_contract(metadata, allow_heatmap_only=True)

        default_folder = f"scanpath_views_{args.view_mode}" if mode == "scanpath" else f"heatmap_views_{args.view_mode}"
        default_out = metadata_path.parent / default_folder
        out_dir = Path(args.out_dir).resolve() if args.out_dir else default_out
        ensure_dir(out_dir)

        frame_paths, stats = build_scanpath_frames_with_stats(
            metadata=metadata,
            output_dir=out_dir,
            max_steps=args.max_steps,
            show_secondary_tracks=args.show_secondary_tracks if mode == "scanpath" else False,
            dominant_tail=args.dominant_tail,
            view_mode=args.view_mode,
            base_image=args.base_image,
            show_secondary_hotspots=args.show_secondary_hotspots,
            show_jump_breaks=not args.hide_jump_breaks,
            show_fixation_numbers=args.show_fixation_numbers,
            fixation_merge_distance_px=args.fixation_merge_distance_px,
            path_tail=args.path_tail,
        )

        if not frame_paths:
            print("stato: fallito")
            print(f"mode: {mode}")
            print("motivo: nessun frame valido")
            print(f"step_considerati: {stats['steps_considered']}")
            print(f"step_saltati_senza_heatmap_path: {stats['skipped_missing_heatmap_path']}")
            print(f"step_saltati_heatmap_mancante: {stats['skipped_missing_heatmap_file']}")
            raise SystemExit(2)

        gif_path = out_dir / "scanpath.gif"
        sheet_path = out_dir / "scanpath_contact_sheet.jpg"
        save_gif(frame_paths, gif_path, duration_ms=args.gif_ms)
        save_contact_sheet(frame_paths, sheet_path, cols=args.sheet_cols)
        summary_payload = {
            "metadata_path": str(metadata_path),
            "mode": mode,
            "view_mode": args.view_mode,
            "base_image": args.base_image,
            "capabilities": caps,
            "stats": stats,
            "gif_path": str(gif_path) if gif_path.exists() else "",
            "contact_sheet_path": str(sheet_path) if sheet_path.exists() else "",
        }
        (out_dir / "scanpath_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        skips = stats["skipped_missing_heatmap_path"] + stats["skipped_missing_heatmap_file"]
        status = "completato" if skips == 0 else "completato_con_scarti"
        print(f"stato: {status}")
        print(f"mode: {mode}")
        print(f"view_mode: {args.view_mode}")
        print(f"base_image: {args.base_image}")
        print(f"capacita_metadata: {json.dumps(caps, ensure_ascii=True)}")
        print(f"frame_generati: {stats['frames_generated']}")
        print(f"fixations_generate: {stats['fixations_generated']}")
        print(f"path_tail: {stats['path_tail']}")
        print(f"step_considerati: {stats['steps_considered']}")
        print(f"step_saltati_senza_heatmap_path: {stats['skipped_missing_heatmap_path']}")
        print(f"step_saltati_heatmap_mancante: {stats['skipped_missing_heatmap_file']}")
        print(f"cartella: {out_dir}")
        if gif_path.exists():
            print(f"gif_salvata: {gif_path}")
        if sheet_path.exists():
            print(f"contact_sheet_salvata: {sheet_path}")
        if stats["summary_path"]:
            print(f"summary_statica: {stats['summary_path']}")
        print(f"summary_json: {out_dir / 'scanpath_summary.json'}")

        if not args.no_preview:
            quick_preview(frame_paths)
    except ViewerInputError as err:
        print("stato: fallito")
        print("motivo: input_non_valido")
        print(f"dettaglio: {err}")
        raise SystemExit(2) from err
    except (json.JSONDecodeError, OSError) as err:
        print("stato: fallito")
        print("motivo: metadata_non_legibile")
        print(f"dettaglio: {err}")
        raise SystemExit(2) from err
    except MetadataContractError as err:
        print("stato: fallito")
        print("motivo: contratto_metadata")
        print(f"dettaglio: {err}")
        raise SystemExit(2) from err


if __name__ == "__main__":
    main()
