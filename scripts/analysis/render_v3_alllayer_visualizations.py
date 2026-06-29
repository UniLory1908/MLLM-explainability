from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


EXPECTED_LAYER_COUNT = 29
HEATMAP_ALPHA = 0.42
HOTSPOT_PERCENTILE = 95.0
MIN_HOTSPOT_AREA = 64

IMAGE_CASES = [
    {"case_id": "kitchen_counter", "image_dir": "000000030213_kitchen_counter", "label": "kitchen_counter"},
    {"case_id": "bathroom_toilet", "image_dir": "000000331352_bathroom_toilet", "label": "bathroom_toilet"},
    {"case_id": "microwave_bottle", "image_dir": "000000426253_microwave_bottle", "label": "microwave_bottle"},
    {"case_id": "street_traffic", "image_dir": "000000393226_street_traffic", "label": "street_traffic"},
    {"case_id": "desk_monitor", "image_dir": "000000555009_desk_monitor", "label": "desk_monitor"},
]

PROMPT_LABELS = [
    "baseline_neutral",
    "image_grounded_visible_only",
    "ambiguous_open",
    "misleading_wrong_subject",
    "extra_knowledge_context",
    "reasoning_controlled_brief",
    "order_disruption_stress",
    "colleague_obj_detection_hard",
]

PROMPT_TEXTS = {
    "baseline_neutral": "Describe the image.",
    "image_grounded_visible_only": "Describe the image using only directly visible details in one sentence.",
    "ambiguous_open": "What is happening in the image?",
    "misleading_wrong_subject": "Describe the dog in the image.",
    "extra_knowledge_context": "Describe the image and the likely function of the main visible object.",
    "reasoning_controlled_brief": "Identify the main visible object and briefly explain why it is the most relevant.",
    "order_disruption_stress": "Describe only visible details in image.",
    "colleague_obj_detection_hard": "Is there a {obj_main}?",
}

FIX256_OVERRIDES = {
    ("kitchen_counter", "baseline_neutral"),
    ("street_traffic", "extra_knowledge_context"),
    ("desk_monitor", "baseline_neutral"),
    ("desk_monitor", "order_disruption_stress"),
}


@dataclass
class CaseSelection:
    case_id: str
    image_dir: str
    prompt_label: str
    metadata_path: Path | None
    selected_prompt_dir: Path | None
    selected_run_dir: Path | None
    original_run_dir: Path | None
    used_fix256: bool
    status: str = "pending"
    message: str = ""
    token_count: int = 0
    word_count: int = 0
    rendered_word_count: int = 0
    layer_count: int = 0
    raw_map_count: int = 0
    expected_map_count: int = 0
    missing_map_count: int = 0


def project_root_from_here() -> Path:
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "README.md").exists() and (candidate / "scripts").exists():
            return candidate
    return current


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "item"


def relpath(path: Path, start: Path) -> str:
    path_abs = path if path.is_absolute() else Path.cwd() / path
    start_abs = start if start.is_absolute() else Path.cwd() / start
    try:
        return Path(os.path.relpath(path_abs, start_abs)).as_posix()
    except ValueError:
        pass
    try:
        return path.resolve().relative_to(start.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def html_rel(path: Path, base: Path) -> str:
    return html.escape(relpath(path, base))


def get_font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def import_word_utils(project_root: Path):
    sys.path.insert(0, str(project_root))
    try:
        from prompt_word_utils import build_word_groups, token_pieces_from_metadata

        return build_word_groups, token_pieces_from_metadata
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"[warn] prompt_word_utils non disponibile, uso fallback semplice: {exc}")
        return None, None


def fallback_word_groups(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    labels = metadata.get("generated_token_labels") or []
    groups = []
    for idx, label in enumerate(labels):
        label = str(label).strip()
        if not re.search(r"[A-Za-z0-9]", label):
            continue
        groups.append(
            {
                "word_index": len(groups),
                "word_label": label,
                "canonical_word_label": re.sub(r"[^a-zA-Z0-9]+", "", label.lower()),
                "source_step_indices": [idx],
            }
        )
    return groups


def load_image_registry(project_root: Path) -> dict[str, dict[str, Any]]:
    path = project_root / "configs" / "image_registry.json"
    if not path.exists():
        return {}
    data = load_json(path)
    return data.get("images", {})


def find_prompt_dir(run_dir: Path, prompt_label: str) -> Path | None:
    if not run_dir.exists():
        return None
    for child in sorted(run_dir.iterdir()):
        if child.is_dir() and child.name.endswith(prompt_label):
            metadata_path = child / "metadata.json"
            if metadata_path.exists():
                try:
                    metadata = load_json(metadata_path)
                    if metadata.get("prompt_id") == prompt_label or metadata.get("prompt_label") == prompt_label:
                        return child
                except json.JSONDecodeError:
                    continue
    return None


def resolve_case(run_root: Path, image_case: dict[str, str], prompt_label: str) -> CaseSelection:
    case_id = image_case["case_id"]
    image_dir = image_case["image_dir"]
    base = run_root / image_dir
    main_run = base / f"v3_wordlevel_gpu_alllayers_{image_case['label']}"
    use_fix = (case_id, prompt_label) in FIX256_OVERRIDES
    selected_run = main_run
    if use_fix:
        selected_run = base / f"v3_wordlevel_gpu_alllayers_fix256_{image_case['label']}"

    prompt_dir = find_prompt_dir(selected_run, prompt_label)
    metadata_path = prompt_dir / "metadata.json" if prompt_dir else None
    selection = CaseSelection(
        case_id=case_id,
        image_dir=image_dir,
        prompt_label=prompt_label,
        metadata_path=metadata_path,
        selected_prompt_dir=prompt_dir,
        selected_run_dir=selected_run if selected_run.exists() else None,
        original_run_dir=main_run if main_run.exists() else None,
        used_fix256=use_fix,
    )
    if metadata_path is None:
        selection.status = "missing_metadata"
        selection.message = f"metadata not found for {prompt_label} in {selected_run}"
    return selection


def discover_cases(
    run_root: Path,
    image_filter: str | None,
    prompt_filter: str | None,
) -> list[CaseSelection]:
    selected_images = [
        case
        for case in IMAGE_CASES
        if not image_filter or image_filter in case["case_id"] or image_filter in case["image_dir"]
    ]
    selected_prompts = [prompt for prompt in PROMPT_LABELS if not prompt_filter or prompt_filter in prompt]
    return [resolve_case(run_root, image_case, prompt) for image_case in selected_images for prompt in selected_prompts]


def raw_map_index(prompt_dir: Path, image_stem: str) -> dict[int, dict[int, Path]]:
    root = prompt_dir / "raw_maps" / image_stem
    index: dict[int, dict[int, Path]] = {}
    if not root.exists():
        return index
    for layer_dir in root.glob("layer_*"):
        if not layer_dir.is_dir():
            continue
        match = re.search(r"layer_(\d+)", layer_dir.name)
        if not match:
            continue
        layer = int(match.group(1))
        for path in layer_dir.glob("step_*.npy"):
            step_match = re.match(r"step_(\d+)_", path.name)
            if not step_match:
                continue
            step = int(step_match.group(1))
            index.setdefault(layer, {})[step] = path
    return index


def rendered_heatmap_index(prompt_dir: Path, image_stem: str) -> dict[int, dict[int, Path]]:
    root = prompt_dir / "vis_results" / image_stem
    index: dict[int, dict[int, Path]] = {}
    if not root.exists():
        return index
    for layer_dir in root.glob("layer_*"):
        if not layer_dir.is_dir():
            continue
        match = re.search(r"layer_(\d+)", layer_dir.name)
        if not match:
            continue
        layer = int(match.group(1))
        for path in layer_dir.glob("step_*.jpg"):
            step_match = re.match(r"step_(\d+)_", path.name)
            if not step_match:
                continue
            step = int(step_match.group(1))
            index.setdefault(layer, {})[step] = path
    return index


def validate_selection(selection: CaseSelection, metadata: dict[str, Any], maps: dict[int, dict[int, Path]]) -> None:
    token_count = len(metadata.get("step_records") or metadata.get("generated_token_ids") or [])
    layer_values = metadata.get("layers") or sorted(maps)
    layer_count = len(layer_values)
    raw_map_count = sum(len(paths) for paths in maps.values())
    expected = token_count * EXPECTED_LAYER_COUNT
    missing = max(expected - raw_map_count, 0)
    selection.token_count = token_count
    selection.layer_count = layer_count
    selection.raw_map_count = raw_map_count
    selection.expected_map_count = expected
    selection.missing_map_count = missing

    messages = []
    if layer_count != EXPECTED_LAYER_COUNT:
        messages.append(f"expected {EXPECTED_LAYER_COUNT} layers, found {layer_count}")
    if raw_map_count != expected:
        messages.append(f"expected {expected} raw maps, found {raw_map_count}")
    if messages:
        selection.status = "inconsistent"
        selection.message = "; ".join(messages)
    else:
        selection.status = "validated"
        selection.message = "ok"


def load_saliency(path: Path) -> np.ndarray:
    value = np.load(path)
    if value.ndim == 3:
        value = value.max(axis=2)
    return np.asarray(value, dtype=np.float32)


def word_layer_maps(
    word_groups: list[dict[str, Any]],
    maps: dict[int, dict[int, Path]],
    layers: list[int],
    max_words: int,
) -> tuple[list[dict[str, Any]], list[list[np.ndarray | None]]]:
    selected_words = word_groups[:max_words]
    arrays: list[list[np.ndarray | None]] = []
    for word in selected_words:
        step_indices = [int(step) for step in word.get("source_step_indices", [])]
        word_arrays = []
        for layer in layers:
            token_maps = []
            for step in step_indices:
                path = maps.get(layer, {}).get(step)
                if path and path.exists():
                    token_maps.append(load_saliency(path))
            word_arrays.append(np.maximum.reduce(token_maps) if token_maps else None)
        arrays.append(word_arrays)
    return selected_words, arrays


def global_bounds(arrays: list[list[np.ndarray | None]], percentiles: tuple[float, float]) -> tuple[float, float]:
    values = []
    for row in arrays:
        for item in row:
            if item is not None and item.size:
                values.append(np.ravel(item))
    if not values:
        return 0.0, 1.0
    combined = np.concatenate(values)
    low, high = np.percentile(combined, percentiles)
    if not np.isfinite(low):
        low = float(np.nanmin(combined))
    if not np.isfinite(high) or high <= low:
        high = float(low + 1e-6)
    return float(low), float(high)


def normalize_global(value: np.ndarray | None, low: float, high: float) -> np.ndarray:
    if value is None:
        return np.zeros((16, 16), dtype=np.float32)
    return np.clip((value - low) / max(high - low, 1e-8), 0.0, 1.0)


def heatmap_image(norm: np.ndarray, size: int) -> Image.Image:
    arr = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").resize((size, size), Image.Resampling.BICUBIC)
    return img.convert("RGB")


def colorize(norm: np.ndarray) -> Image.Image:
    arr = np.clip(norm, 0.0, 1.0)
    red = np.clip(255 * np.minimum(1.0, arr * 2.0), 0, 255)
    green = np.clip(255 * (1.0 - np.abs(arr * 2.0 - 1.0)), 0, 255)
    blue = np.clip(255 * np.maximum(0.0, 1.0 - arr * 2.0), 0, 255)
    rgb = np.dstack([red, green, blue]).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def overlay_on_image(base: Image.Image, saliency: np.ndarray | None, low: float, high: float, size: int) -> Image.Image:
    base = base.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    norm = normalize_global(saliency, low, high)
    heat = colorize(norm).resize((size, size), Image.Resampling.BICUBIC)
    return Image.blend(base, heat, alpha=HEATMAP_ALPHA)


def overlay_on_image_local(base: Image.Image, saliency: np.ndarray | None, size: int) -> Image.Image:
    base = base.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    if saliency is None:
        norm = np.zeros((16, 16), dtype=np.float32)
    else:
        value = np.asarray(saliency, dtype=np.float32)
        low = float(np.nanmin(value))
        high = float(np.nanmax(value))
        norm = normalize_global(value, low, high)
    heat = colorize(norm).resize((size, size), Image.Resampling.BICUBIC)
    return Image.blend(base, heat, alpha=HEATMAP_ALPHA)


def original_rendered_cell(
    rendered_maps: dict[int, dict[int, Path]],
    word: dict[str, Any],
    layer: int,
    size: int,
) -> Image.Image | None:
    step_indices = [int(step) for step in word.get("source_step_indices", [])]
    paths = [rendered_maps.get(layer, {}).get(step) for step in step_indices]
    paths = [path for path in paths if path and path.exists()]
    if not paths:
        return None
    token_images = [
        np.asarray(Image.open(token_path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS), dtype=np.uint8)
        for token_path in paths
    ]
    merged = np.maximum.reduce(token_images)
    return Image.fromarray(merged, mode="RGB")


def dominant_hotspot(
    value: np.ndarray | None,
    threshold_percentile: float = HOTSPOT_PERCENTILE,
    min_area: int = MIN_HOTSPOT_AREA,
) -> tuple[float, float] | None:
    if value is None or value.size == 0:
        return None
    saliency = np.asarray(value, dtype=np.float64)
    saliency = np.nan_to_num(saliency, nan=0.0, posinf=0.0, neginf=0.0)
    saliency = np.maximum(saliency, 0.0)
    positive = saliency[saliency > 0]
    if positive.size == 0:
        return None
    threshold = float(np.percentile(positive, threshold_percentile))
    mask = saliency >= threshold
    if not np.any(mask):
        return None

    height, width = saliency.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[tuple[float, int, float, float]] = []

    for y0 in range(height):
        for x0 in range(width):
            if visited[y0, x0] or not mask[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yn = y + dy
                    xn = x + dx
                    if yn < 0 or yn >= height or xn < 0 or xn >= width:
                        continue
                    if visited[yn, xn] or not mask[yn, xn]:
                        continue
                    visited[yn, xn] = True
                    stack.append((yn, xn))

            area = len(pixels)
            if area < min_area:
                continue
            yy = np.array([pixel[0] for pixel in pixels], dtype=np.int32)
            xx = np.array([pixel[1] for pixel in pixels], dtype=np.int32)
            weights = saliency[yy, xx]
            strength = float(weights.sum())
            if strength <= 1e-12:
                continue
            cx = float((xx * weights).sum() / strength)
            cy = float((yy * weights).sum() / strength)
            components.append((strength, area, cx, cy))

    if not components:
        return None
    _, _, cx, cy = max(components, key=lambda item: (item[0], item[1]))
    return cx, cy


def draw_scanpath_tail_panel(
    base_image: Image.Image | None,
    words: list[dict[str, Any]],
    arrays: list[list[np.ndarray | None]],
    layers: list[int],
    current_pos: int,
    size: int,
    tail: int = 3,
) -> Image.Image:
    panel = Image.new("RGB", (size, size + 44), "white")
    draw = ImageDraw.Draw(panel)
    font = get_font(15)
    small_font = get_font(12)
    final_layer = max(layers) if layers else 0
    if base_image is not None:
        image = base_image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    else:
        image = Image.new("RGB", (size, size), (245, 245, 245))
    panel.paste(image, (0, 34))
    draw.text((6, 8), f"Last 3 word hotspots on L{final_layer:02d}", fill="black", font=font)

    if final_layer not in layers:
        return panel
    layer_pos = layers.index(final_layer)
    start = max(0, current_pos - tail + 1)
    points = []
    for pos in range(start, current_pos + 1):
        if pos >= len(arrays):
            continue
        hotspot = dominant_hotspot(arrays[pos][layer_pos])
        if hotspot is None:
            continue
        source = arrays[pos][layer_pos]
        height, width = source.shape if source is not None else (16, 16)
        x = 0.0 if width <= 1 else hotspot[0] / float(width - 1) * (size - 1)
        y = 0.0 if height <= 1 else hotspot[1] / float(height - 1) * (size - 1)
        points.append((x, y + 34, pos, str(words[pos].get("word_label", ""))))

    if len(points) >= 2:
        xy = [(x, y) for x, y, _, _ in points]
        draw.line(xy, fill=(255, 255, 255, 225), width=3)
    for _, (x, y, pos, label) in enumerate(points, start=1):
        radius = 7.0 if pos == current_pos else 4.8
        fill = (255, 220, 20, 255) if pos == current_pos else (255, 255, 255, 255)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=fill,
            outline=(0, 0, 0, 220),
            width=1,
        )
    return panel


def draw_last_layer_pair_panel(
    words: list[dict[str, Any]],
    arrays: list[list[np.ndarray | None]],
    rendered_maps: dict[int, dict[int, Path]],
    base_image: Image.Image | None,
    layers: list[int],
    current_pos: int,
    current_word: dict[str, Any],
    low: float,
    high: float,
    cell_size: int,
) -> Image.Image:
    final_layer = max(layers) if layers else 0
    layer_pos = layers.index(final_layer) if final_layer in layers else 0
    saliency = arrays[current_pos][layer_pos] if layers else None
    width = (cell_size * 2) + 22
    height = cell_size + 50
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    font = get_font(16)
    small_font = get_font(13)
    draw.text((4, 4), f"Last layer L{final_layer:02d}", fill="black", font=font)
    draw.text((4, 25), "TAM originale + immagine", fill="black", font=small_font)
    draw.text((cell_size + 18, 25), "TAM normalizzata + immagine", fill="black", font=small_font)

    original = original_rendered_cell(rendered_maps, current_word, final_layer, cell_size)
    if original is None:
        original = overlay_on_image(base_image, saliency, low, high, cell_size) if base_image is not None else heatmap_image(normalize_global(saliency, low, high), cell_size)
    if base_image is not None:
        normalized = overlay_on_image_local(base_image, saliency, cell_size)
    elif saliency is None:
        normalized = heatmap_image(np.zeros((16, 16), dtype=np.float32), cell_size)
    else:
        normalized = heatmap_image(normalize_global(saliency, float(np.nanmin(saliency)), float(np.nanmax(saliency))), cell_size)

    original = draw_tail_on_cell(original, words, arrays, layers, current_pos, layer_pos, tail=3)
    normalized = draw_tail_on_cell(normalized, words, arrays, layers, current_pos, layer_pos, tail=3)
    panel.paste(original, (0, 50))
    panel.paste(normalized, (cell_size + 22, 50))
    return panel


def draw_notebook_pair_frame(
    path: Path,
    metadata: dict[str, Any],
    words: list[dict[str, Any]],
    arrays: list[list[np.ndarray | None]],
    rendered_maps: dict[int, dict[int, Path]],
    layers: list[int],
    word_pos: int,
    word: dict[str, Any],
    low: float,
    high: float,
    panel_size: int = 560,
) -> Path:
    final_layer = max(layers) if layers else 0
    layer_pos = layers.index(final_layer) if final_layer in layers else 0
    saliency = arrays[word_pos][layer_pos] if layers else None
    image_path = Path(str(metadata.get("image_path", "")))
    base_image = Image.open(image_path).convert("RGB") if image_path.exists() else None

    original = original_rendered_cell(rendered_maps, word, final_layer, panel_size)
    if original is None:
        original = overlay_on_image(base_image, saliency, low, high, panel_size) if base_image is not None else heatmap_image(normalize_global(saliency, low, high), panel_size)
    if base_image is not None:
        normalized = overlay_on_image_local(base_image, saliency, panel_size)
    elif saliency is None:
        normalized = heatmap_image(np.zeros((16, 16), dtype=np.float32), panel_size)
    else:
        normalized = heatmap_image(normalize_global(saliency, float(np.nanmin(saliency)), float(np.nanmax(saliency))), panel_size)

    original = draw_tail_on_cell(original, words, arrays, layers, word_pos, layer_pos, tail=3)
    normalized = draw_tail_on_cell(normalized, words, arrays, layers, word_pos, layer_pos, tail=3)

    header_h = 32
    pad = 10
    font = get_font(17)
    word_label = str(word.get("word_label", ""))

    def panel(title: str, image: Image.Image) -> Image.Image:
        canvas = Image.new("RGB", (panel_size, panel_size + header_h), "white")
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw.text((10, 8), title[:80], fill=(20, 20, 20), font=font)
        canvas.paste(image, (0, header_h))
        return canvas

    left = panel(f"TAM originale + immagine | {word_label}", original)
    right = panel(f"TAM normalizzata + immagine | {word_label}", normalized)
    canvas = Image.new("RGB", (left.width + right.width + pad, max(left.height, right.height)), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + pad, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95, subsampling=0)
    return path


def draw_tail_on_cell(
    cell: Image.Image,
    words: list[dict[str, Any]],
    arrays: list[list[np.ndarray | None]],
    layers: list[int],
    current_pos: int,
    layer_pos: int,
    tail: int = 3,
    show_labels: bool = False,
) -> Image.Image:
    result = cell.convert("RGBA")
    draw = ImageDraw.Draw(result, "RGBA")
    image_width = result.width
    image_height = result.height
    start = max(0, current_pos - tail + 1)
    points = []
    for pos in range(start, current_pos + 1):
        if pos >= len(arrays):
            continue
        source = arrays[pos][layer_pos]
        hotspot = dominant_hotspot(source)
        if hotspot is None or source is None:
            continue
        height, width = source.shape
        x = 0.0 if width <= 1 else hotspot[0] / float(width - 1) * (image_width - 1)
        y = 0.0 if height <= 1 else hotspot[1] / float(height - 1) * (image_height - 1)
        points.append((x, y, pos))
    if len(points) >= 2:
        xy = [(x, y) for x, y, _ in points]
        draw.line(xy, fill=(255, 255, 255, 225), width=3)
    for _, (x, y, pos) in enumerate(points, start=1):
        radius = 7.0 if pos == current_pos else 4.8
        fill = (255, 220, 20, 255) if pos == current_pos else (255, 255, 255, 255)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=fill,
            outline=(0, 0, 0, 220),
            width=1,
        )
        if show_labels:
            font = get_font(11)
            draw.text((x + radius + 2, y - 7), str(pos), fill=(0, 0, 0, 255), font=font)
    return result.convert("RGB")


def draw_response_gif(
    path: Path,
    metadata: dict[str, Any],
    words: list[dict[str, Any]],
    arrays: list[list[np.ndarray | None]],
    rendered_maps: dict[int, dict[int, Path]],
    layers: list[int],
    low: float,
    high: float,
    width: int = 640,
) -> Path | None:
    if not layers:
        return None
    final_layer = max(layers)
    if final_layer not in layers:
        return None
    layer_pos = layers.index(final_layer)
    image_path = Path(str(metadata.get("image_path", "")))
    if not image_path.exists():
        return None
    base_image = Image.open(image_path).convert("RGB")
    height = max(1, round(width * base_image.height / base_image.width))
    frames = []
    font = get_font(18)
    for pos, word in enumerate(words):
        saliency = arrays[pos][layer_pos]
        base_frame = base_image.resize((width, height), Image.Resampling.LANCZOS)
        if saliency is None:
            norm = np.zeros((16, 16), dtype=np.float32)
        else:
            value = np.asarray(saliency, dtype=np.float32)
            norm = normalize_global(value, float(np.nanmin(value)), float(np.nanmax(value)))
        heat = colorize(norm).resize((width, height), Image.Resampling.BICUBIC)
        frame = Image.blend(base_frame, heat, alpha=HEATMAP_ALPHA)
        frame = draw_tail_on_cell(frame, words, arrays, layers, pos, layer_pos, tail=3)
        draw = ImageDraw.Draw(frame, "RGBA")
        label = str(word.get("word_label", ""))
        title = f"L{final_layer:02d} word {word.get('word_index', pos):03d}: {label[:36]}"
        box = draw.textbbox((0, 0), title, font=font)
        draw.rectangle((8, 8, box[2] + 22, box[3] + 18), fill=(255, 255, 255, 210))
        draw.text((15, 12), title, fill=(0, 0, 0, 255), font=font)
        frames.append(frame)
    if not frames:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=550, loop=0, optimize=True)
    return path


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def draw_matrix_page(
    path: Path,
    words: list[dict[str, Any]],
    arrays: list[list[np.ndarray | None]],
    layers: list[int],
    row_start: int,
    row_end: int,
    low: float,
    high: float,
    thumb_size: int,
) -> None:
    font = get_font(13)
    small_font = get_font(12)
    label_w = 210
    header_h = 36
    row_h = thumb_size + 8
    width = label_w + len(layers) * thumb_size
    height = header_h + (row_end - row_start) * row_h
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for col, layer in enumerate(layers):
        x = label_w + col * thumb_size
        draw.text((x + 8, 10), f"L{layer:02d}", fill="black", font=small_font)
    for local_row, word_idx in enumerate(range(row_start, row_end)):
        y = header_h + local_row * row_h
        word = words[word_idx]
        label = f"{word.get('word_index', word_idx):03d} {word.get('word_label', '')}"
        if len(label) > 28:
            label = label[:25] + "..."
        draw.text((8, y + 8), label, fill="black", font=font)
        for col, layer in enumerate(layers):
            x = label_w + col * thumb_size
            norm = normalize_global(arrays[word_idx][col], low, high)
            cell = heatmap_image(norm, thumb_size - 4)
            image.paste(cell, (x + 2, y + 2))
    image.save(path, quality=92)


def metrics_for_map(value: np.ndarray | None, previous_centroid: tuple[float, float] | None) -> dict[str, float | None]:
    if value is None or value.size == 0:
        return {
            "energy_sum": None,
            "energy_mean": None,
            "max_value": None,
            "entropy_or_dispersion": None,
            "centroid_x": None,
            "centroid_y": None,
            "centroid_shift_from_previous_layer": None,
        }
    arr = np.asarray(value, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    energy = float(arr.sum())
    mean = float(arr.mean())
    max_value = float(arr.max())
    if energy > 1e-12:
        yy, xx = np.indices(arr.shape)
        cx = float((xx * arr).sum() / energy)
        cy = float((yy * arr).sum() / energy)
        prob = arr.ravel() / energy
        prob = prob[prob > 0]
        entropy = float(-(prob * np.log(prob)).sum() / math.log(max(arr.size, 2)))
    else:
        cx = cy = entropy = 0.0
    if previous_centroid is None:
        shift = None
    else:
        shift = float(math.hypot(cx - previous_centroid[0], cy - previous_centroid[1]))
    return {
        "energy_sum": energy,
        "energy_mean": mean,
        "max_value": max_value,
        "entropy_or_dispersion": entropy,
        "centroid_x": cx,
        "centroid_y": cy,
        "centroid_shift_from_previous_layer": shift,
    }


def write_metrics(
    out_dir: Path,
    words: list[dict[str, Any]],
    arrays: list[list[np.ndarray | None]],
    layers: list[int],
) -> tuple[Path, Path, list[dict[str, Any]]]:
    metrics_path = out_dir / "metrics.csv"
    by_word_path = out_dir / "metrics_by_word.json"
    rows = []
    by_word: dict[str, Any] = {}
    for word_pos, word in enumerate(words):
        previous: tuple[float, float] | None = None
        word_key = str(word.get("word_index", word_pos))
        by_word[word_key] = {
            "word_index": word.get("word_index", word_pos),
            "word_label": word.get("word_label", ""),
            "source_step_indices": word.get("source_step_indices", []),
            "layers": [],
        }
        for layer_pos, layer in enumerate(layers):
            item = metrics_for_map(arrays[word_pos][layer_pos], previous)
            if item["centroid_x"] is not None and item["centroid_y"] is not None:
                previous = (float(item["centroid_x"]), float(item["centroid_y"]))
            row = {
                "word_index": word.get("word_index", word_pos),
                "word_label": word.get("word_label", ""),
                "source_step_indices": "|".join(str(x) for x in word.get("source_step_indices", [])),
                "layer_index": layer,
                **item,
            }
            rows.append(row)
            by_word[word_key]["layers"].append({"layer_index": layer, **item})

    fieldnames = [
        "word_index",
        "word_label",
        "source_step_indices",
        "layer_index",
        "energy_sum",
        "energy_mean",
        "max_value",
        "entropy_or_dispersion",
        "centroid_x",
        "centroid_y",
        "centroid_shift_from_previous_layer",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(by_word_path, by_word)
    return metrics_path, by_word_path, rows


def select_words(
    words: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    selected_count: int,
    obj_main: str | None,
) -> list[int]:
    by_word: dict[int, list[dict[str, Any]]] = {}
    for row in metric_rows:
        by_word.setdefault(int(row["word_index"]), []).append(row)
    selected: list[int] = []

    def add(idx: int) -> None:
        if idx not in selected and idx < len(words):
            selected.append(idx)

    if obj_main:
        targets = {re.sub(r"[^a-zA-Z0-9]+", "", part.lower()) for part in obj_main.split()}
        for pos, word in enumerate(words):
            label = re.sub(r"[^a-zA-Z0-9]+", "", str(word.get("word_label", "")).lower())
            if label in targets or any(target and target in label for target in targets):
                add(pos)

    energy_scores = []
    variation_scores = []
    for pos, word in enumerate(words):
        rows = by_word.get(int(word.get("word_index", pos)), [])
        energies = [float(row["energy_sum"]) for row in rows if row["energy_sum"] not in (None, "")]
        if energies:
            energy_scores.append((float(np.mean(energies)), pos))
            variation_scores.append((float(np.std(energies)), pos))
    for _, pos in sorted(energy_scores, reverse=True):
        add(pos)
        if len(selected) >= selected_count:
            return selected[:selected_count]
    for _, pos in sorted(variation_scores, reverse=True):
        add(pos)
        if len(selected) >= selected_count:
            return selected[:selected_count]
    for pos in range(min(len(words), selected_count)):
        add(pos)
    return selected[:selected_count]


def draw_selected_report(
    path: Path,
    metadata: dict[str, Any],
    words: list[dict[str, Any]],
    arrays: list[list[np.ndarray | None]],
    layers: list[int],
    selected_word_positions: list[int],
    selected_layers: list[int],
    low: float,
    high: float,
    title: str = "Selected layer report",
) -> None:
    thumb = 180
    label_w = 260
    top_h = 175
    gap = 16
    row_h = (thumb * 2) + 74
    selected_layer_positions = [layers.index(layer) for layer in selected_layers if layer in layers]
    cols = len(selected_layer_positions)
    width = label_w + cols * thumb
    height = top_h + max(1, len(selected_word_positions)) * row_h
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = get_font(17)
    small_font = get_font(14)
    title_font = get_font(22)

    draw.text((12, 10), title, fill="black", font=title_font)
    draw.text((12, 44), f"Prompt: {metadata.get('prompt_id', '')}", fill="black", font=font)
    draw.text((12, 70), "Top row: global percentile normalization per image/prompt", fill="black", font=small_font)
    draw.text((12, 92), "Bottom row: local min-max normalization per word/layer", fill="black", font=small_font)
    draw.text((12, 114), "TAM heatmaps are overlays, not eye-tracking.", fill="black", font=small_font)

    original_path = Path(str(metadata.get("image_path", "")))
    base_image: Image.Image | None = None
    if original_path.exists():
        base_image = Image.open(original_path).convert("RGB")
        preview = base_image.copy()
        preview.thumbnail((120, 120), Image.Resampling.BILINEAR)
        image.paste(preview, (width - preview.width - 12, 12))

    for col, layer_pos in enumerate(selected_layer_positions):
        x = label_w + col * thumb
        draw.text((x + 10, top_h - 30), f"L{layers[layer_pos]:02d}", fill="black", font=font)

    for row, word_pos in enumerate(selected_word_positions):
        y = top_h + row * row_h
        word = words[word_pos]
        label = f"{word.get('word_index', word_pos):03d} {word.get('word_label', '')}"
        if len(label) > 28:
            label = label[:25] + "..."
        draw.text((12, y + 16), label, fill="black", font=title_font)
        draw.text((12, y + 56), "global", fill="black", font=font)
        draw.text((12, y + 56 + thumb + gap), "local", fill="black", font=font)
        for col, layer_pos in enumerate(selected_layer_positions):
            x = label_w + col * thumb
            saliency = arrays[word_pos][layer_pos]
            if base_image is not None:
                cell_global = overlay_on_image(base_image, saliency, low, high, thumb - 10)
                cell_local = overlay_on_image_local(base_image, saliency, thumb - 10)
            else:
                cell_global = heatmap_image(normalize_global(saliency, low, high), thumb - 10)
                if saliency is None:
                    local_norm = np.zeros((16, 16), dtype=np.float32)
                else:
                    local_norm = normalize_global(saliency, float(np.nanmin(saliency)), float(np.nanmax(saliency)))
                cell_local = heatmap_image(local_norm, thumb - 10)
            image.paste(cell_global, (x + 5, y + 48))
            image.paste(cell_local, (x + 5, y + 48 + thumb + gap))
    image.save(path, quality=92)


def draw_ordered_report_pages(
    out_dir: Path,
    metadata: dict[str, Any],
    words: list[dict[str, Any]],
    arrays: list[list[np.ndarray | None]],
    layers: list[int],
    selected_layers: list[int],
    low: float,
    high: float,
    words_per_page: int = 6,
) -> list[Path]:
    pages = []
    positions = list(range(len(words)))
    for page_idx, start in enumerate(range(0, len(positions), words_per_page), start=1):
        page_positions = positions[start : start + words_per_page]
        path = out_dir / f"ordered_layers_report_page_{page_idx:02d}.jpg"
        draw_selected_report(
            path,
            metadata,
            words,
            arrays,
            layers,
            page_positions,
            selected_layers,
            low,
            high,
            title=f"All rendered words in generation order, page {page_idx:02d}",
        )
        pages.append(path)
    return pages


def draw_word_detail(
    path: Path,
    metadata: dict[str, Any],
    words: list[dict[str, Any]],
    arrays: list[list[np.ndarray | None]],
    word_pos: int,
    word: dict[str, Any],
    word_arrays: list[np.ndarray | None],
    rendered_maps: dict[int, dict[int, Path]],
    layers: list[int],
    selected_layers: list[int],
    low: float,
    high: float,
) -> None:
    layer_positions = list(range(len(layers)))
    thumb = 260
    label_w = 180
    max_cols = 5
    top_h = 120
    group_gap = 34
    layer_h = (thumb * 2) + 78
    groups = [layer_positions[start : start + max_cols] for start in range(0, len(layer_positions), max_cols)]
    width = label_w + max_cols * thumb
    height = top_h + max(1, len(groups)) * layer_h + max(0, len(groups) - 1) * group_gap
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(22)
    font = get_font(16)
    small_font = get_font(13)

    word_label = str(word.get("word_label", ""))
    source_tokens = ", ".join(str(x) for x in word.get("source_token_labels", []))
    source_steps = ", ".join(str(x) for x in word.get("source_step_indices", []))
    draw.text((12, 10), f"Word detail: {word.get('word_index', '')} {word_label}", fill="black", font=title_font)
    draw.text((12, 42), f"source tokens: {source_tokens or word_label}", fill="black", font=small_font)
    draw.text((12, 62), f"source steps: {source_steps}", fill="black", font=small_font)
    draw.text((12, 84), "TAM overlays, not eye-tracking. Original row uses TAM-rendered JPGs, merged pixel-wise for multi-token words.", fill="black", font=small_font)

    original_path = Path(str(metadata.get("image_path", "")))
    base_image = Image.open(original_path).convert("RGB") if original_path.exists() else None

    for group_idx, group in enumerate(groups):
        y = top_h + group_idx * (layer_h + group_gap)
        draw.text((12, y + 36), "original", fill="black", font=font)
        draw.text((12, y + 36 + thumb + 18), "local", fill="black", font=font)
        for col, layer_pos in enumerate(group):
            layer = layers[layer_pos]
            x = label_w + col * thumb
            draw.text((x + 8, y + 8), f"L{layer:02d}", fill="black", font=font)
            saliency = word_arrays[layer_pos]
            original_cell = original_rendered_cell(rendered_maps, word, layer, thumb - 12)
            if original_cell is None:
                original_cell = overlay_on_image(base_image, saliency, low, high, thumb - 12) if base_image is not None else heatmap_image(normalize_global(saliency, low, high), thumb - 12)
            if base_image is not None:
                local_cell = overlay_on_image_local(base_image, saliency, thumb - 12)
            elif saliency is None:
                local_cell = heatmap_image(np.zeros((16, 16), dtype=np.float32), thumb - 12)
            else:
                local_norm = normalize_global(saliency, float(np.nanmin(saliency)), float(np.nanmax(saliency)))
                local_cell = heatmap_image(local_norm, thumb - 12)
            original_cell = draw_tail_on_cell(original_cell, words, arrays, layers, word_pos, layer_pos, tail=3)
            local_cell = draw_tail_on_cell(local_cell, words, arrays, layers, word_pos, layer_pos, tail=3)
            image.paste(original_cell, (x + 6, y + 34))
            image.paste(local_cell, (x + 6, y + 34 + thumb + 18))
    image.save(path, quality=95, subsampling=0)


def draw_word_outputs(
    out_dir: Path,
    metadata: dict[str, Any],
    words: list[dict[str, Any]],
    arrays: list[list[np.ndarray | None]],
    rendered_maps: dict[int, dict[int, Path]],
    layers: list[int],
    selected_layers: list[int],
    low: float,
    high: float,
) -> tuple[list[Path], list[Path]]:
    detail_dir = out_dir / "word_details"
    focus_dir = out_dir / "word_focus"
    detail_dir.mkdir(parents=True, exist_ok=True)
    focus_dir.mkdir(parents=True, exist_ok=True)
    detail_paths = []
    focus_paths = []
    for pos, word in enumerate(words):
        word_index = int(word.get("word_index", pos))
        word_label = safe_name(str(word.get("word_label", "word")))
        detail_path = detail_dir / f"word_{word_index:03d}_{word_label}.jpg"
        focus_path = focus_dir / f"word_{word_index:03d}_{word_label}.jpg"
        draw_notebook_pair_frame(focus_path, metadata, words, arrays, rendered_maps, layers, pos, word, low, high)
        draw_word_detail(detail_path, metadata, words, arrays, pos, word, arrays[pos], rendered_maps, layers, selected_layers, low, high)
        focus_paths.append(focus_path)
        detail_paths.append(detail_path)
    return focus_paths, detail_paths


def render_case(
    selection: CaseSelection,
    project_root: Path,
    out_root: Path,
    args: argparse.Namespace,
    image_registry: dict[str, dict[str, Any]],
    build_word_groups,
    token_pieces_from_metadata,
) -> dict[str, Any]:
    case_out = out_root / selection.image_dir / selection.prompt_label
    case_out.mkdir(parents=True, exist_ok=True)

    if selection.metadata_path is None or selection.selected_prompt_dir is None:
        return case_manifest(selection, case_out, [])

    metadata = load_json(selection.metadata_path)
    image_stem = str(metadata.get("image_stem") or Path(str(metadata.get("image_path", ""))).stem)
    maps = raw_map_index(selection.selected_prompt_dir, image_stem)
    rendered_maps = rendered_heatmap_index(selection.selected_prompt_dir, image_stem)
    validate_selection(selection, metadata, maps)
    layers = sorted(maps.keys())

    step_records = metadata.get("step_records", [])
    if build_word_groups and token_pieces_from_metadata:
        token_pieces = token_pieces_from_metadata(metadata)
        word_groups = build_word_groups(step_records, token_pieces)
    else:
        word_groups = fallback_word_groups(metadata)
    selection.word_count = len(word_groups)
    render_word_limit = selection.word_count if args.max_words <= 0 else min(selection.word_count, args.max_words)
    selection.rendered_word_count = render_word_limit

    rows_per_page = max(1, min(20, max(1, (12000 - 36) // (args.thumb_size + 8))))
    expected_page_count = max(1, math.ceil(max(selection.rendered_word_count, 1) / rows_per_page))
    expected_pages = [
        case_out / f"word_layer_matrix_global_page_{page_idx:02d}.jpg"
        for page_idx in range(1, expected_page_count + 1)
    ]
    detail_dir = case_out / "word_details"
    focus_dir = case_out / "word_focus"
    expected_detail_count = selection.rendered_word_count
    metrics_csv = case_out / "metrics.csv"
    metrics_json = case_out / "metrics_by_word.json"
    response_gif = case_out / "response_l28_global.gif"
    overview_path = case_out / "overview.html"
    existing_lightweight_outputs = (
        metrics_csv.exists()
        and metrics_json.exists()
        and detail_dir.exists()
        and focus_dir.exists()
        and len(list(detail_dir.glob("word_*.jpg"))) >= expected_detail_count
        and len(list(focus_dir.glob("word_*.jpg"))) >= expected_detail_count
        and response_gif.exists()
        and all(page.exists() for page in expected_pages)
    )
    if existing_lightweight_outputs and not args.force:
        print(f"[skip] existing rendered outputs for {selection.image_dir} / {selection.prompt_label}")
        detail_pages = sorted(detail_dir.glob("word_*.jpg"))[:expected_detail_count]
        focus_pages = sorted(focus_dir.glob("word_*.jpg"))[:expected_detail_count]
        write_overview(
            overview_path,
            out_root,
            selection,
            metadata,
            word_groups[:render_word_limit],
            focus_pages,
            detail_pages,
            response_gif,
            expected_pages,
            metrics_csv,
            metrics_json,
            None,
            None,
        )
        generated_paths = [metrics_csv, metrics_json, response_gif, *focus_pages, *detail_pages, *expected_pages, overview_path]
        return case_manifest(selection, case_out, generated_paths, metadata, None, None)

    selected_words, arrays = word_layer_maps(word_groups, maps, layers, render_word_limit)
    selection.rendered_word_count = len(selected_words)
    low, high = global_bounds(arrays, args.global_percentiles)

    generated_paths: list[Path] = []
    if args.force or not (metrics_csv.exists() and metrics_json.exists()):
        metrics_csv, metrics_json, metric_rows = write_metrics(case_out, selected_words, arrays, layers)
    else:
        print(f"[skip] {metrics_csv}")
        print(f"[skip] {metrics_json}")
        metric_rows = []
        with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
            metric_rows = list(csv.DictReader(handle))
    generated_paths.extend([metrics_csv, metrics_json])

    existing_detail_pages = sorted(detail_dir.glob("word_*.jpg")) if detail_dir.exists() else []
    existing_focus_pages = sorted(focus_dir.glob("word_*.jpg")) if focus_dir.exists() else []
    if args.force or len(existing_detail_pages) < len(selected_words) or len(existing_focus_pages) < len(selected_words):
        focus_pages, detail_pages = draw_word_outputs(
            case_out,
            metadata,
            selected_words,
            arrays,
            rendered_maps,
            layers,
            args.selected_layers,
            low,
            high,
        )
    else:
        detail_pages = existing_detail_pages[: len(selected_words)]
        focus_pages = existing_focus_pages[: len(selected_words)]
        for page in [*focus_pages, *detail_pages]:
            print(f"[skip] {page}")
    generated_paths.extend(focus_pages)
    generated_paths.extend(detail_pages)

    if args.force or not response_gif.exists():
        gif_path = draw_response_gif(response_gif, metadata, selected_words, arrays, rendered_maps, layers, low, high)
        if gif_path is not None:
            generated_paths.append(gif_path)
    else:
        print(f"[skip] {response_gif}")
        generated_paths.append(response_gif)

    matrix_pages = []
    for page_idx, start in enumerate(range(0, len(selected_words), rows_per_page), start=1):
        end = min(start + rows_per_page, len(selected_words))
        page_path = case_out / f"word_layer_matrix_global_page_{page_idx:02d}.jpg"
        if args.force or not page_path.exists():
            draw_matrix_page(page_path, selected_words, arrays, layers, start, end, low, high, args.thumb_size)
        else:
            print(f"[skip] {page_path}")
        matrix_pages.append(page_path)
        generated_paths.append(page_path)

    write_overview(
        overview_path,
        out_root,
        selection,
        metadata,
        selected_words,
        focus_pages,
        detail_pages,
        response_gif,
        matrix_pages,
        metrics_csv,
        metrics_json,
        low,
        high,
    )
    generated_paths.append(overview_path)
    return case_manifest(selection, case_out, generated_paths, metadata, low, high)


def case_manifest(
    selection: CaseSelection,
    case_out: Path,
    generated_paths: list[Path],
    metadata: dict[str, Any] | None = None,
    low: float | None = None,
    high: float | None = None,
) -> dict[str, Any]:
    return {
        "case_id": selection.case_id,
        "image_dir": selection.image_dir,
        "prompt_label": selection.prompt_label,
        "status": selection.status,
        "message": selection.message,
        "used_fix256": selection.used_fix256,
        "selected_run_dir": str(selection.selected_run_dir) if selection.selected_run_dir else "",
        "original_run_dir": str(selection.original_run_dir) if selection.original_run_dir else "",
        "metadata_path": str(selection.metadata_path) if selection.metadata_path else "",
        "output_dir": str(case_out),
        "token_count": selection.token_count,
        "word_count": selection.word_count,
        "rendered_word_count": selection.rendered_word_count,
        "layer_count": selection.layer_count,
        "raw_map_count": selection.raw_map_count,
        "expected_map_count": selection.expected_map_count,
        "missing_map_count": selection.missing_map_count,
        "global_norm_low": low,
        "global_norm_high": high,
        "prompt_text": metadata.get("prompt_text", "") if metadata else "",
        "response_text": metadata.get("response_text", "") if metadata else "",
        "generated_paths": [str(path) for path in generated_paths],
        "overview_path": str(case_out / "overview.html"),
    }


def write_overview(
    path: Path,
    out_root: Path,
    selection: CaseSelection,
    metadata: dict[str, Any],
    words: list[dict[str, Any]],
    focus_pages: list[Path],
    detail_pages: list[Path],
    response_gif: Path,
    matrix_pages: list[Path],
    metrics_csv: Path,
    metrics_json: Path,
    low: float | None,
    high: float | None,
) -> None:
    image_path = Path(str(metadata.get("image_path", "")))
    image_preview = ""
    if image_path.exists():
        gif_html = ""
        if response_gif.exists():
            gif_html = (
                '<figure><figcaption>L28 response GIF</figcaption>'
                f'<img loading="lazy" src="{html_rel(response_gif, path.parent)}"></figure>'
            )
        image_preview = (
            '<div class="top-media">'
            f'<figure><figcaption>Original image</figcaption><img loading="lazy" src="{html_rel(image_path, path.parent)}"></figure>'
            f"{gif_html}"
            '</div>'
        )
    matrix_html = "\n".join(
        f'<p><img loading="lazy" src="{html_rel(page, path.parent)}" style="max-width:100%"></p>' for page in matrix_pages
    )
    focus_by_index = {}
    for focus_page in focus_pages:
        match = re.search(r"word_(\d+)_", focus_page.name)
        if match:
            focus_by_index[int(match.group(1))] = focus_page
    detail_by_index = {}
    for detail_page in detail_pages:
        match = re.search(r"word_(\d+)_", detail_page.name)
        if match:
            detail_by_index[int(match.group(1))] = detail_page
    word_buttons = []
    for pos, word in enumerate(words):
        word_index = int(word.get("word_index", pos))
        focus_page = focus_by_index.get(word_index)
        detail_page = detail_by_index.get(word_index)
        if not focus_page or not detail_page:
            continue
        label = str(word.get("word_label", ""))
        tokens = ", ".join(str(x) for x in word.get("source_token_labels", [])) or label
        steps = ", ".join(str(x) for x in word.get("source_step_indices", []))
        word_buttons.append(
            "<button class=\"word-chip\" "
            f"data-focus=\"{html_rel(focus_page, path.parent)}\" "
            f"data-layers=\"{html_rel(detail_page, path.parent)}\" "
            f"data-word=\"{html.escape(label)}\" "
            f"data-tokens=\"{html.escape(tokens)}\" "
            f"data-steps=\"{html.escape(steps)}\">"
            f"{html.escape(label)}</button>"
        )
    bounds_text = "available in manifest from render pass" if low is None or high is None else f"{low:.6g}, {high:.6g}"
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(selection.image_dir)} / {html.escape(selection.prompt_label)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1700px; margin: 24px auto; line-height: 1.35; }}
    pre {{ white-space: pre-wrap; background: #f5f5f5; padding: 12px; }}
    table {{ border-collapse: collapse; }}
    td, th {{ border: 1px solid #ddd; padding: 6px 8px; }}
    details {{ margin: 16px 0; }}
    figure {{ margin: 0; width: 420px; }}
    figcaption {{ font-size: 13px; color: #444; margin-bottom: 6px; min-height: 18px; }}
    .top-media {{ display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start; margin: 16px 0; }}
    .top-media img {{ width: 420px; max-width: 420px; height: auto; display: block; }}
    .word-response {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0; }}
    .word-chip {{ border: 1px solid #bbb; background: #fafafa; padding: 5px 8px; cursor: pointer; font-size: 14px; }}
    .word-chip:hover, .word-chip.active {{ background: #e9f1ff; border-color: #4477aa; }}
    .word-detail {{ border: 1px solid #ddd; padding: 12px; margin-top: 12px; }}
    .word-detail img {{ max-width: 100%; display: none; }}
    .focus-img {{ width: min(100%, 1130px); }}
    .layers-img {{ width: min(100%, 1480px); }}
  </style>
</head>
<body>
  <p><a href="../../index.html">Back to index</a></p>
  <h1>{html.escape(selection.image_dir)} / {html.escape(selection.prompt_label)}</h1>
  {image_preview}
  <table>
    <tr><th>used_fix256</th><td>{selection.used_fix256}</td></tr>
    <tr><th>status</th><td>{html.escape(selection.status)}: {html.escape(selection.message)}</td></tr>
    <tr><th>selected_run_dir</th><td>{html.escape(str(selection.selected_run_dir))}</td></tr>
    <tr><th>token_count</th><td>{selection.token_count}</td></tr>
    <tr><th>word_count</th><td>{selection.word_count}</td></tr>
    <tr><th>rendered_word_count</th><td>{selection.rendered_word_count}</td></tr>
    <tr><th>layer_count</th><td>{selection.layer_count}</td></tr>
    <tr><th>global_percentile_bounds</th><td>{html.escape(bounds_text)}</td></tr>
  </table>
  <h2>Prompt</h2>
  <pre>{html.escape(str(metadata.get("prompt_text", "")))}</pre>
  <h2>Response</h2>
  <pre>{html.escape(str(metadata.get("response_text", "")))}</pre>
  <details>
    <summary>Clickable word-level response</summary>
    <p>Click a word to inspect its TAM maps. Multi-token words use the existing word-level aggregation over source tokens.</p>
    <div class="word-response">
      {''.join(word_buttons)}
    </div>
    <div class="word-detail">
      <strong id="detail-title">Select a word</strong>
      <div id="detail-meta"></div>
      <p>The main frame follows the notebook style on the final layer: TAM originale + immagine and TAM normalizzata + immagine. The TAM-derived path shows only the last 3 word hotspots. For multi-token words, source token maps are merged pixel-wise into one word-level view.</p>
      <img id="focus-img" class="focus-img" loading="lazy" alt="word focus">
      <details id="layers-detail">
        <summary>All 29 layers for this word</summary>
        <p>Each layer is shown as original TAM-rendered view above the locally normalized word-level view.</p>
        <img id="layers-img" class="layers-img" loading="lazy" alt="all layer word detail">
      </details>
    </div>
  </details>
  <details>
    <summary>Word x layer matrix pages, global normalized</summary>
    {matrix_html}
  </details>
  <script>
    const buttons = document.querySelectorAll('.word-chip');
    const title = document.getElementById('detail-title');
    const meta = document.getElementById('detail-meta');
    const focusImg = document.getElementById('focus-img');
    const layersImg = document.getElementById('layers-img');
    const layersDetail = document.getElementById('layers-detail');
    buttons.forEach((button) => {{
      button.addEventListener('click', () => {{
        buttons.forEach((item) => item.classList.remove('active'));
        button.classList.add('active');
        title.textContent = 'Word detail: ' + button.dataset.word;
        meta.textContent = 'source tokens: ' + button.dataset.tokens + ' | source steps: ' + button.dataset.steps;
        focusImg.src = button.dataset.focus;
        focusImg.style.display = 'block';
        layersImg.src = button.dataset.layers;
        layersImg.style.display = 'block';
        layersDetail.open = false;
      }});
    }});
  </script>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def write_summary(out_root: Path, manifests: list[dict[str, Any]]) -> tuple[Path, Path]:
    summary_json = out_root / "summary.json"
    summary_csv = out_root / "summary.csv"
    summary_rows = [
        {
            "case_id": item["case_id"],
            "image_dir": item["image_dir"],
            "prompt_label": item["prompt_label"],
            "used_fix256": item["used_fix256"],
            "status": item["status"],
            "message": item["message"],
            "token_count": item["token_count"],
            "word_count": item["word_count"],
            "rendered_word_count": item["rendered_word_count"],
            "layer_count": item["layer_count"],
            "raw_map_count": item["raw_map_count"],
            "expected_map_count": item["expected_map_count"],
            "missing_map_count": item["missing_map_count"],
            "overview_path": item["overview_path"],
            "metadata_path": item["metadata_path"],
        }
        for item in manifests
    ]
    write_json(summary_json, summary_rows)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)
    return summary_csv, summary_json


def write_index(out_root: Path, manifests: list[dict[str, Any]]) -> Path:
    rows = []
    prompt_rows = []
    for label in PROMPT_LABELS:
        prompt_rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(PROMPT_TEXTS.get(label, ''))}</td>"
            "</tr>"
        )
    for item in manifests:
        overview = Path(item["overview_path"])
        link = ""
        if overview.exists():
            link = f'<a href="{html_rel(overview, out_root)}">overview</a>'
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['image_dir'])}</td>"
            f"<td>{html.escape(item['case_id'])}</td>"
            f"<td>{html.escape(item['prompt_label'])}</td>"
            f"<td>{item['used_fix256']}</td>"
            f"<td>{html.escape(item['status'])}</td>"
            f"<td>{item['token_count']}</td>"
            f"<td>{item['word_count']}</td>"
            f"<td>{item['rendered_word_count']}</td>"
            f"<td>{item['layer_count']}</td>"
            f"<td>{link}</td>"
            "</tr>"
        )
    index_path = out_root / "index.html"
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>V3 all-layer TAM visualizations</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1400px; margin: 24px auto; line-height: 1.35; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border: 1px solid #ddd; padding: 6px 8px; }}
    th {{ background: #f3f3f3; }}
    .prompt-legend {{ margin: 18px 0 28px; }}
  </style>
</head>
<body>
  <h1>V3 all-layer TAM visualizations</h1>
  <p>Derived visualizations from existing Qwen2-VL TAM outputs. Global percentile normalization is per image/prompt, not across images or prompts.</p>
  <h2>Prompt legend</h2>
  <table class="prompt-legend">
    <thead>
      <tr><th>prompt label</th><th>prompt text</th></tr>
    </thead>
    <tbody>
      {''.join(prompt_rows)}
    </tbody>
  </table>
  <table>
    <thead>
      <tr>
        <th>image_dir</th><th>case_id</th><th>prompt</th><th>fix256</th><th>status</th>
        <th>tokens</th><th>words</th><th>rendered words</th><th>layers</th><th>overview</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    index_path.write_text(content, encoding="utf-8")
    return index_path


def parse_layers(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_percentiles(value: str) -> tuple[float, float]:
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 2 or parts[0] >= parts[1]:
        raise argparse.ArgumentTypeError("Expected two increasing percentiles, e.g. 1,99.5")
    return parts[0], parts[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render lightweight V3 all-layer TAM visualizations.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=Path("outputs/v3_visualization"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--image-filter", default=None)
    parser.add_argument("--prompt-filter", default=None)
    parser.add_argument("--max-words", type=int, default=60, help="Maximum words to render per case; use 0 for all words.")
    parser.add_argument("--thumb-size", type=int, default=112)
    parser.add_argument("--selected-word-count", type=int, default=8)
    parser.add_argument("--selected-layers", type=parse_layers, default=parse_layers("0,7,14,21,28"))
    parser.add_argument("--global-percentiles", type=parse_percentiles, default=parse_percentiles("1,99.5"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve() if args.project_root else project_root_from_here()
    run_root = args.run_root.resolve() if args.run_root else project_root / "outputs" / "prompt_sensitivity"
    out_root = args.out_root
    if not out_root.is_absolute():
        out_root = project_root / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    build_word_groups, token_pieces_from_metadata = import_word_utils(project_root)
    image_registry = load_image_registry(project_root)
    selections = discover_cases(run_root, args.image_filter, args.prompt_filter)
    print(f"[discover] cases={len(selections)} run_root={run_root}")

    manifests = []
    rendered = 0
    fix_resolved = 0
    inconsistent = []
    for selection in selections:
        if selection.used_fix256 and selection.metadata_path is not None:
            fix_resolved += 1
        print(f"[case] {selection.image_dir} / {selection.prompt_label} fix256={selection.used_fix256}")
        manifest = render_case(
            selection,
            project_root,
            out_root,
            args,
            image_registry,
            build_word_groups,
            token_pieces_from_metadata,
        )
        manifests.append(manifest)
        if manifest["status"] in {"validated", "inconsistent"} and Path(manifest["overview_path"]).exists():
            rendered += 1
        if manifest["status"] != "validated":
            inconsistent.append(manifest)

    summary_csv, summary_json = write_summary(out_root, manifests)
    manifest_path = out_root / "manifest.json"
    write_json(manifest_path, {"cases": manifests})
    index_path = write_index(out_root, manifests)

    expected = 40 if not args.image_filter and not args.prompt_filter else len(selections)
    print(f"[summary] discovered={len(selections)} expected={expected}")
    print(f"[summary] rendered={rendered}")
    print(f"[summary] fix256_resolved={fix_resolved}")
    print(f"[summary] inconsistent_or_missing={len(inconsistent)}")
    for item in inconsistent:
        print(f"[warn] {item['image_dir']} / {item['prompt_label']}: {item['status']} {item['message']}")
    print(f"[summary] summary_csv={summary_csv}")
    print(f"[summary] summary_json={summary_json}")
    print(f"[summary] manifest={manifest_path}")
    print(f"[summary] index={index_path}")
    return 0 if not inconsistent else 2


if __name__ == "__main__":
    raise SystemExit(main())
