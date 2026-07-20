from __future__ import annotations

import json
import math
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.dashboard.cache import cache_path, get_cached, make_cache_key, put_cached
from scripts.dashboard.config import DashboardConfig, resolve_project_path
from scripts.dashboard.data_access import get_map_row, load_word_layer_map, row_paths
from scripts.dashboard.metrics import extract_regions, minmax, prob, weighted_centroid


PAIR_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
BOX_LIST_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


def font(size: int = 16):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def original_image(conn: sqlite3.Connection, config: DashboardConfig, case_id: str) -> Image.Image | None:
    row = conn.execute("SELECT image_path FROM cases WHERE case_id=?", (case_id,)).fetchone()
    if not row:
        return None
    path = resolve_project_path(row["image_path"], config.project_root)
    if not path.exists():
        return None
    return Image.open(path).convert("RGB")


def case_prompt_text(conn: sqlite3.Connection, config: DashboardConfig, case_id: str) -> str:
    row = conn.execute("SELECT metadata_path, prompt_label FROM cases WHERE case_id=?", (case_id,)).fetchone()
    if not row:
        return ""
    metadata_path = resolve_project_path(row["metadata_path"], config.project_root)
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            return str(payload.get("prompt_text") or payload.get("prompt") or row["prompt_label"] or "")
        except (OSError, json.JSONDecodeError):
            pass
    return str(row["prompt_label"] or "")


def compact_text(value: str, limit: int = 120) -> str:
    cleaned = " ".join(str(value).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def parse_model_locations(response_text: str) -> list[dict[str, object]]:
    """Extract coordinate-like locations from model text without treating plain "box" as a bbox."""
    text = str(response_text or "")
    locations: list[dict[str, object]] = []
    used_spans: list[tuple[int, int]] = []

    for match in BOX_LIST_RE.finditer(text):
        x0, y0, x1, y1 = (float(value) for value in match.groups())
        label = text[max(0, match.start() - 80) : match.start()].strip(" .,:;\n\t")
        label = re.sub(r".*?(?:^|\s)(?:a|an|the|there is|visible details in image)\s+", "", label, flags=re.IGNORECASE).strip()
        locations.append({"kind": "bbox", "label": label or "model bbox", "coords": (x0, y0, x1, y1), "raw": match.group(0)})
        used_spans.append(match.span())

    pairs = list(PAIR_RE.finditer(text))
    idx = 0
    while idx < len(pairs):
        first = pairs[idx]
        if any(start <= first.start() < end for start, end in used_spans):
            idx += 1
            continue
        if idx + 1 < len(pairs):
            second = pairs[idx + 1]
            between = text[first.end() : second.start()]
            if len(between) <= 12 and re.fullmatch(r"\s*,?\s*", between):
                x0, y0 = (float(value) for value in first.groups())
                x1, y1 = (float(value) for value in second.groups())
                label = text[max(0, first.start() - 80) : first.start()].strip(" .,:;\n\t")
                label = re.sub(r".*?(?:^|\s)(?:a|an|the|there is)\s+", "", label, flags=re.IGNORECASE).strip()
                locations.append(
                    {
                        "kind": "bbox",
                        "label": label or "model bbox",
                        "coords": (x0, y0, x1, y1),
                        "raw": f"{first.group(0)},{second.group(0)}",
                    }
                )
                idx += 2
                continue
        x, y = (float(value) for value in first.groups())
        label = text[max(0, first.start() - 80) : first.start()].strip(" .,:;\n\t") or "model point"
        locations.append({"kind": "point", "label": label, "coords": (x, y), "raw": first.group(0)})
        idx += 1
    return locations


@lru_cache(maxsize=1)
def load_coco_instances(annotation_path: str) -> dict[str, object]:
    path = Path(annotation_path)
    if not path.exists():
        return {"categories": {}, "annotations_by_image": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): str(item["name"]) for item in payload.get("categories", [])}
    annotations_by_image: dict[int, list[dict[str, object]]] = {}
    for ann in payload.get("annotations", []):
        if "image_id" not in ann or "bbox" not in ann:
            continue
        image_id = int(ann["image_id"])
        category = categories.get(int(ann.get("category_id", -1)), "unknown")
        x, y, w, h = [float(v) for v in ann["bbox"]]
        annotations_by_image.setdefault(image_id, []).append(
            {
                "category": category,
                "bbox_xywh": (x, y, w, h),
                "bbox_xyxy": (x, y, x + w, y + h),
                "area": float(ann.get("area") or w * h),
            }
        )
    return {"categories": categories, "annotations_by_image": annotations_by_image}


def _label_tokens(label: str) -> set[str]:
    stop = {"a", "an", "the", "there", "is", "visible", "details", "in", "image", "red", "blue", "white", "black", "small", "large"}
    tokens = {token for token in re.findall(r"[a-z0-9]+", label.lower()) if token not in stop}
    aliases = {
        "man": "person",
        "woman": "person",
        "boy": "person",
        "girl": "person",
        "people": "person",
        "motorbike": "motorcycle",
        "bike": "bicycle",
    }
    return {aliases.get(token, token) for token in tokens}


def matching_coco_annotations(annotation_path: Path, image_id: int | None, locations: list[dict[str, object]]) -> tuple[list[dict[str, object]], bool]:
    if image_id is None:
        return [], False
    data = load_coco_instances(str(annotation_path))
    by_image = data.get("annotations_by_image", {})
    anns = list(by_image.get(int(image_id), [])) if isinstance(by_image, dict) else []
    if not anns:
        return [], False
    location_tokens = set()
    for location in locations:
        location_tokens |= _label_tokens(str(location.get("label") or ""))
    matched = []
    for ann in anns:
        category = str(ann.get("category") or "")
        cat_tokens = _label_tokens(category)
        if cat_tokens.intersection(location_tokens):
            matched.append(ann)
    if matched:
        return sorted(matched, key=lambda ann: float(ann.get("area") or 0), reverse=True)[:12], True
    return sorted(anns, key=lambda ann: float(ann.get("area") or 0), reverse=True)[:8], False


def _scale_model_coordinate(value: float, image_size: int) -> float:
    if 0.0 <= value <= 1.0:
        return value * max(image_size - 1, 1)
    if 0.0 <= value <= 1000.0:
        return value / 1000.0 * max(image_size - 1, 1)
    return value


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _scaled_bbox(coords: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = coords
    sx0 = _scale_model_coordinate(x0, width)
    sy0 = _scale_model_coordinate(y0, height)
    sx1 = _scale_model_coordinate(x1, width)
    sy1 = _scale_model_coordinate(y1, height)
    left, right = sorted((_clip(sx0, 0, width - 1), _clip(sx1, 0, width - 1)))
    top, bottom = sorted((_clip(sy0, 0, height - 1), _clip(sy1, 0, height - 1)))
    return left, top, right, bottom


def _scale_coco_bbox(coords: tuple[float, float, float, float], original_size: tuple[int, int], rendered_size: tuple[int, int]) -> tuple[float, float, float, float]:
    original_w, original_h = original_size
    rendered_w, rendered_h = rendered_size
    x0, y0, x1, y1 = coords
    return (
        x0 * rendered_w / max(original_w, 1),
        y0 * rendered_h / max(original_h, 1),
        x1 * rendered_w / max(original_w, 1),
        y1 * rendered_h / max(original_h, 1),
    )


def render_model_location_overlay(
    conn: sqlite3.Connection,
    config: DashboardConfig,
    case_id: str,
    response_text: str,
    width: int = 1100,
    draw_coco_gt: bool = True,
) -> Path:
    base = original_image(conn, config, case_id)
    if base is None:
        raise ValueError("Original image missing")
    locations = parse_model_locations(response_text)
    case = conn.execute("SELECT image_id FROM cases WHERE case_id=?", (case_id,)).fetchone()
    image_id = int(case["image_id"]) if case and case["image_id"] is not None else None
    annotation_path = config.project_root / "data" / "annotations" / "instances_val2017.json"
    gt_annotations, gt_matched = matching_coco_annotations(annotation_path, image_id, locations) if draw_coco_gt else ([], False)
    params = {"case_id": case_id, "width": width, "response_text": response_text, "draw_coco_gt": draw_coco_gt}
    key = make_cache_key("model_location", params, json.dumps(params, sort_keys=True), config.cache_version)
    cached = get_cached(conn, config, key)
    if cached:
        return cached

    original_size = base.size
    image = base.copy()
    if width and image.width != width:
        height = max(1, int(image.height * width / image.width))
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    title_font = font(20)
    label_font = font(15)
    colors = [(255, 48, 48), (0, 176, 255), (255, 214, 0), (0, 210, 130), (230, 70, 255)]

    for ann in gt_annotations:
        x0, y0, x1, y1 = _scale_coco_bbox(ann["bbox_xyxy"], original_size, image.size)  # type: ignore[arg-type]
        gt_color = (20, 210, 80) if gt_matched else (40, 200, 220)
        draw.rectangle((x0, y0, x1, y1), outline=gt_color, width=3)
        draw.text((x0 + 5, max(82, y0 + 5)), f"GT {ann['category']}", fill=gt_color, font=label_font, stroke_width=2, stroke_fill=(0, 0, 0))

    for idx, location in enumerate(locations[:12], start=1):
        color = colors[(idx - 1) % len(colors)]
        label = compact_text(str(location.get("label") or f"location {idx}"), 42)
        if location["kind"] == "bbox":
            x0, y0, x1, y1 = _scaled_bbox(location["coords"], image.width, image.height)  # type: ignore[arg-type]
            draw.rectangle((x0, y0, x1, y1), outline=color, width=5)
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            r = 8
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color, outline=(0, 0, 0), width=2)
            draw.text((x0 + 6, max(4, y0 + 6)), f"{idx}. {label}", fill=color, font=label_font, stroke_width=2, stroke_fill=(0, 0, 0))
        else:
            x, y = location["coords"]  # type: ignore[misc]
            sx = _clip(_scale_model_coordinate(float(x), image.width), 0, image.width - 1)
            sy = _clip(_scale_model_coordinate(float(y), image.height), 0, image.height - 1)
            r = 10
            draw.ellipse((sx - r, sy - r, sx + r, sy + r), fill=color, outline=(0, 0, 0), width=2)
            draw.text((sx + 12, sy + 6), f"{idx}. {label}", fill=color, font=label_font, stroke_width=2, stroke_fill=(0, 0, 0))

    if not locations:
        draw_caption_box(image, ["No parseable model coordinates", "Plain words such as 'box' are not drawn as bbox."])
    else:
        draw.rectangle((12, 12, min(image.width - 12, 610), 74), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        draw.text((24, 22), "Model-described bbox / point", fill=(0, 0, 0), font=title_font)
        gt_text = "green=matched COCO GT" if gt_matched else ("cyan=COCO GT reference" if gt_annotations else "COCO GT unavailable")
        draw.text((24, 48), f"red/colored=model; dot=center/point; {gt_text}.", fill=(0, 0, 0), font=label_font)

    out = cache_path(config, key, ".jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, quality=92)
    put_cached(conn, config, key, out, "model_location", params, [], json.dumps(params, sort_keys=True))
    return out


def colorize_gray(
    arr: np.ndarray,
    size: int | None = None,
    normalization: str = "local",
    bounds: tuple[float, float] | None = None,
) -> Image.Image:
    if normalization == "global" and bounds is not None and bounds[1] > bounds[0]:
        value = np.clip((arr - bounds[0]) / max(bounds[1] - bounds[0], 1e-12), 0.0, 1.0)
    else:
        value = minmax(arr)
    img = Image.fromarray(np.clip(value * 255.0, 0, 255).astype(np.uint8), mode="L")
    if size:
        img = img.resize((size, size), Image.Resampling.BICUBIC)
    return img.convert("RGB")


def jet_overlay(
    base: Image.Image,
    arr: np.ndarray,
    alpha: float = 0.42,
    width: int = 1100,
    normalization: str = "local",
    bounds: tuple[float, float] | None = None,
) -> Image.Image:
    if normalization == "global" and bounds is not None and bounds[1] > bounds[0]:
        norm = np.clip((arr - bounds[0]) / max(bounds[1] - bounds[0], 1e-12), 0.0, 1.0)
    else:
        norm = minmax(arr)
    rgba = plt.get_cmap("jet")(norm)
    heat = Image.fromarray(np.clip(rgba[:, :, :3] * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    base = base.resize(heat.size, Image.Resampling.LANCZOS)
    blended = Image.blend(base, heat, alpha=alpha)
    if width and blended.width != width:
        height = max(1, int(blended.height * width / blended.width))
        blended = blended.resize((width, height), Image.Resampling.LANCZOS)
    return blended


def draw_regions(draw: ImageDraw.ImageDraw, regions, scale_x: float, scale_y: float) -> None:
    colors = [(255, 220, 0), (255, 255, 255), (0, 255, 255), (255, 80, 80), (80, 255, 80)]
    for region in regions:
        color = colors[(region.rank - 1) % len(colors)]
        x = region.centroid_x_px * scale_x
        y = region.centroid_y_px * scale_y
        r = 9 if region.rank == 1 else 7
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)
        draw.rectangle(
            (
                region.bbox_x0 * scale_x,
                region.bbox_y0 * scale_y,
                region.bbox_x1 * scale_x,
                region.bbox_y1 * scale_y,
            ),
            outline=color,
            width=2,
        )


def draw_caption_box(image: Image.Image, lines: list[str]) -> None:
    draw = ImageDraw.Draw(image)
    title_font = font(18)
    line_font = font(15)
    padding = 10
    line_heights = [22] + [19] * max(0, len(lines) - 1)
    width = min(image.width - 24, max(420, int(image.width * 0.62)))
    height = padding * 2 + sum(line_heights)
    draw.rectangle((12, 12, 12 + width, 12 + height), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    y = 12 + padding
    for idx, line in enumerate(lines):
        draw.text((22, y), line, fill=(0, 0, 0), font=title_font if idx == 0 else line_font)
        y += line_heights[idx]


def add_header(image: Image.Image, lines: list[str]) -> Image.Image:
    title_font = font(18)
    line_font = font(14)
    header_h = 18 + 24 + max(0, len(lines) - 1) * 18
    canvas = Image.new("RGB", (image.width, image.height + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    y = 8
    for idx, line in enumerate(lines):
        draw.text((12, y), line, fill=(0, 0, 0), font=title_font if idx == 0 else line_font)
        y += 24 if idx == 0 else 18
    canvas.paste(image, (0, header_h))
    return canvas


def render_map(
    conn: sqlite3.Connection,
    config: DashboardConfig,
    case_id: str,
    word_index: int,
    layer_index: int,
    mode: str = "overlay",
    threshold: float = 0.90,
    normalization: str = "local",
) -> Path:
    row = get_map_row(conn, case_id, word_index, layer_index)
    if row is None:
        raise ValueError("Map not found")
    sources = row_paths(row, config)
    prompt_text = case_prompt_text(conn, config, case_id)
    params = {
        "case_id": case_id,
        "word_index": word_index,
        "layer_index": layer_index,
        "mode": mode,
        "threshold": round(float(threshold), 4),
        "normalization": normalization,
    }
    key = make_cache_key("map", params, row["source_signature"], config.cache_version)
    cached = get_cached(conn, config, key)
    if cached:
        return cached
    arr = load_word_layer_map(sources)
    if arr is None:
        raise ValueError("Raw map missing")
    if mode == "gray":
        bounds = global_case_bounds(conn, case_id) if normalization == "global" else None
        image = colorize_gray(arr, size=1100, normalization=normalization, bounds=bounds)
    else:
        base = original_image(conn, config, case_id) or Image.new("RGB", (arr.shape[1], arr.shape[0]), "white")
        bounds = global_case_bounds(conn, case_id) if normalization == "global" else None
        image = jet_overlay(base, arr, normalization=normalization, bounds=bounds)
    draw = ImageDraw.Draw(image)
    regions = extract_regions(arr, threshold=threshold)
    draw_regions(draw, regions, image.width / arr.shape[1], image.height / arr.shape[0])
    cx, cy = weighted_centroid(prob(arr))
    if cx is not None and cy is not None:
        x = float(cx) * image.width / arr.shape[1]
        y = float(cy) * image.height / arr.shape[0]
        draw.line((x - 14, y, x + 14, y), fill=(0, 0, 0), width=3)
        draw.line((x, y - 14, x, y + 14), fill=(0, 0, 0), width=3)
    title = "Locally normalized heatmap" if mode == "gray" else "Heatmap overlay"
    image = add_header(
        image,
        [
            title,
            f"prompt: {compact_text(prompt_text)}",
            f"selected word={word_index}  selected layer=L{layer_index}",
            f"normalization={normalization}  threshold={threshold:.2f}",
            "black cross=global centroid; yellow/white boxes=salient regions",
        ],
    )
    out = cache_path(config, key, ".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    put_cached(conn, config, key, out, f"map_{mode}", params, sources, row["source_signature"])
    return out


def render_scanpath(
    conn: sqlite3.Connection,
    config: DashboardConfig,
    case_id: str,
    mode: str,
    layer_index: int | None = None,
    word_index: int | None = None,
    threshold: float = 0.90,
) -> Path:
    params = {"case_id": case_id, "mode": mode, "layer_index": layer_index, "word_index": word_index, "threshold": threshold}
    key = make_cache_key("scanpath", params, json.dumps(params), config.cache_version)
    cached = get_cached(conn, config, key)
    if cached:
        return cached
    base = original_image(conn, config, case_id) or Image.new("RGB", (900, 650), "white")
    width = 1100
    height = max(1, int(base.height * width / base.width))
    image = base.resize((width, height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    points = []
    labels = []
    if mode == "word":
        rows = conn.execute(
            """
            SELECT w.word_index, w.word_label, m.global_centroid_x_norm, m.global_centroid_y_norm
            FROM words w
            JOIN map_metrics m ON w.case_id=m.case_id AND w.word_index=m.word_index
            WHERE w.case_id=? AND m.layer_index=?
            ORDER BY w.word_index
            """,
            (case_id, int(layer_index or 0)),
        )
    else:
        rows = conn.execute(
            """
            SELECT m.layer_index AS word_index, CAST(m.layer_index AS TEXT) AS word_label, m.global_centroid_x_norm, m.global_centroid_y_norm
            FROM map_metrics m
            WHERE m.case_id=? AND m.word_index=?
            ORDER BY m.layer_index
            """,
            (case_id, int(word_index or 0)),
        )
    for row in rows:
        if row["global_centroid_x_norm"] is None or row["global_centroid_y_norm"] is None:
            continue
        x = float(row["global_centroid_x_norm"]) * width
        y = float(row["global_centroid_y_norm"]) * height
        points.append((x, y))
        labels.append(str(row["word_label"]))
    if len(points) >= 2:
        draw.line(points, fill=(255, 255, 255), width=4)
        draw.line(points, fill=(0, 0, 0), width=2)
    label_font = font(15)
    for idx, (x, y) in enumerate(points):
        r = 7 if idx < len(points) - 1 else 11
        fill = (255, 255, 255) if idx < len(points) - 1 else (255, 220, 0)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=(0, 0, 0), width=2)
        if idx == 0 or idx == len(points) - 1 or idx % 10 == 0:
            draw.text((x + 8, y + 4), labels[idx][:20], fill=(0, 0, 0), font=label_font)
    if mode == "word":
        title = f"Word-wise scanpath at selected layer L{int(layer_index or 0)}"
        detail = "line follows centroids over generated words"
    else:
        title = f"Layer-wise scanpath for selected word {int(word_index or 0)}"
        detail = "line follows centroids over layers"
    image = add_header(
        image,
        [
            title,
            detail,
            "white line=path; yellow point=current/end point",
            f"threshold={threshold:.2f}",
        ],
    )
    out = cache_path(config, key, ".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    put_cached(conn, config, key, out, f"scanpath_{mode}", params, [], json.dumps(params))
    return out


def render_final_layer_preview(
    conn: sqlite3.Connection,
    config: DashboardConfig,
    case_id: str,
    max_words: int = 12,
    threshold: float = 0.90,
) -> Path:
    layers = [row["layer_index"] for row in conn.execute("SELECT DISTINCT layer_index FROM maps WHERE case_id=? ORDER BY layer_index", (case_id,))]
    if not layers:
        raise ValueError("No layers for case")
    final_layer = int(layers[-1])
    row_count = conn.execute("SELECT COUNT(*) AS n FROM words WHERE case_id=?", (case_id,)).fetchone()["n"]
    params = {"case_id": case_id, "max_words": max_words, "threshold": threshold, "final_layer": final_layer, "word_count": row_count}
    key = make_cache_key("final_layer_preview", params, json.dumps(params), config.cache_version)
    cached = get_cached(conn, config, key)
    if cached:
        return cached
    word_rows = list(conn.execute("SELECT word_index, word_label FROM words WHERE case_id=? ORDER BY word_index", (case_id,)))
    if not word_rows:
        raise ValueError("No words for case")
    if len(word_rows) > max_words:
        positions = sorted({round(i * (len(word_rows) - 1) / max(max_words - 1, 1)) for i in range(max_words)})
        word_rows = [word_rows[pos] for pos in positions]
    base = original_image(conn, config, case_id)
    thumbs = []
    label_font = font(13)
    for word in word_rows:
        map_row = get_map_row(conn, case_id, int(word["word_index"]), final_layer)
        if not map_row:
            continue
        arr = load_word_layer_map(row_paths(map_row, config))
        if arr is None:
            continue
        tile_base = base or Image.new("RGB", (arr.shape[1], arr.shape[0]), "white")
        tile = jet_overlay(tile_base, arr, width=260, normalization="local")
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, tile.width, 28), fill=(255, 255, 255), outline=(0, 0, 0))
        draw.text((6, 6), f"w{word['word_index']} {str(word['word_label'])[:22]}", fill=(0, 0, 0), font=label_font)
        thumbs.append(tile)
    if not thumbs:
        raise ValueError("No preview maps available")
    cols = min(4, len(thumbs))
    rows = math.ceil(len(thumbs) / cols)
    gap = 8
    title_h = 58
    width = cols * thumbs[0].width + (cols + 1) * gap
    height = title_h + rows * thumbs[0].height + (rows + 1) * gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 8), f"Final-layer TAM preview, L{final_layer}", fill=(0, 0, 0), font=font(18))
    draw.text((gap, 32), "Sampled generated words; raw maps rendered as TAM-style overlays.", fill=(40, 40, 40), font=font(14))
    for idx, tile in enumerate(thumbs):
        x = gap + (idx % cols) * (tile.width + gap)
        y = title_h + gap + (idx // cols) * (tile.height + gap)
        sheet.paste(tile, (x, y))
    out = cache_path(config, key, ".jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)
    put_cached(conn, config, key, out, "final_layer_preview", params, [], json.dumps(params))
    return out


def render_final_layer_animation(
    conn: sqlite3.Connection,
    config: DashboardConfig,
    case_id: str,
    max_words: int = 32,
    duration_ms: int = 450,
) -> Path:
    layers = [row["layer_index"] for row in conn.execute("SELECT DISTINCT layer_index FROM maps WHERE case_id=? ORDER BY layer_index", (case_id,))]
    if not layers:
        raise ValueError("No layers for case")
    final_layer = int(layers[-1])
    word_rows = list(conn.execute("SELECT word_index, word_label FROM words WHERE case_id=? ORDER BY word_index", (case_id,)))
    if not word_rows:
        raise ValueError("No words for case")
    if len(word_rows) > max_words:
        positions = sorted({round(i * (len(word_rows) - 1) / max(max_words - 1, 1)) for i in range(max_words)})
        word_rows = [word_rows[pos] for pos in positions]
    params = {"case_id": case_id, "max_words": max_words, "duration_ms": duration_ms, "final_layer": final_layer, "word_count": len(word_rows)}
    key = make_cache_key("final_layer_animation", params, json.dumps(params), config.cache_version)
    cached = get_cached(conn, config, key)
    if cached:
        return cached
    base = original_image(conn, config, case_id)
    frames = []
    recent_points: list[tuple[float, float]] = []
    for word in word_rows:
        map_row = get_map_row(conn, case_id, int(word["word_index"]), final_layer)
        if not map_row:
            continue
        arr = load_word_layer_map(row_paths(map_row, config))
        if arr is None:
            continue
        tile_base = base or Image.new("RGB", (arr.shape[1], arr.shape[0]), "white")
        frame = jet_overlay(tile_base, arr, width=720, normalization="local")
        regions = extract_regions(arr, threshold=0.90)
        if regions:
            cx = regions[0].centroid_x_px
            cy = regions[0].centroid_y_px
        else:
            cx, cy = weighted_centroid(prob(arr))
        if cx is not None and cy is not None:
            recent_points.append((float(cx) * frame.width / arr.shape[1], float(cy) * frame.height / arr.shape[0]))
            recent_points = recent_points[-3:]
        draw = ImageDraw.Draw(frame)
        if len(recent_points) >= 2:
            draw.line(recent_points, fill=(255, 255, 255), width=6)
            draw.line(recent_points, fill=(0, 0, 0), width=3)
        for idx, (x, y) in enumerate(recent_points):
            is_current = idx == len(recent_points) - 1
            radius = 12 if is_current else 8
            fill = (255, 220, 0) if is_current else (255, 255, 255)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=(0, 0, 0), width=3)
            if is_current:
                draw.line((x - 16, y, x + 16, y), fill=(0, 0, 0), width=2)
                draw.line((x, y - 16, x, y + 16), fill=(0, 0, 0), width=2)
        frame = add_header(
            frame,
            [
                f"Final-layer TAM over generated words, L{final_layer}",
                f"word {word['word_index']}: {str(word['word_label'])[:48]} | yellow=current, white=previous 2",
            ],
        )
        frames.append(frame)
    if not frames:
        raise ValueError("No animation frames available")
    out = cache_path(config, key, ".gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    put_cached(conn, config, key, out, "final_layer_animation", params, [], json.dumps(params))
    return out


def render_matrix_cell(
    conn: sqlite3.Connection,
    config: DashboardConfig,
    case_id: str,
    word_index: int,
    layer_index: int,
    normalization: str = "local",
    size: int = 120,
) -> Path:
    row = get_map_row(conn, case_id, word_index, layer_index)
    if row is None:
        raise ValueError("Map not found")
    sources = row_paths(row, config)
    params = {"case_id": case_id, "word_index": word_index, "layer_index": layer_index, "normalization": normalization, "size": size}
    key = make_cache_key("matrix_cell", params, row["source_signature"], config.cache_version)
    cached = get_cached(conn, config, key)
    if cached:
        return cached
    arr = load_word_layer_map(sources)
    if arr is None:
        raise ValueError("Raw map missing")
    out = cache_path(config, key, ".jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    bounds = global_case_bounds(conn, case_id) if normalization == "global" else None
    colorize_gray(arr, size=size, normalization=normalization, bounds=bounds).save(out, quality=88)
    put_cached(conn, config, key, out, "matrix_cell", params, sources, row["source_signature"])
    return out


def render_difference(
    conn: sqlite3.Connection,
    config: DashboardConfig,
    case_id_a: str,
    case_id_b: str,
    word_index: int,
    layer_index: int,
    mode: str = "absolute",
) -> Path:
    row_a = get_map_row(conn, case_id_a, word_index, layer_index)
    row_b = get_map_row(conn, case_id_b, word_index, layer_index)
    if row_a is None or row_b is None:
        raise ValueError("Map pair not found")
    sources_a = row_paths(row_a, config)
    sources_b = row_paths(row_b, config)
    source_sig = f"{row_a['source_signature']}|{row_b['source_signature']}"
    params = {
        "case_id_a": case_id_a,
        "case_id_b": case_id_b,
        "word_index": word_index,
        "layer_index": layer_index,
        "mode": mode,
    }
    key = make_cache_key("difference", params, source_sig, config.cache_version)
    cached = get_cached(conn, config, key)
    if cached:
        return cached
    arr_a = load_word_layer_map(sources_a)
    arr_b = load_word_layer_map(sources_b)
    if arr_a is None or arr_b is None:
        raise ValueError("Raw map missing")
    if arr_a.shape != arr_b.shape:
        arr_b = np.asarray(Image.fromarray(arr_b.astype(np.float32), mode="F").resize((arr_a.shape[1], arr_a.shape[0]), Image.BILINEAR))
    norm_a = minmax(arr_a)
    norm_b = minmax(arr_b)
    if mode == "signed":
        diff = norm_b - norm_a
        scaled = np.clip((diff + 1.0) * 0.5, 0.0, 1.0)
        rgba = plt.get_cmap("coolwarm")(scaled)
    else:
        diff = np.abs(norm_b - norm_a)
        rgba = plt.get_cmap("magma")(diff)
    image = Image.fromarray(np.clip(rgba[:, :, :3] * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    width = 1100
    height = max(1, int(image.height * width / image.width))
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(image)
    if mode == "signed":
        lines = [
            "Signed difference map",
            f"word={word_index}  layer=L{layer_index}",
            "red=B stronger; blue=A stronger; white=similar",
        ]
    else:
        lines = [
            "Absolute difference map",
            f"word={word_index}  layer=L{layer_index}",
            "brighter = larger normalized heatmap difference",
        ]
    image = add_header(image, lines)
    out = cache_path(config, key, ".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    put_cached(conn, config, key, out, f"difference_{mode}", params, [*sources_a, *sources_b], source_sig)
    return out


def global_case_bounds(conn: sqlite3.Connection, case_id: str) -> tuple[float, float] | None:
    row = conn.execute(
        "SELECT MIN(min_value) AS low, MAX(max_value) AS high FROM map_metrics WHERE case_id=?",
        (case_id,),
    ).fetchone()
    if not row or row["low"] is None or row["high"] is None:
        return None
    return float(row["low"]), float(row["high"])
