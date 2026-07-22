from __future__ import annotations

import csv
import io
import json
import os
import re
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


EXPECTED_PROMPT_COUNTS = {
    "order_disruption_stress": 72,
    "colleague_obj_detection_hard": 28,
    "misleading_wrong_subject": 2,
}
EXPECTED_STATUS_COUNTS = {
    "target_strong": 50,
    "valid_other_object": 12,
    "background_or_wrong": 5,
    "ambiguous": 35,
}


class LocationValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocationValidationData:
    archive_path: Path
    cases: list[dict[str, str]]
    summary_rows: list[dict[str, str]]
    prompt_rows: list[dict[str, str]]
    category_rows: list[dict[str, str]]
    wrong_rows: list[dict[str, str]]
    manual_rows: list[dict[str, str]]
    label_rows: list[dict[str, str]]
    presentation_note: str
    validation_note: str
    interpretation_note: str
    by_case_id: dict[str, dict[str, str]]
    status_counts: dict[str, int]
    prompt_counts: dict[str, int]


def archive_path(project_root: Path) -> Path:
    override = os.environ.get("LOCATION_VALIDATION_ARCHIVE", "").strip()
    return Path(override) if override else project_root / "LORENZO_LOCATION_VALIDATION_102.zip"


def load_validation(project_root: Path) -> LocationValidationData:
    path = archive_path(project_root).resolve()
    if not path.exists():
        raise LocationValidationError("Location validation archive missing")
    stat = path.stat()
    return _load_validation_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _load_validation_cached(path_text: str, _mtime_ns: int, _size: int) -> LocationValidationData:
    path = Path(path_text)
    try:
        with zipfile.ZipFile(path) as zf:
            cases = _read_csv(zf, "data/location_validation_all_102.csv")
            summary_rows = _read_csv(zf, "data/location_validation_summary.csv")
            prompt_rows = _read_csv(zf, "data/location_validation_by_prompt.csv")
            category_rows = _read_csv(zf, "data/location_validation_by_category.csv")
            wrong_rows = _read_csv(zf, "data/wrong_or_background_cases.csv")
            manual_rows = _read_csv(zf, "data/manual_review_cases.csv")
            label_rows = _read_csv(zf, "data/label_mapping.csv")
            presentation_note = _read_text(zf, "PRESENTATION_CASE_376284.md")
            validation_note = _read_text(zf, "VALIDATION.md")
            interpretation_note = _read_text(zf, "LOCATION_VALIDATION_NOTES.md")
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise LocationValidationError("Location validation archive could not be read") from exc

    _validate_cases(cases)
    by_case_id = {row["case_id"]: row for row in cases}
    return LocationValidationData(
        archive_path=path,
        cases=cases,
        summary_rows=summary_rows,
        prompt_rows=prompt_rows,
        category_rows=category_rows,
        wrong_rows=wrong_rows,
        manual_rows=manual_rows,
        label_rows=label_rows,
        presentation_note=presentation_note,
        validation_note=validation_note,
        interpretation_note=interpretation_note,
        by_case_id=by_case_id,
        status_counts=_counts(cases, "review_status"),
        prompt_counts=_counts(cases, "prompt_label"),
    )


def _read_csv(zf: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    with zf.open(name) as handle:
        reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8-sig", newline=""))
        return [dict(row) for row in reader]


def _read_text(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8", errors="replace")


def _counts(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = row.get(column, "")
        out[key] = out.get(key, 0) + 1
    return out


def _validate_cases(cases: list[dict[str, str]]) -> None:
    if len(cases) != 102:
        raise LocationValidationError(f"Expected 102 validation rows, found {len(cases)}")
    case_ids = [row.get("case_id", "") for row in cases]
    if len(set(case_ids)) != len(case_ids):
        raise LocationValidationError("Location validation rows contain duplicate case_id values")
    prompt_counts = _counts(cases, "prompt_label")
    if prompt_counts != EXPECTED_PROMPT_COUNTS:
        raise LocationValidationError(f"Unexpected prompt counts: {prompt_counts}")
    status_counts = _counts(cases, "review_status")
    if status_counts != EXPECTED_STATUS_COUNTS:
        raise LocationValidationError(f"Unexpected status counts: {status_counts}")


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_box(value: str) -> tuple[float, float, float, float] | None:
    nums = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", str(value or ""))]
    if len(nums) < 4:
        return None
    return nums[0], nums[1], nums[2], nums[3]


def compact_text(value: str, limit: int = 110) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


def sort_cases(rows: list[dict[str, str]], key: str, desc: bool = False) -> list[dict[str, str]]:
    def sort_value(row: dict[str, str]) -> object:
        if key.endswith(("iou", "precision", "coverage", "ratio")):
            return safe_float(row.get(key), float("-inf"))
        return str(row.get(key, ""))

    return sorted(rows, key=sort_value, reverse=desc)


def status_label(status: str) -> str:
    return status.replace("_", " ")


@lru_cache(maxsize=1)
def load_coco(project_root_text: str) -> dict[str, Any]:
    path = Path(project_root_text) / "data" / "annotations" / "instances_val2017.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    categories = {int(row["id"]): row["name"] for row in payload.get("categories", [])}
    images = {int(row["id"]): row for row in payload.get("images", [])}
    by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in payload.get("annotations", []):
        image_id = int(ann["image_id"])
        x, y, w, h = [float(v) for v in ann["bbox"]]
        by_image.setdefault(image_id, []).append(
            {
                "id": int(ann["id"]),
                "image_id": image_id,
                "category_id": int(ann["category_id"]),
                "category": categories.get(int(ann["category_id"]), "unknown"),
                "bbox_xyxy": (x, y, x + w, y + h),
                "area": float(ann.get("area") or w * h),
                "iscrowd": int(ann.get("iscrowd") or 0),
                "segmentation": ann.get("segmentation"),
            }
        )
    return {"categories": categories, "images": images, "annotations_by_image": by_image}


def annotations_for_case(project_root: Path, case: dict[str, str]) -> list[dict[str, Any]]:
    data = load_coco(str(project_root.resolve()))
    return list(data["annotations_by_image"].get(int(case["image_id"]), []))


def render_validation_panel(project_root: Path, image_path: Path, case: dict[str, str], panel: str, options: dict[str, Any]) -> Image.Image:
    base = Image.open(image_path).convert("RGB")
    max_width = int(options.get("width") or 1000)
    original_size = base.size
    if max_width and base.width > max_width:
        base = base.resize((max_width, max(1, int(base.height * max_width / base.width))), Image.Resampling.LANCZOS)
    rendered_size = base.size
    if panel == "original":
        return base

    img = base.convert("RGBA")
    anns = [ann for ann in annotations_for_case(project_root, case) if not ann.get("iscrowd")]
    show_labels = boolish(options.get("labels", "1"))
    opacity = max(0, min(180, int(float(options.get("opacity") or 70))))
    model_column = "model_box_scaled_unclipped_xyxy" if boolish(options.get("unclipped")) else "model_box_clipped_xyxy"
    model_box = parse_box(case.get(model_column, ""))
    target_id = str(case.get("best_target_annotation_id") or "")
    any_id = str(case.get("best_any_annotation_id") or "")

    if panel in {"coco", "combined"} and boolish(options.get("masks", "1")):
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(overlay)
        for ann in anns:
            color = (170, 170, 170, max(18, opacity // 2))
            if str(ann["id"]) == target_id:
                color = (0, 180, 80, opacity)
            elif str(ann["id"]) == any_id and str(ann["id"]) != target_id:
                color = (40, 110, 230, opacity)
            _draw_segmentation(mask_draw, ann.get("segmentation"), original_size, rendered_size, color)
        img = Image.alpha_composite(img, overlay)

    draw = ImageDraw.Draw(img)
    if panel in {"coco", "combined"} and boolish(options.get("boxes", "1")):
        for ann in anns:
            color = (190, 190, 190, 255)
            width = 1
            if str(ann["id"]) == target_id:
                color, width = (0, 170, 80, 255), 4
            elif str(ann["id"]) == any_id and str(ann["id"]) != target_id:
                color, width = (30, 100, 230, 255), 4
            box = _scale_box(ann["bbox_xyxy"], original_size, rendered_size)
            draw.rectangle(box, outline=color, width=width)
            if show_labels:
                draw.text((box[0] + 3, max(2, box[1] + 3)), str(ann["category"]), fill=color, stroke_width=2, stroke_fill=(255, 255, 255, 220))

    if panel in {"model", "combined"} and model_box and boolish(options.get("model", "1")):
        box = _scale_box(model_box, original_size, rendered_size)
        draw.rectangle(box, outline=(230, 30, 30, 255), width=5)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=(230, 30, 30, 255), outline=(255, 255, 255, 255), width=2)
        if show_labels:
            draw.text((box[0] + 5, max(2, box[1] + 5)), f"model: {case.get('parsed_label')}", fill=(230, 30, 30, 255), stroke_width=2, stroke_fill=(255, 255, 255, 230))

    _draw_legend(draw, case, panel)
    return img.convert("RGB")


def _scale_box(box: tuple[float, float, float, float], original: tuple[int, int], rendered: tuple[int, int]) -> tuple[float, float, float, float]:
    sx = rendered[0] / max(original[0], 1)
    sy = rendered[1] / max(original[1], 1)
    return box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy


def _draw_segmentation(draw: ImageDraw.ImageDraw, segmentation: Any, original: tuple[int, int], rendered: tuple[int, int], color: tuple[int, int, int, int]) -> None:
    if isinstance(segmentation, list):
        sx = rendered[0] / max(original[0], 1)
        sy = rendered[1] / max(original[1], 1)
        for poly in segmentation:
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            points = [(float(poly[i]) * sx, float(poly[i + 1]) * sy) for i in range(0, len(poly) - 1, 2)]
            draw.polygon(points, fill=color)


def _draw_legend(draw: ImageDraw.ImageDraw, case: dict[str, str], panel: str) -> None:
    lines = [
        f"{panel}: red=model, green=target, blue=best other, gray=other COCO",
        f"status={case.get('review_status')} target IoU={_fmt(case.get('target_iou'))} any IoU={_fmt(case.get('best_any_iou'))}",
    ]
    width, height = 690, 48
    draw.rectangle((10, 10, width, height + 10), fill=(255, 255, 255, 225), outline=(30, 30, 30, 255), width=1)
    y = 17
    for line in lines:
        draw.text((18, y), line, fill=(0, 0, 0, 255))
        y += 18


def _fmt(value: Any) -> str:
    number = safe_float(value)
    return "n/a" if number is None else f"{number:.3f}"
