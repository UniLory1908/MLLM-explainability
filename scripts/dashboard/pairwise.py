from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from scripts.dashboard.config import DashboardConfig
from scripts.dashboard.data_access import load_word_layer_map, row_paths
from scripts.dashboard.metrics import (
    clean,
    cosine_similarity,
    extract_regions,
    hausdorff_peak_distance,
    hotspot_iou,
    jsd,
    minmax,
    pearson_correlation,
    prob,
    top_iou,
    weighted_centroid,
)

try:  # optional
    from skimage.metrics import structural_similarity
except Exception:  # pragma: no cover
    structural_similarity = None

try:  # optional
    import ot
except Exception:  # pragma: no cover
    ot = None


def pair_id(params: dict) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()


def resize_same(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape == b.shape:
        return a, b
    img = Image.fromarray(b.astype(np.float32), mode="F")
    resized = np.asarray(img.resize((a.shape[1], a.shape[0]), Image.BILINEAR), dtype=np.float64)
    return a, resized


def ssim_metric(a: np.ndarray, b: np.ndarray) -> float | None:
    if structural_similarity is None:
        return None
    av, bv = resize_same(minmax(a), minmax(b))
    if av.shape[0] < 7 or av.shape[1] < 7:
        return None
    return float(structural_similarity(av, bv, data_range=1.0))


def emd_metric(a: np.ndarray, b: np.ndarray, grid_size: int = 32) -> float | None:
    if ot is None:
        return None
    pa = downsample_prob(a, grid_size)
    pb = downsample_prob(b, grid_size)
    if pa.sum() <= 0 or pb.sum() <= 0:
        return None
    coords = np.array([(y, x) for y in range(grid_size) for x in range(grid_size)], dtype=np.float64)
    cost = ot.dist(coords, coords, metric="euclidean")
    cost /= max(float(cost.max()), 1.0)
    return float(ot.emd2(pa.ravel(), pb.ravel(), cost))


def downsample_prob(arr: np.ndarray, grid_size: int) -> np.ndarray:
    value = clean(arr)
    img = Image.fromarray(value.astype(np.float32), mode="F")
    small = np.asarray(img.resize((grid_size, grid_size), Image.BILINEAR), dtype=np.float64)
    total = float(small.sum())
    if total <= 1e-12:
        return np.zeros_like(small)
    return small / total


def radial_profile_distance(a: np.ndarray, b: np.ndarray, bins: int = 24) -> float | None:
    pa, pb = resize_same(prob(a), prob(b))
    cx, cy = weighted_centroid(pa)
    if cx is None or cy is None:
        return None
    yy, xx = np.indices(pa.shape)
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_r = float(radius.max())
    if max_r <= 0:
        return None
    edges = np.linspace(0.0, max_r, bins + 1)
    prof_a = np.zeros(bins, dtype=np.float64)
    prof_b = np.zeros(bins, dtype=np.float64)
    for idx in range(bins):
        mask = (radius >= edges[idx]) & (radius < edges[idx + 1])
        prof_a[idx] = pa[mask].sum()
        prof_b[idx] = pb[mask].sum()
    return float(np.linalg.norm(prof_a - prof_b))


def compute_pairwise(a: np.ndarray, b: np.ndarray, include_emd: bool = False) -> dict[str, float | None]:
    a, b = resize_same(a, b)
    ca = weighted_centroid(prob(a))
    cb = weighted_centroid(prob(b))
    centroid_shift = None
    if ca[0] is not None and cb[0] is not None:
        centroid_shift = float(math.hypot(float(ca[0]) - float(cb[0]), float(ca[1]) - float(cb[1])))
    ra = extract_regions(a, threshold=0.90)
    rb = extract_regions(b, threshold=0.90)
    primary_shift = None
    if ra and rb:
        primary_shift = float(math.hypot(ra[0].centroid_x_px - rb[0].centroid_x_px, ra[0].centroid_y_px - rb[0].centroid_y_px))
    return {
        "cosine_similarity": cosine_similarity(a, b),
        "pearson_correlation": pearson_correlation(a, b),
        "ssim": ssim_metric(a, b),
        "jsd": jsd(a, b),
        "emd_2d": emd_metric(a, b) if include_emd else None,
        "l1_distance": float(np.abs(clean(a) - clean(b)).sum()),
        "l2_distance": float(np.linalg.norm(clean(a) - clean(b))),
        "top_1_iou": top_iou(a, b, 1.0),
        "top_5_iou": top_iou(a, b, 5.0),
        "top_10_iou": top_iou(a, b, 10.0),
        "hotspot_iou_percentile_90": hotspot_iou(a, b, 90.0),
        "hotspot_iou_percentile_95": hotspot_iou(a, b, 95.0),
        "hausdorff_peak_distance": hausdorff_peak_distance(a, b),
        "argmax_distance": argmax_distance(a, b),
        "global_centroid_shift": centroid_shift,
        "primary_centroid_shift": primary_shift,
        "spread_delta": None,
        "anisotropy_delta": None,
        "radial_profile_distance": radial_profile_distance(a, b),
    }


def argmax_distance(a: np.ndarray, b: np.ndarray) -> float | None:
    av = clean(a)
    bv = clean(b)
    if av.size == 0 or bv.size == 0:
        return None
    ay, ax = np.unravel_index(int(np.argmax(av)), av.shape)
    by, bx = np.unravel_index(int(np.argmax(bv)), bv.shape)
    return float(math.hypot(ax - bx, ay - by))


def compute_pair_for_rows(conn: sqlite3.Connection, config: DashboardConfig, row_a, row_b, include_emd: bool = False) -> dict:
    arr_a = load_word_layer_map(row_paths(row_a, config))
    arr_b = load_word_layer_map(row_paths(row_b, config))
    if arr_a is None or arr_b is None:
        raise ValueError("Missing map for pairwise comparison")
    metrics = compute_pairwise(arr_a, arr_b, include_emd=include_emd)
    params = {
        "a": [row_a["case_id"], row_a["word_index"], row_a["layer_index"]],
        "b": [row_b["case_id"], row_b["word_index"], row_b["layer_index"]],
        "include_emd": include_emd,
    }
    pid = pair_id(params)
    conn.execute(
        """
        INSERT OR REPLACE INTO map_pairs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            pid,
            row_a["case_id"],
            row_a["word_index"],
            row_a["layer_index"],
            row_b["case_id"],
            row_b["word_index"],
            row_b["layer_index"],
            "default_emd" if include_emd else "default",
            metrics["cosine_similarity"],
            metrics["pearson_correlation"],
            metrics["ssim"],
            metrics["jsd"],
            metrics["emd_2d"],
            metrics["l1_distance"],
            metrics["l2_distance"],
            metrics["top_1_iou"],
            metrics["top_5_iou"],
            metrics["top_10_iou"],
            metrics["hotspot_iou_percentile_90"],
            metrics["hotspot_iou_percentile_95"],
            metrics["hausdorff_peak_distance"],
            metrics["argmax_distance"],
            metrics["global_centroid_shift"],
            metrics["primary_centroid_shift"],
            metrics["spread_delta"],
            metrics["anisotropy_delta"],
            metrics["radial_profile_distance"],
            f"{row_a['source_signature']}|{row_b['source_signature']}",
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    return {"pair_id": pid, **metrics}
