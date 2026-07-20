from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff


EPS = 1e-12


@dataclass
class Region:
    threshold: float
    rank: int
    mass: float
    ratio_to_primary: float | None
    centroid_x_px: float
    centroid_y_px: float
    centroid_x_norm: float
    centroid_y_norm: float
    bbox_x0: int
    bbox_y0: int
    bbox_x1: int
    bbox_y1: int
    bbox_x0_norm: float
    bbox_y0_norm: float
    bbox_x1_norm: float
    bbox_y1_norm: float
    area: int
    peak_value: float


def clean(arr: np.ndarray) -> np.ndarray:
    value = np.asarray(arr, dtype=np.float64)
    value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(value, 0.0)


def minmax(arr: np.ndarray) -> np.ndarray:
    value = clean(arr)
    low = float(value.min()) if value.size else 0.0
    high = float(value.max()) if value.size else 0.0
    if high <= low:
        return np.zeros_like(value, dtype=np.float64)
    return (value - low) / (high - low)


def prob(arr: np.ndarray) -> np.ndarray:
    value = clean(arr)
    total = float(value.sum())
    if total <= EPS:
        return np.zeros_like(value, dtype=np.float64)
    return value / total


def zscore(arr: np.ndarray) -> np.ndarray:
    value = clean(arr)
    std = float(value.std())
    if std <= EPS:
        return np.zeros_like(value, dtype=np.float64)
    return (value - float(value.mean())) / std


def top_mass(p: np.ndarray, percent: float) -> float:
    flat = np.sort(p.ravel())[::-1]
    count = max(1, int(math.ceil(flat.size * percent / 100.0)))
    return float(flat[:count].sum())


def weighted_centroid(p: np.ndarray) -> tuple[float | None, float | None]:
    total = float(p.sum())
    if total <= EPS:
        return None, None
    yy, xx = np.indices(p.shape)
    return float((xx * p).sum() / total), float((yy * p).sum() / total)


def spatial_spread(p: np.ndarray, cx: float | None, cy: float | None) -> dict[str, float | None]:
    if cx is None or cy is None or float(p.sum()) <= EPS:
        return {
            "spread_trace": None,
            "spread_x": None,
            "spread_y": None,
            "covariance_xy": None,
            "anisotropy": None,
        }
    yy, xx = np.indices(p.shape)
    dx = xx - cx
    dy = yy - cy
    var_x = float((p * dx * dx).sum())
    var_y = float((p * dy * dy).sum())
    cov_xy = float((p * dx * dy).sum())
    cov = np.array([[var_x, cov_xy], [cov_xy, var_y]], dtype=np.float64)
    eigvals = np.linalg.eigvalsh(cov)
    largest = float(max(eigvals))
    smallest = float(min(eigvals))
    anisotropy = None if largest <= EPS else float(1.0 - smallest / largest)
    return {
        "spread_trace": var_x + var_y,
        "spread_x": var_x,
        "spread_y": var_y,
        "covariance_xy": cov_xy,
        "anisotropy": anisotropy,
    }


def extract_regions(arr: np.ndarray, threshold: float = 0.90, top_k: int = 5, min_area: int = 8) -> list[Region]:
    norm = minmax(arr)
    if norm.size == 0:
        return []
    mask = norm >= float(threshold)
    if not np.any(mask):
        return []
    labels, count = ndimage.label(mask)
    p = prob(arr)
    regions = []
    height, width = norm.shape
    for label in range(1, count + 1):
        ys, xs = np.where(labels == label)
        area = int(xs.size)
        if area < min_area:
            continue
        weights = p[ys, xs]
        mass = float(weights.sum())
        if mass <= EPS:
            continue
        cx = float((xs * weights).sum() / mass)
        cy = float((ys * weights).sum() / mass)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        regions.append(
            {
                "mass": mass,
                "centroid_x_px": cx,
                "centroid_y_px": cy,
                "bbox_x0": x0,
                "bbox_y0": y0,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "area": area,
                "peak_value": float(norm[ys, xs].max()),
            }
        )
    regions.sort(key=lambda item: (item["mass"], item["area"]), reverse=True)
    primary_mass = regions[0]["mass"] if regions else None
    result = []
    for idx, item in enumerate(regions[:top_k], start=1):
        result.append(
            Region(
                threshold=float(threshold),
                rank=idx,
                mass=float(item["mass"]),
                ratio_to_primary=None if not primary_mass else float(item["mass"] / primary_mass),
                centroid_x_px=float(item["centroid_x_px"]),
                centroid_y_px=float(item["centroid_y_px"]),
                centroid_x_norm=float(item["centroid_x_px"] / max(width - 1, 1)),
                centroid_y_norm=float(item["centroid_y_px"] / max(height - 1, 1)),
                bbox_x0=int(item["bbox_x0"]),
                bbox_y0=int(item["bbox_y0"]),
                bbox_x1=int(item["bbox_x1"]),
                bbox_y1=int(item["bbox_y1"]),
                bbox_x0_norm=float(item["bbox_x0"] / max(width - 1, 1)),
                bbox_y0_norm=float(item["bbox_y0"] / max(height - 1, 1)),
                bbox_x1_norm=float(item["bbox_x1"] / max(width - 1, 1)),
                bbox_y1_norm=float(item["bbox_y1"] / max(height - 1, 1)),
                area=int(item["area"]),
                peak_value=float(item["peak_value"]),
            )
        )
    return result


def per_map_metrics(arr: np.ndarray, regions_90: list[Region] | None = None) -> dict[str, float | int | None]:
    value = clean(arr)
    p = prob(value)
    height, width = value.shape
    energy = float(value.sum())
    nonzero = int(np.count_nonzero(value))
    flat_p = p.ravel()
    positive_p = flat_p[flat_p > 0]
    entropy = None
    if positive_p.size:
        entropy = float(-(positive_p * np.log(positive_p)).sum() / math.log(max(flat_p.size, 2)))
    hhi = float((flat_p * flat_p).sum())
    effective_area = None if hhi <= EPS else float(1.0 / hhi)
    l1 = float(np.abs(value).sum())
    l2 = float(np.sqrt((value * value).sum()))
    n = value.size
    hoyer = None
    if n > 1 and l2 > EPS:
        hoyer = float((math.sqrt(n) - l1 / l2) / (math.sqrt(n) - 1.0))
    cx, cy = weighted_centroid(p)
    spread = spatial_spread(p, cx, cy)
    regions_90 = regions_90 or []
    primary = regions_90[0] if regions_90 else None
    secondary = regions_90[1] if len(regions_90) > 1 else None
    return {
        "energy_sum": energy,
        "energy_mean": float(value.mean()) if value.size else None,
        "min_value": float(value.min()) if value.size else None,
        "max_value": float(value.max()) if value.size else None,
        "std_value": float(value.std()) if value.size else None,
        "nonzero_ratio": float(nonzero / value.size) if value.size else None,
        "entropy_norm": entropy,
        "top_1_mass": top_mass(p, 1.0),
        "top_5_mass": top_mass(p, 5.0),
        "top_10_mass": top_mass(p, 10.0),
        "hhi": hhi,
        "effective_area": effective_area,
        "effective_area_norm": None if effective_area is None else float(effective_area / max(value.size, 1)),
        "hoyer_sparsity": hoyer,
        "global_centroid_x_px": cx,
        "global_centroid_y_px": cy,
        "global_centroid_x_norm": None if cx is None else float(cx / max(width - 1, 1)),
        "global_centroid_y_norm": None if cy is None else float(cy / max(height - 1, 1)),
        **spread,
        "peak_count": len(regions_90),
        "primary_region_mass": None if primary is None else primary.mass,
        "secondary_region_mass": None if secondary is None else secondary.mass,
        "secondary_primary_ratio": None if secondary is None else secondary.ratio_to_primary,
        "primary_region_centroid_x_px": None if primary is None else primary.centroid_x_px,
        "primary_region_centroid_y_px": None if primary is None else primary.centroid_y_px,
        "primary_region_centroid_x_norm": None if primary is None else primary.centroid_x_norm,
        "primary_region_centroid_y_norm": None if primary is None else primary.centroid_y_norm,
        "secondary_region_centroid_x_norm": None if secondary is None else secondary.centroid_x_norm,
        "secondary_region_centroid_y_norm": None if secondary is None else secondary.centroid_y_norm,
        "is_multipeak": 1 if len(regions_90) > 1 else 0,
    }


def path_metrics(points: list[tuple[float, float]], large_jump_threshold: float = 0.15) -> dict[str, float | int | None]:
    valid = [(float(x), float(y)) for x, y in points if x is not None and y is not None]
    if len(valid) < 2:
        return {
            "path_length": 0.0,
            "mean_step": None,
            "max_jump": None,
            "net_displacement": 0.0,
            "tortuosity": None,
            "large_jump_count": 0,
            "bbox_area": 0.0,
            "revisit_count": 0,
        }
    dists = [float(math.hypot(valid[i][0] - valid[i - 1][0], valid[i][1] - valid[i - 1][1])) for i in range(1, len(valid))]
    path_len = float(sum(dists))
    net = float(math.hypot(valid[-1][0] - valid[0][0], valid[-1][1] - valid[0][1]))
    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    rounded = [(round(x, 2), round(y, 2)) for x, y in valid]
    return {
        "path_length": path_len,
        "mean_step": float(np.mean(dists)),
        "max_jump": float(max(dists)),
        "net_displacement": net,
        "tortuosity": None if net <= EPS else float(path_len / net),
        "large_jump_count": int(sum(dist >= large_jump_threshold for dist in dists)),
        "bbox_area": float((max(xs) - min(xs)) * (max(ys) - min(ys))),
        "revisit_count": int(len(rounded) - len(set(rounded))),
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
    av = clean(a).ravel()
    bv = clean(b).ravel()
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= EPS:
        return None
    return float(np.dot(av, bv) / denom)


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    av = zscore(a).ravel()
    bv = zscore(b).ravel()
    if av.size != bv.size or av.size == 0:
        return None
    return float(np.mean(av * bv))


def jsd(a: np.ndarray, b: np.ndarray) -> float | None:
    pa = prob(a).ravel()
    pb = prob(b).ravel()
    if pa.sum() <= EPS or pb.sum() <= EPS:
        return None
    m = 0.5 * (pa + pb)
    div = 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)
    return float(div)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float((p[mask] * np.log(p[mask] / np.maximum(q[mask], EPS))).sum())


def top_iou(a: np.ndarray, b: np.ndarray, percent: float) -> float | None:
    am = top_mask(prob(a), percent)
    bm = top_mask(prob(b), percent)
    union = np.logical_or(am, bm).sum()
    if union == 0:
        return None
    return float(np.logical_and(am, bm).sum() / union)


def top_mask(p: np.ndarray, percent: float) -> np.ndarray:
    flat = p.ravel()
    count = max(1, int(math.ceil(flat.size * percent / 100.0)))
    if count >= flat.size:
        return np.ones_like(p, dtype=bool)
    threshold = np.partition(flat, -count)[-count]
    return p >= threshold


def hotspot_iou(a: np.ndarray, b: np.ndarray, percentile: float) -> float | None:
    av = clean(a)
    bv = clean(b)
    apos = av[av > 0]
    bpos = bv[bv > 0]
    if apos.size == 0 or bpos.size == 0:
        return None
    am = av >= float(np.percentile(apos, percentile))
    bm = bv >= float(np.percentile(bpos, percentile))
    union = np.logical_or(am, bm).sum()
    if union == 0:
        return None
    return float(np.logical_and(am, bm).sum() / union)


def hausdorff_peak_distance(a: np.ndarray, b: np.ndarray, percentile: float = 95.0) -> float | None:
    av = clean(a)
    bv = clean(b)
    if av.max() <= EPS or bv.max() <= EPS:
        return None
    ap = np.argwhere(av >= np.percentile(av[av > 0], percentile))
    bp = np.argwhere(bv >= np.percentile(bv[bv > 0], percentile))
    if ap.size == 0 or bp.size == 0:
        return None
    return float(max(directed_hausdorff(ap, bp)[0], directed_hausdorff(bp, ap)[0]))
