from __future__ import annotations

import numpy as np

from scripts.dashboard.metrics import (
    cosine_similarity,
    extract_regions,
    jsd,
    per_map_metrics,
    top_iou,
    weighted_centroid,
)
from scripts.dashboard.precompute_normalization_variants import normalize_for_mode


def test_per_map_metrics_include_normalized_centroid_and_regions() -> None:
    arr = np.zeros((20, 30), dtype=np.float32)
    arr[4:8, 10:14] = 2.0
    arr[14:17, 20:23] = 1.0
    regions = extract_regions(arr, threshold=0.70)
    metrics = per_map_metrics(arr, regions)

    assert metrics["energy_sum"] > 0
    assert 0.0 <= metrics["global_centroid_x_norm"] <= 1.0
    assert 0.0 <= metrics["global_centroid_y_norm"] <= 1.0
    assert metrics["peak_count"] >= 1
    assert regions[0].rank == 1
    assert 0.0 <= regions[0].centroid_x_norm <= 1.0


def test_similarity_metrics_are_well_behaved() -> None:
    a = np.zeros((12, 12), dtype=np.float32)
    b = np.zeros((12, 12), dtype=np.float32)
    a[2:5, 2:5] = 1
    b[2:5, 2:5] = 1

    assert cosine_similarity(a, b) == 1.0
    assert jsd(a, b) == 0.0
    assert top_iou(a, b, 5.0) == 1.0
    assert weighted_centroid(a)[0] is not None


def test_normalization_variant_modes_are_explicit() -> None:
    arr = np.array([[0.0, 2.0], [6.0, 10.0]], dtype=np.float32)

    native = normalize_for_mode(arr, "tam_uint8_native")
    local = normalize_for_mode(arr, "local_minmax_0_1")
    mass = normalize_for_mode(arr, "probability_sum_1")

    assert native.max() == 10.0
    assert local.min() == 0.0
    assert local.max() == 1.0
    assert np.isclose(mass.sum(), 1.0)
