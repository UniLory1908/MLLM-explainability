from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.dashboard.app import create_app
from scripts.dashboard.metrics import (
    cosine_similarity,
    extract_regions,
    jsd,
    per_map_metrics,
    top_iou,
    weighted_centroid,
)
from scripts.dashboard.precompute_normalization_variants import normalize_for_mode
from scripts.dashboard.rendering import matching_coco_annotations, parse_model_locations


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


def test_model_location_parser_extracts_bboxes_and_points_only() -> None:
    bbox = parse_model_locations("a person(419,210),(859,991)")
    assert bbox[0]["kind"] == "bbox"
    assert bbox[0]["coords"] == (419.0, 210.0, 859.0, 991.0)

    listed = parse_model_locations("bbox [10, 20, 300, 400]")
    assert listed[0]["kind"] == "bbox"
    assert listed[0]["coords"] == (10.0, 20.0, 300.0, 400.0)

    point = parse_model_locations("object center (0.25, 0.75)")
    assert point[0]["kind"] == "point"
    assert point[0]["coords"] == (0.25, 0.75)

    assert parse_model_locations("The man is holding a pizza box.") == []


def test_model_location_coco_matching_uses_simple_category_aliases(tmp_path: Path) -> None:
    annotation_path = tmp_path / "instances_val2017.json"
    annotation_path.write_text(
        json.dumps(
            {
                "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "fire hydrant"}],
                "annotations": [
                    {"image_id": 10, "category_id": 1, "bbox": [10, 20, 30, 40], "area": 1200},
                    {"image_id": 10, "category_id": 2, "bbox": [100, 200, 50, 70], "area": 3500},
                ],
            }
        ),
        encoding="utf-8",
    )

    locations = parse_model_locations("a man(419,210),(859,991)")
    matches, matched = matching_coco_annotations(annotation_path, 10, locations)

    assert matched is True
    assert len(matches) == 1
    assert matches[0]["category"] == "person"


def test_model_locations_route_is_registered() -> None:
    app = create_app()
    rules = {rule.rule for rule in app.url_map.iter_rules()}

    assert "/analysis/v6/model-locations" in rules
    assert "/render/model-location/<case_id>.jpg" in rules
