from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.dashboard.app import contextual_metric_value, create_app
from scripts.dashboard.config import DashboardConfig
from scripts.dashboard.location_validation import load_validation


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = Path(os.environ.get("DASHBOARD_RUNTIME_ROOT", PUBLIC_ROOT))
ARCHIVE_PATH = Path(os.environ.get("LOCATION_VALIDATION_ARCHIVE", RUNTIME_ROOT / "LORENZO_LOCATION_VALIDATION_102.zip"))
CASE_376284 = "000000376284__order_disruption_stress__stat_timebox_20260523_8prompts_c000000376284"


def runtime_config() -> DashboardConfig:
    return DashboardConfig(
        project_root=RUNTIME_ROOT,
        source_root=RUNTIME_ROOT / "outputs" / "prompt_sensitivity",
        index_dir=RUNTIME_ROOT / "outputs" / "dashboard_index",
        cache_dir=RUNTIME_ROOT / "outputs" / "dashboard_cache",
        db_path=RUNTIME_ROOT / "outputs" / "dashboard_index" / "tam_index.sqlite",
    )


def require_archive() -> None:
    if not ARCHIVE_PATH.exists():
        pytest.skip("local validation archive is not available")


def test_location_validation_loader_counts(monkeypatch) -> None:
    require_archive()
    monkeypatch.setenv("LOCATION_VALIDATION_ARCHIVE", str(ARCHIVE_PATH))
    data = load_validation(RUNTIME_ROOT)
    assert len(data.cases) == 102
    assert data.status_counts == {
        "target_strong": 50,
        "valid_other_object": 12,
        "background_or_wrong": 5,
        "ambiguous": 35,
    }
    assert data.prompt_counts == {
        "order_disruption_stress": 72,
        "colleague_obj_detection_hard": 28,
        "misleading_wrong_subject": 2,
    }


def test_location_validation_routes_nav_and_case_376284(monkeypatch) -> None:
    require_archive()
    monkeypatch.setenv("LOCATION_VALIDATION_ARCHIVE", str(ARCHIVE_PATH))
    app = create_app(runtime_config())
    client = app.test_client()

    home = client.get("/")
    assert home.status_code == 200
    assert b"COCO box validation" in home.data

    response = client.get("/analysis/location-validation")
    assert response.status_code == 200
    assert b"Supplemental COCO validation of generated coordinates" in response.data
    assert b"102" in response.data

    detail = client.get(f"/analysis/location-validation/{CASE_376284}")
    assert detail.status_code == 200
    assert b"target_strong" in detail.data
    assert b"0.776" in detail.data


def test_location_validation_image_endpoints_return_images(monkeypatch) -> None:
    require_archive()
    monkeypatch.setenv("LOCATION_VALIDATION_ARCHIVE", str(ARCHIVE_PATH))
    app = create_app(runtime_config())
    client = app.test_client()
    for panel in ("original", "coco", "model", "combined"):
        response = client.get(f"/render/location-validation/{CASE_376284}/{panel}.jpg")
        assert response.status_code == 200
        assert response.mimetype == "image/jpeg"
        assert len(response.data) > 1000


def test_location_validation_missing_archive_is_readable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCATION_VALIDATION_ARCHIVE", str(tmp_path / "missing.zip"))
    app = create_app(runtime_config())
    response = app.test_client().get("/analysis/location-validation")
    assert response.status_code == 503
    assert b"Location validation archive missing" in response.data
    assert str(tmp_path).encode() not in response.data


def test_location_validation_summary_does_not_touch_database_mtime(monkeypatch) -> None:
    require_archive()
    monkeypatch.setenv("LOCATION_VALIDATION_ARCHIVE", str(ARCHIVE_PATH))
    config = runtime_config()
    before = config.db_path.stat().st_mtime_ns
    app = create_app(config)
    response = app.test_client().get("/analysis/location-validation?status=target_strong")
    after = config.db_path.stat().st_mtime_ns
    assert response.status_code == 200
    assert before == after
    assert len(load_validation(RUNTIME_ROOT).cases) == 102


def test_metric_undefined_labels_are_contextual() -> None:
    zero_map = {"energy_sum": 0.0, "peak_count": 1}
    assert contextual_metric_value("entropy_norm", None, zero_map) == "— zero-mass map"
    assert contextual_metric_value("secondary_primary_ratio", None, zero_map) == "— requires ≥2 regions"
    assert contextual_metric_value("layer_tortuosity", None, {"layer_net_displacement": 0.0}) == "— zero net displacement"
    assert contextual_metric_value("early_late_jsd", None, {}) == "— comparison unavailable"
    assert contextual_metric_value("energy_sum", 0.0, {}) == "0.0000"


def test_metric_guide_has_explanations_not_blanket_na() -> None:
    app = create_app(runtime_config())
    response = app.test_client().get("/metrics-guide")
    assert response.status_code == 200
    assert b"Undefined when" in response.data
    assert b"No monotonic interpretation" in response.data
    assert b"Position coordinate" in response.data
    assert b">n/a<" not in response.data
