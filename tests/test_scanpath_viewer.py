from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.analysis.scanpath_viewer import (
    MetadataContractError,
    ViewerInputError,
    build_scanpath_frames_with_stats,
    save_contact_sheet,
    validate_metadata_contract,
)


def _write_dummy_image(path: Path, w: int = 32, h: int = 24) -> None:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = 120
    Image.fromarray(arr, mode="RGB").save(path)


def _scanpath_ready_metadata(img_path: Path) -> dict:
    return {
        "step_records": [
            {
                "step_idx": 0,
                "token_label": "hello",
                "heatmap_path": str(img_path),
                "hotspots": [{"centroid_x": 10, "centroid_y": 8, "area": 20, "strength": 1.0}],
                "dominant_hotspot": {"centroid_x": 10, "centroid_y": 8, "area": 20, "strength": 1.0},
            }
        ],
        "scanpath": {"tracks": [{"track_id": 0, "points": [{"step_idx": 0, "centroid_x": 10, "centroid_y": 8}]}]},
    }


def test_sheet_cols_zero_fails(tmp_path: Path) -> None:
    img = tmp_path / "f0.png"
    _write_dummy_image(img)
    with pytest.raises(ViewerInputError):
        save_contact_sheet([img], tmp_path / "sheet.jpg", cols=0)


def test_metadata_without_scanpath_fails_by_default(tmp_path: Path) -> None:
    img = tmp_path / "h0.png"
    _write_dummy_image(img)
    legacy = {"step_records": [{"step_idx": 0, "token_label": "a", "heatmap_path": str(img)}]}
    with pytest.raises(MetadataContractError):
        validate_metadata_contract(legacy, allow_heatmap_only=False)


def test_no_valid_frames_returns_zero_and_skips(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    metadata = _scanpath_ready_metadata(missing)
    out = tmp_path / "out"
    frames, stats = build_scanpath_frames_with_stats(metadata, out)
    assert frames == []
    assert stats["frames_generated"] == 0
    assert stats["skipped_missing_heatmap_file"] >= 1


def test_scanpath_smoke_generates_frame_and_artifacts(tmp_path: Path) -> None:
    img = tmp_path / "h0.png"
    _write_dummy_image(img)
    metadata = _scanpath_ready_metadata(img)
    out = tmp_path / "out"
    frames, stats = build_scanpath_frames_with_stats(metadata, out)
    assert len(frames) >= 1
    assert stats["frames_generated"] >= 1

    from scripts.analysis.scanpath_viewer import save_gif

    gif = out / "scanpath.gif"
    sheet = out / "scanpath_contact_sheet.jpg"
    save_gif(frames, gif, duration_ms=50)
    save_contact_sheet(frames, sheet, cols=4)
    assert gif.exists()
    assert sheet.exists()
