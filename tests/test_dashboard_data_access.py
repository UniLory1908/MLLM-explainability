from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.dashboard.config import DashboardConfig
from scripts.dashboard.data_access import build_case_record, load_word_layer_map, map_paths_for_word_layer


def test_map_paths_use_step_record_layer_raw_maps(tmp_path: Path) -> None:
    root = tmp_path
    prompt_dir = root / "outputs" / "prompt_sensitivity" / "000000000001_demo" / "run" / "00_baseline"
    raw_dir = prompt_dir / "raw_maps" / "000000000001" / "layer_000"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "step_0000_demo.npy"
    np.save(raw_path, np.ones((4, 5), dtype=np.float32))
    metadata = {
        "img_id": 1,
        "image_stem": "000000000001",
        "prompt_label": "baseline",
        "all_layers": True,
        "layers": [0],
        "step_records": [
            {
                "step_idx": 0,
                "token_label": "demo",
                "token_piece": "demo",
                "layer_raw_maps": {"0": str(raw_path)},
                "resized_raw_map_shape": [4, 5],
            }
        ],
        "word_records": [
            {
                "word_index": 0,
                "word_label": "demo",
                "canonical_word_label": "demo",
                "source_step_indices": [0],
                "source_token_pieces": ["demo"],
            }
        ],
    }
    meta_path = prompt_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    config = DashboardConfig(
        project_root=root,
        source_root=root / "outputs" / "prompt_sensitivity",
        index_dir=root / "outputs" / "dashboard_index",
        cache_dir=root / "outputs" / "dashboard_cache",
        db_path=root / "outputs" / "dashboard_index" / "tam_index.sqlite",
    )

    case = build_case_record(meta_path, config)
    assert case is not None
    paths = map_paths_for_word_layer(metadata, case.words[0], 0, prompt_dir)
    assert paths == [raw_path]
    arr = load_word_layer_map(paths)
    assert arr is not None
    assert arr.shape == (4, 5)
