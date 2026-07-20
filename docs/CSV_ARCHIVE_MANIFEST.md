# CSV Archive Manifest

Complete CSV files are not committed to GitHub because the metric tables are
large. The archive prepared for shared Drive is:

Drive course folder: `FVAB 2025-2026 / Gruppo 17 / CSV completi/`

Local export folder layout:

```text
outputs/statistical_archive/final_csv_archive_20260629/
```

The archive was exported from the dashboard SQLite database and derived metric
tables. Exporting it does not rerun Qwen/TAM inference and does not recompute
raw TAM maps.

## Dataset Scope

| Item | Count |
| --- | ---: |
| Cases | 800 |
| Images | 100 |
| Prompt conditions per image | 8 |
| Indexed maps | 995686 |
| Map metric rows | 995683 |
| Missing map metric rows | 3 |
| Layer scanpath rows | 34334 |
| Word scanpath rows | 23200 |

## CSV Files

| CSV | Description | Rows | Columns | Size |
| --- | --- | ---: | ---: | ---: |
| `cases.csv` | One row per image x prompt case, with metadata, run paths and dashboard links. | 800 | 26 | 904206 bytes |
| `dashboard_links.csv` | Direct dashboard URLs for case, matrix, compare and word views. | 800 | 11 | 387834 bytes |
| `diagnostic_scores.csv` | Case-level heuristic proxy scores for ranking and inspection. | 800 | 16 | 466088 bytes |
| `layer_scanpaths.csv` | TAM-derived layer-wise trajectory descriptors by case and word. | 34334 | 25 | 13896394 bytes |
| `map_metrics_core.csv` | Core per-map TAM descriptors: intensity, concentration, centroid, spread, peaks and regions. | 995683 | 43 | 750885826 bytes |
| `map_metrics_normalization_variants.csv` | Additive TAM-native normalization metric table, currently including `tam_uint8_native`. | 995683 | 45 | 808635476 bytes |
| `output_diagnostics.csv` | Prompt/response text diagnostics, length/overlap fields and bbox/location-style flags. | 800 | 35 | 882867 bytes |
| `region_summary.csv` | Compact per-case region summaries aggregated from connected activation regions. | 4000 | 12 | 1067480 bytes |
| `token_category_summary.csv` | Case summaries grouped by generated token category. | 2035 | 13 | 977992 bytes |
| `visual_sensitivity_vs_baseline.csv` | Baseline-relative visual sensitivity and metric deltas using `baseline_neutral`. | 800 | 38 | 717415 bytes |
| `word_scanpaths.csv` | TAM-derived word-level scanpath descriptors. | 23200 | 10 | 4808137 bytes |

Parquet equivalents and a small XLSX preview are also present in the Drive
folder when generated locally. The CSV files are the requested complete metric
delivery.

## Regeneration Command

From the local project root containing the dashboard DB:

```bash
python -m scripts.dashboard.export_statistical_archive --archive-name final_csv_archive_20260629 --dashboard-base-url http://127.0.0.1:5050 --overwrite
```

Inputs:

- `outputs/dashboard_index/tam_index.sqlite`
- `outputs/prompt_sensitivity/`

Outputs:

- `outputs/statistical_archive/final_csv_archive_20260629/csv/`
- `outputs/statistical_archive/final_csv_archive_20260629/parquet/`
- `outputs/statistical_archive/final_csv_archive_20260629/manifest/`

## Caveats

- Three indexed raw maps are not covered by `map_metrics_core` because their
  `.npy` files are unreadable or filesystem-stalling.
- Candidate diagnostic scores are exploratory proxies for ranking and
  inspection, not proof of hallucination, grounding or causal faithfulness.
- TAM-derived scanpaths are attribution-derived trajectories, not human
  eye-tracking.
- BBox/location-style textual output is a response-format signal and does not
  establish correct localization.
