# Data And Outputs

## Directory Map

| Path | Role | Notes |
| --- | --- | --- |
| `outputs/prompt_sensitivity/` | Canonical raw experiment source of truth | Raw maps and original metadata live here. |
| `outputs/dashboard_index/tam_index.sqlite` | Dashboard SQLite index | Rebuild only when intended. |
| `outputs/dashboard_cache/` | Dashboard cache | Derived; do not clear unless requested or necessary. |
| `outputs/analysis/v6_horizontal_analysis/` | Generated v6 horizontal analysis output | Ignored/generated; contains case features, tables, plots and dashboard-ready views. |
| `outputs/v3_visualization/` | Legacy derived visualization | Not canonical; can be regenerated. |
| `data/` | Local COCO data | Heavy local data. |
| `models/` | Local model files/cache | Heavy local data. |
| `prompt_sets/` | Prompt definitions | Tracked project inputs. |
| `configs/` | Image registry and config | Tracked project inputs. |
| `scripts/dashboard/` | Dashboard code | Runtime code; not touched in docs-only tasks. |

## Source Of Truth

CURRENT canonical source for raw experiment data:

```text
outputs/prompt_sensitivity/
```

Use this path for raw maps, original metadata and case discovery when exact
experiment data matters.

Do not use `outputs/v3_visualization/` as canonical. It contains derived/static
visualization artifacts and may be stale relative to the raw experiment data.

## Raw Vs Derived Artifacts

Raw / canonical:

- raw TAM `.npy` maps;
- original `metadata.json`;
- original generated response and token/word metadata;
- run configuration saved with the case.

Derived / regenerable:

- dashboard DB index;
- dashboard precomputed metric rows;
- dashboard cache files;
- rendered JPG/PNG/GIF/contact sheets;
- `outputs/v3_visualization/`;
- CSV/Parquet exports produced from DB and metadata.
- `outputs/analysis/` horizontal-analysis tables, plots, reports and dashboard
  views.

Derived does not mean disposable without permission. Confirm before deleting
large derived outputs if they are being used by notebooks, reports or the
dashboard.

## DB / Cache Lifecycle

Dashboard DB:

```text
outputs/dashboard_index/tam_index.sqlite
```

Dashboard cache:

```text
outputs/dashboard_cache/
```

Treat both as derived from raw experiment data, but do not remove or rebuild
them casually. Rebuilding can be expensive and may change the visible dashboard
state.

Safe to regenerate when explicitly intended:

- dashboard index from canonical cases;
- metric precompute tables from indexed raw maps;
- cache thumbnails/renders;
- static visualization exports;
- statistical archive tables.

Must not be touched without explicit instruction:

- `outputs/prompt_sensitivity/`;
- raw maps;
- original metadata;
- datasets and model caches;
- output junctions.

Dashboard/statistical archive dependencies belong in the main project
dependency file, currently repository-root `requirements.txt`.

## Statistical Archive Location

CURRENT archive root:

```text
outputs/statistical_archive/<run_name>/
```

Implemented subfolders:

```text
csv/
parquet/
preview/
manifest/
logs/
```

Implemented readme:

```text
DATASET_README.md
```

Export command:

The command below is for the historical 40-case official subset. For the
100-image / 800-case v6 analysis, do not use `--official-only` unless
`is_official` has been intentionally updated.

```powershell
.\.venv\Scripts\python.exe -m scripts.dashboard.export_statistical_archive --official-only --archive-name current_official_40
```

Safe defaults:

- an existing archive is not overwritten unless `--overwrite` is passed;
- unofficial/probe cases are excluded when `--official-only` is used;
- CSV export is required;
- Parquet is attempted unless `--skip-parquet` is passed, but missing
  `pyarrow`/`fastparquet` does not fail the export;
- XLSX preview is attempted unless `--skip-xlsx` is passed, but missing Excel
  writer support does not fail the export.

Current main project dependencies include `pyarrow`, `openpyxl`, `scikit-image`
and `POT` in `requirements.txt`.

## Expected Tables

CURRENT exported table/file names:

- `cases`
- `map_metrics_core`
- `map_metrics_normalization_variants`
- `region_summary`
- `layer_scanpaths`
- `word_scanpaths`
- `output_diagnostics`
- `visual_sensitivity_vs_baseline`
- `diagnostic_scores`
- `token_category_summary`
- `dashboard_links`

Currently not exported as full tables:

- `regions_full`;
- `case_summary` separate from `output_diagnostics`;
- pairwise baseline summary tables.

Add those only when the analysis requires them and the size/cost is acceptable.

Format policy:

- CSV is required for spreadsheet compatibility.
- Parquet is preferred for Python and large data analysis.
- XLSX should be preview/index only, not the full large dataset.

Stable keys for dashboard compatibility:

- `case_id`
- `image_id`
- `condition_id`
- `prompt_label`
- `word_index`
- `layer_index`

`map_metrics_normalization_variants` is additive. It must not replace or
reinterpret `map_metrics_core`. The current full export includes
`normalization_mode=tam_uint8_native`, which makes the saved TAM postprocessed
0-255-like map scale explicit while preserving the original dashboard metric
definitions in `map_metrics_core`.

Useful dashboard URL fields:

- `dashboard_case_url`
- `dashboard_matrix_url`
- `dashboard_compare_url`
- `dashboard_word_url`

## V6 Horizontal Analysis Output

Current generated v6 root:

```text
outputs/analysis/v6_horizontal_analysis/
```

Important files and folders:

- `case_features_800_v2.csv`
- `tables/`
- `plots/`
- `dashboard_views/`
- `dashboard_views/questions/`
- `manifest.json`
- `horizontal_analysis_report_v1.md`
- `final_narrative_v1.md`

`case_features_800_v2` is the main 800-row case-level table. `tables/` and
`plots/` contain prompt summaries, image sensitivity rankings, bbox summaries,
statistical-test outputs, prompt separability outputs, representative cases and
dataset-level visualizations.

`dashboard_views/` contains compact dashboard-ready summaries and absolute TAM
profile views. `dashboard_views/questions/` contains the question-driven
findings layer: question-answer evidence, prompt fingerprints, metric family
clusters, concordant/discordant cases, COCO image-structure joins and claim
plots.

Generated output manifests and reports contain exact inventories. Heavy
generated CSV/Parquet/PNG artifacts are local analysis products, not primary
repository documentation.
