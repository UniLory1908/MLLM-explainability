# Dashboard

## Architecture

CURRENT: the dashboard is a custom Flask/Jinja/SQLite/JavaScript application.

Root:

```text
scripts/dashboard/
```

Important files:

```text
scripts/dashboard/app.py
scripts/dashboard/config.py
scripts/dashboard/db.py
scripts/dashboard/data_access.py
scripts/dashboard/index_cases.py
scripts/dashboard/precompute_metrics.py
scripts/dashboard/precompute_derived_metrics.py
scripts/dashboard/export_statistical_archive.py
scripts/dashboard/metrics.py
scripts/dashboard/pairwise.py
scripts/dashboard/rendering.py
scripts/dashboard/cache.py
scripts/dashboard/clear_cache.py
scripts/dashboard/metric_registry.py
scripts/dashboard/templates/
scripts/dashboard/static/dashboard.css
scripts/dashboard/static/dashboard.js
```

DB path:

```text
outputs/dashboard_index/tam_index.sqlite
```

Cache path:

```text
outputs/dashboard_cache/
```

## Start And Check

Start locally:

```powershell
.\.venv\Scripts\python.exe -m scripts.dashboard.app --host 127.0.0.1 --port 5050
```

Local URL:

```text
http://127.0.0.1:5050/
```

Tailscale URL may be available, but should not be documented as stable unless
the current address has been explicitly verified.

## Routes

CURRENT routes to know:

| Route | Purpose |
| --- | --- |
| `/` | Case index / dashboard home. |
| `/prompts` | Prompt overview. |
| `/metrics-guide` | Metric explanation page. |
| `/metrics` | Metric rankings / browsing. |
| `/analysis` | Analysis hub. |
| `/analysis/v6` | V6 horizontal analysis overview. |
| `/analysis/v6/findings` | Question-driven v6 findings and claim-level evidence. |
| `/analysis/v6/prompts` | Prompt fingerprints, behavior matrix and separability. |
| `/analysis/v6/images` | Image sensitivity and COCO image-structure summaries. |
| `/analysis/v6/bbox` | BBox/location-style output summaries and discordant cases. |
| `/analysis/v6/cases` | Representative, concordant and discordant case gallery. |
| `/analysis/v6/explorer` | Filterable v6 case-level data explorer. |
| `/case/<case_id>` | Single case page. |
| `/case/<case_id>/word/<word_index>` | Word detail page. |
| `/case/<case_id>/matrix` | Word/layer matrix page. |
| `/compare` | Case comparison page. |

Endpoint smoke checks should include the pages above plus one real case, one
real word page, one matrix page and one compare page.

Internal/render/API routes also exist for generated dashboard assets and JSON
data:

```text
/render/map/<case_id>/<word_index>/<layer_index>.png
/render/cell/<case_id>/<word_index>/<layer_index>.jpg
/render/final-preview/<case_id>.jpg
/render/final-animation/<case_id>.gif
/render/original/<case_id>.jpg
/render/scanpath/<mode>/<case_id>.png
/render/diff/<case_id_a>/<case_id_b>/<word_index>/<layer_index>.png
/api/case/<case_id>/words
/api/case/<case_id>/map_metrics
/api/case/<case_id>/word_metrics/<word_index>
/api/case/<case_id>/summary_metrics
/api/case/<case_id>/metric_highlights
/api/case/<case_id>/regions
```

These are support endpoints for the UI/cache; the main human navigation routes
remain the routes in the table above.

## V6 Analysis Pages

The v6 analysis site tree is presentation-only. It reads generated v6 CSV/PNG
artifacts from:

```text
outputs/analysis/v6_horizontal_analysis/
```

High-level inputs by page:

| Page | Main generated inputs |
| --- | --- |
| `/analysis/v6` | `manifest.json`, v6 KPI tables, selected overview plots. |
| `/analysis/v6/findings` | `dashboard_views/questions/question_answer_evidence.csv`, `prompt_fingerprint.csv`, `metric_family_clusters.csv`, `claim_plots/*.png`. |
| `/analysis/v6/prompts` | `dashboard_views/questions/prompt_fingerprint.csv`, `dashboard_views/prompt_absolute_profile*.csv`, prompt summary and separability tables. |
| `/analysis/v6/images` | `tables/image_sensitivity_ranking.csv`, `dashboard_views/questions/image_structure_*.csv`, image-structure claim plot. |
| `/analysis/v6/bbox` | `tables/bbox_by_prompt.csv`, `tables/bbox_metric_comparison.csv`, concordant/discordant case CSVs and bbox plots. |
| `/analysis/v6/cases` | `tables/representative_cases.csv`, `concordant_cases.csv`, `discordant_cases.csv`. |
| `/analysis/v6/explorer` | `case_features_800_v2.csv`. |

The analysis pages do not recompute raw TAM maps, do not update the DB/cache and
do not run model inference. They are intended for Final report inspection of
prompt-associated changes in TAM-derived attribution behavior.

## Index / Precompute Flow

Expected flow:

1. Discover canonical cases from `outputs/prompt_sensitivity/`.
2. Build/update `outputs/dashboard_index/tam_index.sqlite`.
3. Precompute core map metrics, regions, layer scanpaths and word scanpaths.
4. Precompute derived diagnostic/baseline tables when statistical export or
   ranking analysis needs them.
5. Export the statistical archive when requested.
6. Use `outputs/dashboard_cache/` for derived dashboard render/cache files.
7. Open the dashboard for visual inspection and route checks.

The DB/cache are derived, but should not be deleted or rebuilt unless that is
the intended task.

Common commands:

The command below is for the historical 40-case official subset. For the
100-image / 800-case v6 analysis, do not use `--official-only` unless
`is_official` has been intentionally updated.

```powershell
.\.venv\Scripts\python.exe -m scripts.dashboard.index_cases --official-only
.\.venv\Scripts\python.exe -m scripts.dashboard.precompute_metrics --official-only
.\.venv\Scripts\python.exe -m scripts.dashboard.precompute_derived_metrics --official-only
.\.venv\Scripts\python.exe -m scripts.dashboard.export_statistical_archive --official-only --archive-name current_official_40
```

Useful bounded options:

- `index_cases.py`: `--metadata-root`, `--case-filter`, `--limit-cases`,
  `--dry-run`, `--rebuild`.
- `precompute_metrics.py`: `--case-filter`, `--case-limit`, `--init-db`.
- `precompute_derived_metrics.py`: `--case-filter`, `--case-limit`,
  `--compute-map-similarities`, `--no-ssim`.
- `export_statistical_archive.py`: `--dashboard-base-url`, `--overwrite`,
  `--skip-parquet`, `--skip-xlsx`.

## Compatibility With Statistical Exports

The dashboard must remain compatible with tabular exports. Export files should
carry stable identifiers that allow rows to link back to dashboard views.

Required stable identifiers:

- `case_id`
- `image_id`
- `condition_id`
- `prompt_label`
- `word_index`
- `layer_index`

Recommended dashboard link fields:

- `dashboard_case_url`
- `dashboard_matrix_url`
- `dashboard_compare_url`
- `dashboard_word_url`

The statistical archive should not replace the dashboard. It should make large
numeric analysis easier while the dashboard remains the visual inspection tool.

## Baseline Convention

Official baseline prompt label:

```text
baseline_neutral
```

CURRENT DB has one baseline per image in the official set. Future scripts should
fail or report clearly if a baseline is missing or duplicated. Do not invent a
fuzzy fallback baseline silently.
