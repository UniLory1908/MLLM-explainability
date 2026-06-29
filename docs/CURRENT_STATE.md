# Current State

## Project Goal

CURRENT: this workspace studies explainability for Multimodal Large Language
Models using Token Attribution Maps (TAM) / saliency heatmaps from
`Qwen/Qwen2-VL-2B-Instruct`.

The main analysis target is visual attribution behavior:

- raw TAM heatmaps;
- map intensity and concentration;
- centroids and spatial geometry;
- hotspots, regions and multipeak structure;
- word-wise scanpaths;
- layer-wise scanpaths;
- numerical metrics and rankings.

Prompt wording and generated text are still important context, but this is not
primarily an NLP response-evaluation project.

## Dashboard Status

CURRENT: the dashboard is a custom Flask/Jinja/SQLite/JavaScript application in:

```text
scripts/dashboard/
```

Local start command:

```powershell
.\.venv\Scripts\python.exe -m scripts.dashboard.app --host 127.0.0.1 --port 5050
```

Local URL:

```text
http://127.0.0.1:5050/
```

Tailscale access may be available, but should not be assumed stable.

## Current Statistical Dataset Snapshot

CURRENT dashboard/statistical archive snapshot:

```text
100 images x 8 prompts = 800 cases
```

Verified DB counts for this snapshot:

| Table / item | Count |
| --- | ---: |
| cases | 800 |
| distinct images | 100 |
| maps | 995686 |
| map_metrics | 995683 |
| missing map_metrics | 3 |
| regions | 7668169 |
| layer_scanpaths | 34334 |
| word_scanpaths | 23200 |
| output_diagnostics | 800 |
| visual_sensitivity_vs_baseline | 800 |
| diagnostic_scores | 800 |
| token_category_summary | 2035 |
| map_metrics_normalization_variants | 995683 |

The three missing `map_metrics` rows are due to unreadable/filesystem-stalling
raw `.npy` maps, not missing indexed cases. Re-check the SQLite DB before
citing exact counts in a report.

## Current Priority

CURRENT: the dashboard now has implemented scripts for derived metric
precompute, additive TAM-native normalization metrics, and statistical archive
export:

```text
scripts/dashboard/precompute_derived_metrics.py
scripts/dashboard/precompute_normalization_variants.py
scripts/dashboard/export_statistical_archive.py
```

CURRENT: the larger statistical archive run has been exported at:

```text
outputs/statistical_archive/stat_timebox_20260523_progress/
```

The excluded failed/partial image `224724` is not indexed in the 100-image
dataset. Compact image `34071` is included.

## Current V6 Horizontal Analysis

CURRENT: v6 horizontal analysis exists locally under:

```text
outputs/analysis/v6_horizontal_analysis/
```

This is a generated/ignored analysis artifact for:

```text
100 COCO images x 8 prompt conditions = 800 cases
```

Primary generated case-level table:

```text
case_features_800_v2.csv
```

The v6 analysis uses existing dashboard/statistical artifacts. It does not
recompute raw TAM maps and does not run new model inference.

Important v6 subfolders:

- `tables/`: prompt summaries, image sensitivity rankings, bbox summaries,
  statistical tests, prompt separability and representative cases.
- `plots/`: dataset-level visualizations and prompt-effect plots.
- `dashboard_views/`: dashboard-ready summaries and absolute/general TAM
  prompt profiles.
- `dashboard_views/questions/`: question-driven findings, prompt fingerprints,
  metric family clusters, concordant/discordant cases, image-structure joins
  and claim plots.

Final report docs generated in v6 include:

- `horizontal_analysis_report_v1.md`
- `final_narrative_v1.md`
- `manifest.json`

Dashboard analysis routes:

- `/analysis`
- `/analysis/v6`
- `/analysis/v6/findings`
- `/analysis/v6/prompts`
- `/analysis/v6/images`
- `/analysis/v6/bbox`
- `/analysis/v6/cases`
- `/analysis/v6/explorer`

Interpret v6 as repeated-measures analysis of TAM-derived attribution behavior.
It can support conservative claims about prompt-associated metric changes,
prompt fingerprints, metric-space separability, concentration/diffusion,
fragmentation and spatial-shift tendencies. It does not prove grounding,
correct localization or causal faithfulness.

Current next priority:

- use the existing 100-image / 800-case archive and v6 horizontal analysis for
  Final report interpretation, report figures and dashboard inspection;
- keep generated v6 artifacts local/ignored unless a lightweight tracking
  policy is explicitly selected for manifests or summaries;
- treat future large runs as a separate historical-run-policy / future-run
  protocol task, with preflight, pilot timing, disk checks, logging and
  skip/resume behavior.

## Deprecated / Legacy

`outputs/v3_visualization/` is legacy/derived/static visualization output. It is
not canonical raw experiment data and can be regenerated if needed.

Historical run inventory is archive-only unless referenced by a current doc:

- `outputs/RUN_INVENTORY.md`

## Current Constraints

- Do not delete canonical outputs, raw maps, original metadata, notebooks or
  dashboard DB/cache unless explicitly instructed.
- Do not silently invent a baseline when `baseline_neutral` is missing or
  duplicated.
- Do not hide coordinate/bounding-box style output from
  `order_disruption_stress`; it is real model output.
- Core dashboard metrics are offline descriptive/proxy metrics, not causal
  faithfulness measurements.
- Derived metric/export scripts operate from existing metadata and DB metrics;
  they do not launch new model inference.

## Commenti Finali

La pipeline TAM fornisce una pipeline completa per ispezionare e misurare
prompt-associated changes in TAM-derived attribution behavior. La snapshot
corrente supporta analisi di concentrazione, diffusione, centroid shift,
hotspot/regioni, multipeak structure, scanpath descriptors e prompt metric
fingerprints su 100 immagini x 8 prompt.

I risultati restano diagnostici ed esplorativi: TAM non e' ground truth,
scanpath non e' eye-tracking, bbox/location-style output non implica
localizzazione corretta e i proxy non dimostrano causal faithfulness. I prossimi
sviluppi naturali sono validazione su annotazioni spaziali, test perturbativi,
negative queries piu' bilanciate e revisione qualitativa dei casi selezionati.
