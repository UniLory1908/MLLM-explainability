# V6 Horizontal Analysis

## Purpose

V6 is a horizontal repeated-measures analysis of TAM-derived attribution
behavior across prompt conditions. It is designed to make prompt-associated
changes visible at dataset level instead of only through single-case dashboard
inspection.

The analysis studies visual attribution behavior: concentration, diffusion,
fragmentation, centroid shift, scanpath instability, bbox/location-style output
signals and prompt metric fingerprints. It does not recompute raw TAM maps.

## Scope

The v6 snapshot covers:

- 100 COCO images;
- 8 prompt conditions per image;
- 800 image x prompt cases;
- the same prompt set for every image.

V6 uses existing generated TAM/dashboard/statistical artifacts. It reads the
dashboard DB/statistical exports and derived v6 CSV/Parquet files, but does not
launch Qwen/TAM inference, regenerate raw maps, run deletion/insertion tests or
perform COCO ground-truth localization evaluation.

## Main Generated Location

```text
outputs/analysis/v6_horizontal_analysis/
```

This directory is generated output and is ignored by the repository. Use the
manifest, reports and this documentation to understand its purpose; do not
commit the heavy CSV/Parquet/PNG artifacts.

## Key Generated Artifacts

Important v6 outputs include:

- `case_features_800_v2.csv`
- `tables/`
- `plots/`
- `dashboard_views/`
- `dashboard_views/questions/`
- `manifest.json`
- `horizontal_analysis_report_v1.md`
- `final_narrative_v1.md`

`case_features_800_v2` is the primary case-level table: one row per image x
prompt case. It joins identifiers, quality flags, output diagnostics,
baseline-relative visual sensitivity, diagnostic proxy scores, map metric
aggregates, region summaries, scanpath summaries, token category summaries,
bbox/location-style flags and dashboard links.

## Question-Driven Findings Layer

The question layer lives under:

```text
outputs/analysis/v6_horizontal_analysis/dashboard_views/questions/
```

It turns the v6 feature tables into claim-oriented evidence:

- `question_answer_evidence.csv` / `.md`: question, short answer, evidence
  metrics, key numbers, allowed claim, forbidden claim and next check.
- `prompt_fingerprint.csv` / `.md`: one row per prompt with metric-family
  ranks, qualitative label and interpretation sentence.
- `metric_family_clusters.csv`: grouped metrics such as concentration,
  diffusion, fragmentation, spatial shift, scanpath instability,
  bbox/location format and proxy diagnostic scores.
- `concordant_cases.csv`: cases where multiple metrics agree.
- `discordant_cases.csv`: cases where metric families or output style disagree.
- `image_structure_features.csv`: COCO annotation-derived image metadata from
  local `data/annotations/instances_val2017.json`.
- `claim_plots/*.png`: Final report plots for fingerprints, behavior
  matrix, metric correlations, concordant/discordant quadrants and image
  structure summaries.

Image-structure fields are descriptive metadata. They support exploratory
analysis of object count, category count, largest object area ratio, total
annotated area and broad category flags. They are not a ground-truth evaluation
of TAM localization.

## Dashboard Analysis Site Tree

The dashboard exposes v6 through an analysis site tree:

| Route | Role |
| --- | --- |
| `/analysis` | Analysis hub. |
| `/analysis/v6` | Compact v6 overview with KPI cards, main takeaways and links. |
| `/analysis/v6/findings` | Question-driven findings: question, answer, evidence and caveat. |
| `/analysis/v6/prompts` | Prompt fingerprints, prompt behavior matrix and prompt separability. |
| `/analysis/v6/images` | Image sensitivity, stable/sensitive images and COCO image-structure summaries. |
| `/analysis/v6/bbox` | BBox/location-style output counts, metric comparison and discordant cases. |
| `/analysis/v6/model-locations` | Parseable coordinate outputs with qualitative overlays on original images. |
| `/analysis/v6/cases` | Representative, concordant and discordant case gallery. |
| `/analysis/v6/explorer` | Filterable case-level data explorer over `case_features_800_v2.csv`. |

These pages read existing v6 CSV/PNG artifacts. They are presentation and
inspection pages, not raw-map recomputation routes.

## What Can Be Claimed

Use conservative wording:

- prompt-associated changes in TAM-derived attribution behavior;
- prompt metric fingerprints;
- metric-space separability of prompt conditions;
- nonbaseline prompt separability remains above 7-class chance under grouped
  image splits;
- concentration, diffusion, fragmentation and spatial-shift tendencies;
- bbox/location-style output as a response-format signal;
- exploratory image-structure associations.

The final v6 snapshot contains 139 strict bbox/location-style responses and
102 parseable coordinate responses. Parseable coordinates occur in
`order_disruption_stress` (72 cases), `colleague_obj_detection_hard` (28 cases)
and `misleading_wrong_subject` (2 cases). The report may shorten
`colleague_obj_detection_hard` to `object_detection_hard`, but the exported
tables keep the full data label.

V6 can support statements such as:

```text
Prompt conditions are distinguishable above chance in TAM metric space.
```

or:

```text
Some prompts show more concentrated, diffuse, fragmented or spatially shifted
TAM-derived attribution profiles than others.
```

## What Cannot Be Claimed

Do not claim:

- proof of grounding;
- proof of causal faithfulness;
- proof of correct object localization;
- human eye-tracking or gaze behavior;
- deletion/insertion faithfulness benchmark results;
- COCO ground-truth bbox/segmentation localization performance in v6;
- that bbox/location-style textual output is correct localization.

TAM scanpaths are attribution-derived trajectories, not human eye-tracking.
Diagnostic scores are heuristic proxy rankings, not causal proof. PCA,
clustering and image-structure associations are exploratory unless followed by
targeted statistical or causal validation.

The model-location overlay uses parsed response coordinates and optional COCO
annotation boxes for qualitative inspection. It does not compute IoU, mAP,
pointing-game, mass-in-mask or any other localization-accuracy benchmark.

## Reproducibility And Tracking Policy

Generated outputs under `outputs/analysis/` are ignored and should not be
committed. Repository documentation should point to manifests, reports and the
dashboard site tree rather than duplicating every CSV/PNG result.

Do not modify or delete raw maps, original metadata, dashboard DB/cache,
generated v6 artifacts or model outputs during documentation-only updates.
Raw maps remain under `outputs/prompt_sensitivity/`; dashboard indexing and
cache artifacts remain derived local state.
