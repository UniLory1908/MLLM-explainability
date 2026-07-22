# Metrics

## Core Metric Categories

CURRENT dashboard/offline metric categories:

| Category | Examples / intent |
| --- | --- |
| Map intensity / raw stats | Sum, max, mean, variance and raw activation summaries. |
| Concentration / diffusion | Entropy-like concentration, spread and mass distribution. |
| Spatial / centroid geometry | Centroid position, movement and distance summaries. |
| Hotspot / region / multipeak | Connected regions, dominant hotspot, region count and multipeak structure. |
| Pairwise map similarity | Similarity/distance between maps under controlled comparisons. |
| Hotspot overlap / spatial shift | Overlap and displacement between salient regions. |
| Layer-wise scanpath metrics | TAM-derived movement across layers. |
| Word-wise scanpath metrics | TAM-derived movement across generated words. |

## Normalization Policy

Keep normalization explicit. Do not mix these silently:

| Normalization | Meaning |
| --- | --- |
| `raw` | Original numeric TAM values. Good for raw intensity summaries when scale is meaningful. |
| `minmax` | Per-map contrast normalization. Good for visualization and shape comparison, not absolute intensity. |
| `probability` / `mass` | Map normalized to sum to 1. Good for entropy, mass and distribution-style metrics. |
| `zscore` | Standardized values. Useful for relative deviations within a map/distribution. |

Rendered images are not the computational source of truth. Metrics should use
raw numerical maps or explicitly normalized numerical maps.

## TAM-Native Normalization Variant

The original TAM code normalizes image and text scores together, applies
`rank_guassian_filter` to the image-token grid, and scales the resulting image
map to uint8-like 0-255 values. The prompt sweep saves those postprocessed TAM
maps as resized `float32` `.npy` arrays so dashboard coordinates match the
rendered image size.

Current dashboard metrics remain unchanged:

- raw/statistical fields are computed from cleaned saved TAM arrays;
- distribution fields use sum-to-1 probability normalization;
- region masks use local minmax thresholds;
- word-level maps combine subtoken maps with pixel-wise max pooling.

Additive normalization-variant output lives in:

```text
map_metrics_normalization_variants
```

The full 100-image archive currently exports `normalization_mode`:

- `tam_uint8_native`: explicit metrics on the saved TAM postprocessed
  0-255-like map scale, copied from verified core `map_metrics` rows.

The precompute command can also compute bounded/resumable experimental modes
for selected cases:

- `local_minmax_0_1`;
- `probability_sum_1`.

Those optional modes read raw maps and are intentionally not required for the
full archive by default.

## Scanpath Metrics

CURRENT scanpaths are TAM-derived, not eye-tracking.

Word-wise scanpath:

- aggregates token maps into word-level maps;
- tracks hotspot/centroid movement through generated words;
- helps inspect how visual support changes during generation.

Layer-wise scanpath:

- tracks hotspot/centroid behavior across model layers;
- helps inspect whether spatial focus changes through the network stack.

Do not write that scanpaths replicate human gaze. Use cautious wording:

```text
TAM-derived trajectory
```

## Pairwise Metrics

Pairwise comparisons should be bounded and purposeful:

- prompt vs `baseline_neutral`;
- aligned word/layer comparisons where valid;
- selected case-to-case comparisons for inspection.

Do not bulk-precompute full all-vs-all pairwise metrics unless explicitly
requested.

EMD should not be bulk-precomputed unless already cached/materialized or
explicitly requested.

## Methodological Distinction

Stability metrics:

- measure variation across prompts, words, layers or repeated conditions;
- can show sensitivity or instability;
- do not prove whether the model is causally relying on a region.

Plausibility / alignment proxy metrics:

- compare attribution shape with expected visual behavior or heuristic signals;
- useful for ranking suspicious or interesting cases;
- not ground truth and not proof of hallucination.

Causal faithfulness metrics:

- require causal perturbations, new inference, log-prob access, ablations or
  retraining-style evidence;
- are not part of the current offline dashboard core.

CURRENT core dashboard does NOT measure causal faithfulness.

## Derived Metrics

CURRENT implemented derived metrics from existing metadata plus DB metrics:

- output diagnostics;
- response divergence vs baseline;
- visual prompt sensitivity vs baseline;
- candidate diagnostic scores;
- token category summaries.

These require no new model inference, no ground truth and no full pairwise
all-vs-all precompute.

Implemented DB tables:

- `output_diagnostics`
- `visual_sensitivity_vs_baseline`
- `diagnostic_scores`
- `token_category_summary`
- `map_metrics_normalization_variants`

Precompute command:

```powershell
.\.venv\Scripts\python.exe -m scripts.dashboard.precompute_derived_metrics --official-only
```

`visual_sensitivity_vs_baseline` uses `baseline_neutral` only and records
`alignment_method=positional_common_index`. Aggregate map deltas, centroid
shifts and scanpath deltas are computed from existing DB metrics. Raw-map
cosine/top5/JSD similarities are opt-in with `--compute-map-similarities`
because the full positional pass can be expensive; SSIM is optional and skipped
with `--no-ssim`. EMD is not bulk-computed.

Candidate diagnostic score names:

- `unstable_explanation_candidate_score`
- `prompt_dominated_candidate_score`
- `weak_grounding_candidate_score`
- `multipeak_ambiguity_score`
- `bbox_or_grounding_format_score`

Mark these as candidate/proxy/diagnostic/ranking heuristics. They are not causal
proof and not hallucination proof.

## Future Optional Metrics With Ground Truth

FUTURE/OPTIONAL if ground-truth masks/boxes are intentionally brought into the
analysis:

- `Obj-IoU`
- `Func-IoU`
- `F1-IoU`
- `Pointing Game`
- `Energy-based Pointing`
- `mass_in_mask`
- `gt_centroid_distance`
- `gt_region_iou_top5`

These should be documented separately from the current offline dashboard core.

## Not-Now Causal Metrics

Do not implement now:

- Deletion AUC
- Insertion AUC
- ROAR
- Infidelity
- Sensitivity perturbative metrics
- PDM exact
- true causal faithfulness metrics

Reason: they require ground truth, perturbations, new model inference,
log-probs, or retraining.

## `order_disruption_stress`

`order_disruption_stress` may trigger Qwen2-VL object-reference or
bounding-box-style output. Coordinate tokens are real model output, not
dashboard formatting bugs.

Do not hide or sanitize them. Treat this as a grounding-format /
bbox-triggered stress condition with a caveat in analysis text.

## Undefined And Not-Applicable Values

Dashboard metric cells distinguish numeric zero from values that are undefined
or not applicable. Examples:

- zero-mass maps can make entropy, HHI, effective area, centroid, spread and
  anisotropy undefined;
- secondary/primary ratio requires at least two valid salient regions;
- tortuosity requires a valid path and nonzero net displacement;
- baseline-relative or adjacent-layer metrics require an aligned comparison
  map;
- missing metric records are displayed as unavailable, not as numeric zero.

The Metric Guide documents `undefined_when` for the exposed metrics. These
labels are display semantics only; they do not change stored metric values.

## V6 Interpretation Concepts

V6 horizontal analysis groups case-level metrics into prompt and case
signatures for Final report interpretation. These concepts are descriptive
metric summaries, not causal proof.

Prompt fingerprint:

- a compact metric signature for one prompt condition across the 100-image
  repeated-measures dataset;
- combines visual shift, concentration, diffusion, spatial shift,
  fragmentation, scanpath instability, bbox/location tendency and proxy
  diagnostic scores;
- useful for comparing prompt conditions in metric space.

Metric families used in v6:

| Family | Typical evidence |
| --- | --- |
| Concentration | Higher top-5 mass, higher HHI/sparsity, lower entropy/effective area. |
| Diffusion | Higher entropy/effective area/spread and lower top-5 mass. |
| Fragmentation | Higher region count, peak count, multipeak indicators and secondary/primary mass ratio. |
| Spatial shift | Baseline-relative centroid shift and visual-change summaries. |
| Scanpath instability | TAM-derived layer/word path length, jumps, displacement and tortuosity. |
| BBox/location format | Stored bbox flag, strict/broad bbox-style output flags and coordinate-token indicators. |
| Proxy diagnostic scores | Weak grounding, prompt-dominated, unstable explanation and multipeak ambiguity scores. |

Strict bbox/location-style counts and parseable-coordinate counts are related
but not identical. Strict/broad flags come from output diagnostics and regex /
token heuristics over generated responses; the current strict count is 139 out
of 800. Parseable coordinates are counted only when the dashboard parser can
render a coordinate object: `[x0,y0,x1,y1]`, two nearby pairs
`(x0,y0),(x1,y1)` as a bbox, or one pair `(x,y)` as a point. The current
parseable total is 102 out of 800: `order_disruption_stress` 72,
`colleague_obj_detection_hard` 28 and `misleading_wrong_subject` 2. The report
uses `object_detection_hard` as an alias for `colleague_obj_detection_hard`.
Neither count is a localization-accuracy metric.

Concordant cases are cases where multiple metric families agree, such as low
entropy plus high top-5 mass plus low effective area for concentrated
attribution. Discordant cases are cases where output style or metric families
disagree, such as bbox/location-style output with diffuse TAM behavior.

Prompt separability in v6 means metric-space separability: a classifier can
predict prompt condition from TAM/text-derived features above chance under
grouped image splits. It does not mean better reasoning, correct grounding or
causal faithfulness.

The COCO image-structure join uses local annotation metadata such as object
count, category count, largest object area ratio and broad category flags. It is
descriptive context for exploratory image-structure analysis, not a
ground-truth localization evaluation of TAM heatmaps.

COCO boxes shown in model-location overlays are qualitative references selected
by simple label matching when possible. The separate
`/analysis/location-validation` view evaluates the 102 parseable generated
coordinate outputs against COCO boxes and available segmentation masks. It
reports target-category matches, valid alternative-object matches, ambiguous
cases and background/wrong cases. This validates explicit response coordinates;
it does not validate TAM causal faithfulness or replace perturbative grounding
metrics.
