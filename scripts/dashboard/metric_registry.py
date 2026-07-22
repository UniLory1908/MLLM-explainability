from __future__ import annotations

from typing import Any


GROUP_DESCRIPTIONS = {
    "intensity": "These metrics describe overall TAM intensity and raw value distribution.",
    "concentration": "These metrics describe whether TAM mass is focused or diffuse.",
    "spatial": "These metrics describe where saliency mass is located and how wide or elongated it is.",
    "hotspot": "These metrics describe dominant regions and multi-peak behavior.",
    "pairwise_similarity": "These metrics compare two heatmaps as distributions and structures.",
    "pairwise_distance": "These metrics compare absolute numeric differences between map values.",
    "pairwise_overlap": "These metrics compare overlap of top saliency hotspots.",
    "pairwise_shift": "These metrics describe spatial movement of saliency focus between two maps.",
    "layer_scanpath": "These metrics describe how one word's focus moves across layers.",
    "word_scanpath": "These metrics describe how focus moves across generated words at a fixed layer.",
    "condition_aggregate": "These metrics summarize condition-level differences with robust aggregates.",
}


def _m(
    key: str,
    label: str,
    category: str,
    short: str,
    high: str = "",
    low: str = "",
    why: str = "",
    source: str = "",
    limits: str = "",
    undefined_when: str = "",
) -> tuple[str, dict[str, str]]:
    return key, {
        "key": key,
        "label": label,
        "category": category,
        "short": short,
        "high": high,
        "low": low,
        "why": why,
        "source": source,
        "limits": limits,
        "undefined_when": undefined_when,
    }


METRIC_REGISTRY: dict[str, dict[str, str]] = dict(
    [
        _m("energy_sum", "Energy sum", "Map intensity / raw stats", "Sum of heatmap values.", "More total attribution mass."),
        _m("energy_mean", "Energy mean", "Map intensity / raw stats", "Average heatmap value.", "Higher average activation."),
        _m("min_value", "Min value", "Map intensity / raw stats", "Minimum map value."),
        _m("max_value", "Max value", "Map intensity / raw stats", "Maximum map value.", "Stronger peak activation."),
        _m("std_value", "Std value", "Map intensity / raw stats", "Standard deviation of values.", "More contrast / heterogeneity."),
        _m("nonzero_ratio", "Nonzero ratio", "Map intensity / raw stats", "Fraction of nonzero pixels.", "Broader activation support."),
        _m("entropy_norm", "Normalized entropy", "Concentration / diffusion", "How diffuse the heatmap is.", "More diffuse / less focused.", "More concentrated.", "Detect broad vs focused attribution.", "Information entropy", "Does not capture location."),
        _m("top_1_mass", "Top 1% mass", "Concentration / diffusion", "Mass inside top 1% pixels.", "More concentrated peaks."),
        _m("top_5_mass", "Top 5% mass", "Concentration / diffusion", "Mass inside top 5% pixels.", "More concentrated."),
        _m("top_10_mass", "Top 10% mass", "Concentration / diffusion", "Mass inside top 10% pixels.", "More concentrated."),
        _m("hhi", "HHI", "Concentration / diffusion", "Concentration index over prob map.", "Mass concentrated on fewer pixels."),
        _m("effective_area", "Effective area", "Concentration / diffusion", "Effective support size from concentration.", "More spread.", "More focus."),
        _m("effective_area_norm", "Effective area norm", "Concentration / diffusion", "Effective area normalized by image area.", "More spread."),
        _m("hoyer_sparsity", "Hoyer sparsity", "Concentration / diffusion", "Sparsity index on flattened map.", "Sparser / peakier."),
        _m("global_centroid_x_px", "Global centroid x (px)", "Spatial / centroid geometry", "Saliency-weighted centroid X in pixels."),
        _m("global_centroid_y_px", "Global centroid y (px)", "Spatial / centroid geometry", "Saliency-weighted centroid Y in pixels."),
        _m("global_centroid_x_norm", "Global centroid x (norm)", "Spatial / centroid geometry", "Centroid X normalized [0,1]."),
        _m("global_centroid_y_norm", "Global centroid y (norm)", "Spatial / centroid geometry", "Centroid Y normalized [0,1]."),
        _m("spread_trace", "Spread trace", "Spatial / centroid geometry", "Total second-moment spread around centroid.", "More spatially spread activation."),
        _m("spread_x", "Spread x", "Spatial / centroid geometry", "Variance-like spread along X."),
        _m("spread_y", "Spread y", "Spatial / centroid geometry", "Variance-like spread along Y."),
        _m("covariance_xy", "Covariance xy", "Spatial / centroid geometry", "Covariance between X and Y."),
        _m("anisotropy", "Anisotropy", "Spatial / centroid geometry", "Directional elongation of map mass.", "More elongated."),
        _m("peak_count", "Peak count", "Hotspot / region / multipeak", "Number of salient connected peaks.", "More multi-peak behavior."),
        _m("primary_region_mass", "Primary region mass", "Hotspot / region / multipeak", "Mass captured by top-ranked region."),
        _m("secondary_region_mass", "Secondary region mass", "Hotspot / region / multipeak", "Mass of second region if present."),
        _m("secondary_primary_ratio", "Secondary/primary ratio", "Hotspot / region / multipeak", "Secondary mass divided by primary mass.", "Less single-region dominance."),
        _m("primary_region_centroid_x_px", "Primary centroid x (px)", "Hotspot / region / multipeak", "Primary-region centroid X in pixels."),
        _m("primary_region_centroid_y_px", "Primary centroid y (px)", "Hotspot / region / multipeak", "Primary-region centroid Y in pixels."),
        _m("primary_region_centroid_x_norm", "Primary centroid x (norm)", "Hotspot / region / multipeak", "Primary-region centroid X normalized."),
        _m("primary_region_centroid_y_norm", "Primary centroid y (norm)", "Hotspot / region / multipeak", "Primary-region centroid Y normalized."),
        _m("secondary_region_centroid_x_norm", "Secondary centroid x (norm)", "Hotspot / region / multipeak", "Secondary-region centroid X normalized."),
        _m("secondary_region_centroid_y_norm", "Secondary centroid y (norm)", "Hotspot / region / multipeak", "Secondary-region centroid Y normalized."),
        _m("is_multipeak", "Is multipeak", "Hotspot / region / multipeak", "Binary flag for multiple strong regions."),
        _m("cosine_similarity", "Cosine similarity", "Pairwise map similarity", "Similarity of flattened map vectors.", "More similar."),
        _m("pearson_correlation", "Pearson correlation", "Pairwise map similarity", "Linear correlation of normalized values.", "More similar trend."),
        _m("ssim", "SSIM", "Pairwise map similarity", "Structural similarity after min-max normalization.", "More structurally similar."),
        _m("jsd", "Jensen-Shannon divergence", "Pairwise map similarity", "Distribution divergence over probability maps.", "More different mass distribution.", "More similar."),
        _m("emd_2d", "EMD 2D", "Pairwise map similarity", "Earth mover distance on downsampled grids.", "Mass moved farther."),
        _m("l1_distance", "L1 distance", "Pixel/value distances", "Absolute-value difference sum.", "Larger value mismatch."),
        _m("l2_distance", "L2 distance", "Pixel/value distances", "Euclidean difference norm.", "Larger value mismatch."),
        _m("top_1_iou", "Top 1% IoU", "Hotspot overlap / spatial shift", "IoU of top-1% masks.", "More hotspot overlap."),
        _m("top_5_iou", "Top 5% IoU", "Hotspot overlap / spatial shift", "IoU of top-5% masks.", "More hotspot overlap."),
        _m("top_10_iou", "Top 10% IoU", "Hotspot overlap / spatial shift", "IoU of top-10% masks.", "More hotspot overlap."),
        _m("hotspot_iou_percentile_90", "Hotspot IoU p90", "Hotspot overlap / spatial shift", "IoU of percentile-90 hotspot masks.", "More overlap."),
        _m("hotspot_iou_percentile_95", "Hotspot IoU p95", "Hotspot overlap / spatial shift", "IoU of percentile-95 hotspot masks.", "More overlap."),
        _m("hausdorff_peak_distance", "Hausdorff peak distance", "Hotspot overlap / spatial shift", "Distance between peak sets.", "Peaks farther apart."),
        _m("argmax_distance", "Argmax distance", "Hotspot overlap / spatial shift", "Distance between max-activation pixels.", "Larger focus shift."),
        _m("global_centroid_shift", "Global centroid shift", "Hotspot overlap / spatial shift", "Distance between global centroids.", "Larger spatial shift."),
        _m("primary_centroid_shift", "Primary centroid shift", "Hotspot overlap / spatial shift", "Distance between primary-region centroids.", "Larger shift."),
        _m("spread_delta", "Spread delta", "Hotspot overlap / spatial shift", "Difference in spread_trace."),
        _m("anisotropy_delta", "Anisotropy delta", "Hotspot overlap / spatial shift", "Difference in anisotropy."),
        _m("radial_profile_distance", "Radial profile distance", "Hotspot overlap / spatial shift", "Difference in radial mass profiles.", "Different center-out structure."),
        _m("layer_path_length", "Layer path length", "Layer-wise scanpath metrics", "Total centroid path over layers.", "More motion across layers."),
        _m("layer_mean_step", "Layer mean step", "Layer-wise scanpath metrics", "Mean step size across layer transitions."),
        _m("layer_max_jump", "Layer max jump", "Layer-wise scanpath metrics", "Largest layer-to-layer centroid jump.", "Less stable transition."),
        _m("layer_net_displacement", "Layer net displacement", "Layer-wise scanpath metrics", "Start-to-end displacement."),
        _m("layer_tortuosity", "Layer tortuosity", "Layer-wise scanpath metrics", "Path length / net displacement.", "More zig-zag."),
        _m("layer_large_jump_count", "Layer large jump count", "Layer-wise scanpath metrics", "Count of large transitions."),
        _m("layer_bbox_area", "Layer bbox area", "Layer-wise scanpath metrics", "Area of centroid trajectory bbox."),
        _m("adjacent_layer_cosine_mean", "Adjacent cosine mean", "Layer-wise scanpath metrics", "Mean cosine between adjacent layers."),
        _m("adjacent_layer_cosine_min", "Adjacent cosine min", "Layer-wise scanpath metrics", "Worst adjacent-layer cosine."),
        _m("adjacent_layer_ssim_mean", "Adjacent SSIM mean", "Layer-wise scanpath metrics", "Mean adjacent-layer SSIM."),
        _m("adjacent_layer_jsd_mean", "Adjacent JSD mean", "Layer-wise scanpath metrics", "Mean adjacent-layer JSD."),
        _m("adjacent_layer_emd_mean", "Adjacent EMD mean", "Layer-wise scanpath metrics", "Mean adjacent-layer EMD."),
        _m("adjacent_layer_top5_iou_mean", "Adjacent top5 IoU mean", "Layer-wise scanpath metrics", "Mean top-5% IoU between adjacent layers."),
        _m("early_late_cosine", "Early-late cosine", "Layer-wise scanpath metrics", "Similarity between early and late layer maps."),
        _m("early_late_jsd", "Early-late JSD", "Layer-wise scanpath metrics", "Divergence between early and late maps."),
        _m("early_late_centroid_shift", "Early-late centroid shift", "Layer-wise scanpath metrics", "Centroid shift between early/late layer groups."),
        _m("early_late_spread_delta", "Early-late spread delta", "Layer-wise scanpath metrics", "Spread difference between early/late groups."),
        _m("peak_count_mean", "Peak count mean", "Layer-wise scanpath metrics", "Mean peak_count across layers."),
        _m("peak_count_max", "Peak count max", "Layer-wise scanpath metrics", "Maximum peak_count across layers."),
        _m("secondary_primary_ratio_mean", "Secondary ratio mean", "Layer-wise scanpath metrics", "Mean secondary/primary across layers."),
        _m("secondary_primary_ratio_max", "Secondary ratio max", "Layer-wise scanpath metrics", "Maximum secondary/primary across layers."),
        _m("multipeak_layer_count", "Multipeak layer count", "Layer-wise scanpath metrics", "How many layers are multipeak."),
        _m("multipeak_layer_ratio", "Multipeak layer ratio", "Layer-wise scanpath metrics", "Fraction of layers marked multipeak."),
        _m("word_path_length", "Word path length", "Word-wise scanpath metrics", "Total centroid path over generated words.", "More motion across words."),
        _m("word_mean_step", "Word mean step", "Word-wise scanpath metrics", "Mean step over generated words."),
        _m("word_max_jump", "Word max jump", "Word-wise scanpath metrics", "Largest word-to-word centroid jump."),
        _m("word_net_displacement", "Word net displacement", "Word-wise scanpath metrics", "Start-to-end displacement over words."),
        _m("word_tortuosity", "Word tortuosity", "Word-wise scanpath metrics", "Path length / net displacement."),
        _m("word_large_jump_count", "Word large jump count", "Word-wise scanpath metrics", "Count of large word-to-word jumps."),
        _m("word_bbox_area", "Word bbox area", "Word-wise scanpath metrics", "Area of word-scanpath bbox."),
        _m("word_revisit_count", "Word revisit count", "Word-wise scanpath metrics", "Count of revisits to previously attended zones."),
    ]
)


GUIDE_GROUPS = [
    ("A. Map intensity / raw stats", ["energy_sum", "energy_mean", "min_value", "max_value", "std_value", "nonzero_ratio"]),
    ("B. Concentration / diffusion", ["entropy_norm", "top_1_mass", "top_5_mass", "top_10_mass", "hhi", "effective_area", "effective_area_norm", "hoyer_sparsity"]),
    ("C. Spatial / centroid geometry", ["global_centroid_x_px", "global_centroid_y_px", "global_centroid_x_norm", "global_centroid_y_norm", "spread_trace", "spread_x", "spread_y", "covariance_xy", "anisotropy"]),
    ("D. Hotspot / region / multipeak", ["peak_count", "primary_region_mass", "secondary_region_mass", "secondary_primary_ratio", "primary_region_centroid_x_px", "primary_region_centroid_y_px", "primary_region_centroid_x_norm", "primary_region_centroid_y_norm", "secondary_region_centroid_x_norm", "secondary_region_centroid_y_norm", "is_multipeak"]),
    ("E. Pairwise map similarity", ["cosine_similarity", "pearson_correlation", "ssim", "jsd", "emd_2d", "l1_distance", "l2_distance"]),
    ("F. Hotspot overlap / spatial shift", ["top_1_iou", "top_5_iou", "top_10_iou", "hotspot_iou_percentile_90", "hotspot_iou_percentile_95", "hausdorff_peak_distance", "argmax_distance", "global_centroid_shift", "primary_centroid_shift", "spread_delta", "anisotropy_delta", "radial_profile_distance"]),
    ("G. Layer-wise scanpath metrics", ["layer_path_length", "layer_mean_step", "layer_max_jump", "layer_net_displacement", "layer_tortuosity", "layer_large_jump_count", "layer_bbox_area", "adjacent_layer_cosine_mean", "adjacent_layer_cosine_min", "adjacent_layer_ssim_mean", "adjacent_layer_jsd_mean", "adjacent_layer_emd_mean", "adjacent_layer_top5_iou_mean", "early_late_cosine", "early_late_jsd", "early_late_centroid_shift", "early_late_spread_delta", "peak_count_mean", "peak_count_max", "secondary_primary_ratio_mean", "secondary_primary_ratio_max", "multipeak_layer_count", "multipeak_layer_ratio"]),
    ("H. Word-wise scanpath metrics", ["word_path_length", "word_mean_step", "word_max_jump", "word_net_displacement", "word_tortuosity", "word_large_jump_count", "word_bbox_area", "word_revisit_count"]),
]


OPTIONAL_GT = [
    "Pointing Game",
    "Energy-based Pointing",
    "mass_in_mask",
    "mass_outside_mask",
    "mass_in_mask_curve_auc",
    "gt_centroid_distance",
    "gt_region_iou_top5",
]

FUTURE_METRICS = ["deletion AUC", "insertion AUC", "ROAR", "infidelity", "sensitivity", "PDM exact"]


NO_MONOTONIC = "No monotonic interpretation."
POSITION_COORDINATE = "Position coordinate."
SEE_DEFINITION = "See definition."
UNAVAILABLE = "Unavailable when the required map, region or comparison record is missing."


def _default_high_low(key: str, category: str, label: str) -> tuple[str, str]:
    text = f"{key} {label} {category}".lower()
    if "centroid" in text and "shift" not in text:
        return POSITION_COORDINATE, POSITION_COORDINATE
    if any(term in text for term in ("covariance", "delta", "min value", "max value")):
        return NO_MONOTONIC, NO_MONOTONIC
    if "similarity" in text or "cosine" in text or "pearson" in text or "ssim" in text or "iou" in text:
        return "More similar or more overlapping.", "Less similar or less overlapping."
    if "jsd" in text or "emd" in text or "distance" in text or "shift" in text:
        return "Larger difference or spatial movement.", "Smaller difference or spatial movement."
    if "entropy" in text or "effective area" in text or "spread" in text:
        return "More diffuse or spatially broad.", "More concentrated or spatially compact."
    if "top" in text or "hhi" in text or "hoyer" in text or "primary" in text or "secondary" in text or "peak" in text:
        return "More concentrated, sparse or multi-region structure.", "Less concentrated or less multi-region structure."
    if "path" in text or "step" in text or "jump" in text or "tortuosity" in text or "revisit" in text or "bbox area" in text:
        return "More movement or less stable trajectory.", "Less movement or more stable trajectory."
    return NO_MONOTONIC, NO_MONOTONIC


def _default_source(category: str) -> str:
    if "pairwise" in category.lower() or "hotspot overlap" in category.lower():
        return "Derived from aligned TAM map pairs."
    if "scanpath" in category.lower():
        return "Derived from TAM centroids across layers or generated words."
    if "hotspot" in category.lower():
        return "Derived from thresholded connected saliency regions."
    return "Derived from per-map TAM values."


def _default_limits(category: str) -> str:
    if "scanpath" in category.lower():
        return "Depends on valid centroid points and should not be interpreted as human eye tracking."
    if "pairwise" in category.lower() or "hotspot overlap" in category.lower():
        return "Requires comparable maps and does not by itself prove semantic correctness."
    if "spatial" in category.lower():
        return "Undefined for zero-mass maps and sensitive to preprocessing resolution."
    if "hotspot" in category.lower():
        return "Depends on thresholding and connected-component extraction."
    return "Descriptive proxy; not a causal faithfulness metric."


def _default_undefined(key: str, category: str) -> str:
    text = f"{key} {category}".lower()
    if any(term in text for term in ("entropy", "hhi", "effective_area", "centroid", "spread", "anisotropy")):
        return "Zero-mass maps or missing metric records."
    if "secondary" in text:
        return "Requires at least two valid salient regions."
    if "tortuosity" in text:
        return "Requires at least two valid points and nonzero net displacement."
    if "adjacent" in text or "early_late" in text:
        return "Requires aligned comparison maps across layers."
    if "similarity" in text or "distance" in text or "iou" in text or "shift" in text or "delta" in text:
        return "Requires a valid aligned comparison map."
    if "scanpath" in category.lower() or "path" in text or "jump" in text:
        return "Requires enough valid scanpath points."
    return UNAVAILABLE


def metric_info(key: str) -> dict[str, str]:
    if key in METRIC_REGISTRY:
        row = dict(METRIC_REGISTRY[key])
        high, low = _default_high_low(key, row["category"], row["label"])
        row["high"] = row.get("high") or high
        row["low"] = row.get("low") or low
        row["why"] = row.get("why") or "Used to summarize TAM intensity, geometry, concentration or stability in a compact dashboard view."
        row["source"] = row.get("source") or _default_source(row["category"])
        row["limits"] = row.get("limits") or _default_limits(row["category"])
        row["undefined_when"] = row.get("undefined_when") or _default_undefined(key, row["category"])
        return row
    return {
        "key": key,
        "label": key,
        "category": "Other",
        "short": "No description yet.",
        "high": NO_MONOTONIC,
        "low": NO_MONOTONIC,
        "why": SEE_DEFINITION,
        "source": "Metric registry fallback.",
        "limits": UNAVAILABLE,
        "undefined_when": UNAVAILABLE,
    }


def format_metric_value(value: Any) -> str:
    if value is None:
        return "— unavailable"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    abs_v = abs(number)
    if abs_v >= 100000:
        return f"{number:.3g}"
    if abs_v >= 1000:
        return f"{number:,.1f}"
    if abs_v >= 1:
        return f"{number:.3f}"
    return f"{number:.4f}"
