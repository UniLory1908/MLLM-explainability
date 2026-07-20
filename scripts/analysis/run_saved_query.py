from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_LEVEL = PROJECT_ROOT / "outputs" / "analysis" / "v1_case_level" / "analysis_case_level_v1.parquet"
DEFAULT_QUERY_ROOT = PROJECT_ROOT / "outputs" / "analysis" / "v5_queries"
DEFAULT_DASHBOARD_BASE_URL = "https://lory-fisso.tailbaabf0.ts.net"


QUERY_DESCRIPTIONS = {
    "coordinate_outputs": "Cases where the model output contains coordinate/bbox-style signals instead of or alongside free text.",
    "high_text_visual_drift": "Non-baseline cases with high textual_change and high visual_change.",
    "q3_text_changed_visual_stable": "Cases where text changes but visual sensitivity remains comparatively stable.",
    "q4_visual_changed_text_stable": "Cases where visual sensitivity changes while text remains comparatively stable.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a saved lightweight query over the TAM case-level Parquet.")
    parser.add_argument("--list", action="store_true", help="List available canned queries.")
    parser.add_argument("--query", choices=sorted(QUERY_DESCRIPTIONS), help="Canned query to run.")
    parser.add_argument("--question", help="Human-readable question to save with the query.")
    parser.add_argument("--slug", help="Output folder slug. Defaults to query name plus timestamp.")
    parser.add_argument("--case-level", type=Path, default=DEFAULT_CASE_LEVEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_QUERY_ROOT)
    parser.add_argument("--dashboard-base-url", default=DEFAULT_DASHBOARD_BASE_URL)
    parser.add_argument("--top-n", type=int, default=100)
    return parser


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "query"


def coordinate_outputs(df: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, list[str]]:
    coord_regex = r"\(\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\)|\[\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?"
    mask = (
        df.get("coordinate_token_count", 0).fillna(0).astype(float).gt(0)
        | df.get("bbox_style_output_flag", 0).fillna(0).astype(float).gt(0)
        | df.get("has_box_tokens", 0).fillna(0).astype(float).gt(0)
        | df.get("response_text", "").fillna("").str.contains(coord_regex, regex=True)
    )
    rows = df.loc[mask].copy()
    rows["coordinate_like_score"] = (
        rows.get("bbox_style_output_flag", 0).fillna(0).astype(float)
        + rows.get("has_box_tokens", 0).fillna(0).astype(float)
        + rows.get("coordinate_token_count", 0).fillna(0).astype(float).clip(0, 8) / 8.0
        + rows.get("bbox_or_grounding_format_score", 0).fillna(0).astype(float)
    )
    rows = rows.sort_values(["coordinate_like_score", "coordinate_token_count"], ascending=False).head(top_n)
    notes = [
        "Selection uses exported case-level columns plus a regex over response_text.",
        "This is a navigation/diagnostic query, not a new metric precompute.",
    ]
    return rows, notes


def high_text_visual_drift(df: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, list[str]]:
    rows = df.loc[~df["is_baseline"].astype(bool)].copy()
    rows["joint_drift_score"] = rows["textual_change"].fillna(0) + rows["visual_change"].fillna(0)
    rows = rows.sort_values("joint_drift_score", ascending=False).head(top_n)
    return rows, ["Sorted by textual_change + visual_change over non-baseline rows."]


def q3_text_changed_visual_stable(df: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, list[str]]:
    rows = df.loc[df["quadrant"].eq("Q3_text_changed_visual_stable")].copy()
    rows["q3_score"] = rows["textual_change"].fillna(0) * (1.0 - rows["visual_change"].fillna(0))
    rows = rows.sort_values("q3_score", ascending=False).head(top_n)
    return rows, ["Q3 quadrant from Batch 1: text changes + visual stable."]


def q4_visual_changed_text_stable(df: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, list[str]]:
    rows = df.loc[df["quadrant"].eq("Q4_text_stable_visual_changed")].copy()
    rows["q4_score"] = rows["visual_change"].fillna(0) * (1.0 - rows["textual_change"].fillna(0))
    rows = rows.sort_values("q4_score", ascending=False).head(top_n)
    return rows, ["Q4 quadrant from Batch 1: text stable + visual changes."]


QUERY_FUNCTIONS = {
    "coordinate_outputs": coordinate_outputs,
    "high_text_visual_drift": high_text_visual_drift,
    "q3_text_changed_visual_stable": q3_text_changed_visual_stable,
    "q4_visual_changed_text_stable": q4_visual_changed_text_stable,
}


def dashboard_url(value: object, base_url: str) -> str:
    if pd.isna(value) or not str(value):
        return ""
    text = str(value)
    parsed = urlsplit(text)
    if parsed.scheme and parsed.netloc:
        path = parsed.path or "/"
        suffix = f"{path}?{parsed.query}" if parsed.query else path
        return base_url.rstrip("/") + suffix if base_url else text
    return base_url.rstrip("/") + text if base_url and text.startswith("/") else text


def compact_columns(df: pd.DataFrame, dashboard_base_url: str) -> pd.DataFrame:
    preferred = [
        "image_id",
        "case_id",
        "prompt_label",
        "prompt_category",
        "quadrant",
        "textual_change",
        "visual_change",
        "coordinate_token_count",
        "bbox_style_output_flag",
        "has_object_ref_tokens",
        "has_box_tokens",
        "bbox_or_grounding_format_score",
        "unstable_explanation_candidate_score",
        "prompt_dominated_candidate_score",
        "weak_grounding_candidate_score",
        "multipeak_ambiguity_score",
        "response_text",
        "dashboard_case_url",
        "dashboard_matrix_url",
        "dashboard_compare_url",
    ]
    extra_scores = [c for c in df.columns if c.endswith("_score") and c not in preferred]
    cols = [c for c in preferred + extra_scores if c in df.columns]
    out = df[cols].copy()
    for col in ["dashboard_case_url", "dashboard_matrix_url", "dashboard_compare_url"]:
        if col in out.columns:
            out[col] = out[col].map(lambda value: dashboard_url(value, dashboard_base_url))
    return out


def write_answer(
    out_dir: Path,
    query_name: str,
    question: str,
    rows: pd.DataFrame,
    result: pd.DataFrame,
    notes: list[str],
) -> None:
    by_prompt = (
        rows.groupby(["prompt_label", "prompt_category"], dropna=False)
        .agg(cases=("case_id", "count"), images=("image_id", "nunique"))
        .reset_index()
        .sort_values(["cases", "prompt_label"], ascending=[False, True])
    )
    lines = [
        f"# Query: {query_name}",
        "",
        f"Question: {question}",
        "",
        "## Summary",
        "",
        f"- Matching cases: `{len(rows)}`",
        f"- Matching images: `{rows['image_id'].nunique() if 'image_id' in rows else 0}`",
        "",
        "## By prompt",
        "",
        "| prompt_label | prompt_category | cases | images |",
        "| --- | --- | ---: | ---: |",
    ]
    for _, row in by_prompt.iterrows():
        lines.append(f"| `{row['prompt_label']}` | `{row['prompt_category']}` | {int(row['cases'])} | {int(row['images'])} |")
    lines.extend(["", "## Notes", ""])
    lines.extend([f"- {note}" for note in notes])
    lines.extend(
        [
            "",
            "## Top rows",
            "",
            "| image_id | prompt_label | quadrant | response_text | case_url |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for _, row in result.head(20).iterrows():
        response = str(row.get("response_text", "")).replace("\n", " ")
        if len(response) > 120:
            response = response[:117].rstrip() + "..."
        lines.append(
            f"| {row.get('image_id', '')} | `{row.get('prompt_label', '')}` | `{row.get('quadrant', '')}` | "
            f"{response} | {row.get('dashboard_case_url', '')} |"
        )
    (out_dir / "answer.md").write_text("\n".join(lines), encoding="utf-8")


def run_query(args: argparse.Namespace) -> Path:
    df = pd.read_parquet(args.case_level.resolve())
    query_name = args.query
    assert query_name is not None
    rows, notes = QUERY_FUNCTIONS[query_name](df, args.top_n)
    result = compact_columns(rows, args.dashboard_base_url)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slugify(args.slug or f"{stamp}_{query_name}")
    out_dir = args.output_root.resolve() / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    question = args.question or QUERY_DESCRIPTIONS[query_name]
    (out_dir / "question.md").write_text(f"# Question\n\n{question}\n", encoding="utf-8")
    result.to_csv(out_dir / "results.csv", index=False)
    result.to_parquet(out_dir / "results.parquet", index=False)
    write_answer(out_dir, query_name, question, rows, result, notes)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query": query_name,
        "question": question,
        "source_case_level": str(args.case_level.resolve()),
        "dashboard_base_url": args.dashboard_base_url,
        "output_dir": str(out_dir),
        "matching_cases": int(len(rows)),
        "matching_images": int(rows["image_id"].nunique()) if "image_id" in rows else 0,
        "notes": notes,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def main() -> int:
    args = build_parser().parse_args()
    if args.list:
        for name, description in QUERY_DESCRIPTIONS.items():
            print(f"{name}: {description}")
        return 0
    if not args.query:
        raise SystemExit("Pass --query or --list.")
    out_dir = run_query(args)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
