from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CSV = PROJECT_ROOT / "outputs" / "analysis" / "v3_case_studies" / "final_case_studies.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "v4_qualitative_review"
FIRST_REVIEW = ["CS09", "CS07", "CS08", "CS02", "CS03", "CS05"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build static qualitative-review index for selected TAM case studies.")
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dashboard-base-url", default="", help="Optional public dashboard base URL for static links.")
    return parser


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def md_link(label: str, url: str) -> str:
    return f"[{label}]({url})" if url else "missing"


def link_url(row: dict[str, str], key: str, dashboard_base_url: str) -> str:
    value = row.get(key, "")
    if not value or not dashboard_base_url:
        return value
    from urllib.parse import urlsplit

    parsed = urlsplit(value)
    path = parsed.path or "/"
    suffix = f"{path}?{parsed.query}" if parsed.query else path
    return dashboard_base_url.rstrip("/") + suffix


def write_markdown(rows: list[dict[str, str]], out_dir: Path, dashboard_base_url: str) -> None:
    first = [row for row in rows if row["case_study_id"] in FIRST_REVIEW]
    first.sort(key=lambda row: FIRST_REVIEW.index(row["case_study_id"]))
    lines = [
        "# Selected Case Studies - qualitative review",
        "",
        "Static navigation page for the 12 Batch 3 case studies. Use it only for qualitative dashboard review.",
        "",
        "## Recommended first review",
        "",
        "| case | image | prompt | links |",
        "| --- | ---: | --- | --- |",
    ]
    for row in first:
        links = " / ".join(
            [
                md_link("case", row.get("dashboard_case_url", "")),
                md_link("matrix", link_url(row, "dashboard_matrix_url", dashboard_base_url)),
                md_link("compare", link_url(row, "dashboard_compare_url", dashboard_base_url)),
            ]
        )
        links = links.replace(row.get("dashboard_case_url", ""), link_url(row, "dashboard_case_url", dashboard_base_url))
        lines.append(f"| `{row['case_study_id']}` | {row['image_id']} | `{row['prompt_label']}` | {links} |")
    lines.extend(
        [
            "",
            "## All selected cases",
            "",
            "| id | image | prompt | pattern | quadrant | cluster | outlier | links | cosa controllare |",
            "| --- | ---: | --- | --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for row in rows:
        links = " / ".join(
            [
                md_link("case", row.get("dashboard_case_url", "")),
                md_link("matrix", link_url(row, "dashboard_matrix_url", dashboard_base_url)),
                md_link("compare", link_url(row, "dashboard_compare_url", dashboard_base_url)),
            ]
        )
        links = links.replace(row.get("dashboard_case_url", ""), link_url(row, "dashboard_case_url", dashboard_base_url))
        lines.append(
            f"| `{row['case_study_id']}` | {row['image_id']} | `{row['prompt_label']}` | "
            f"{row['main_pattern']} | `{row['quadrant']}` | {row['cluster_id']} | "
            f"{row['is_outlier']} | {links} | {row['recommended_dashboard_action']} |"
        )
    (out_dir / "case_studies_dashboard_index.md").write_text("\n".join(lines), encoding="utf-8")


def a(label: str, url: str) -> str:
    if not url:
        return '<span class="missing">missing</span>'
    return f'<a href="{html.escape(url)}">{html.escape(label)}</a>'


def write_html(rows: list[dict[str, str]], out_dir: Path, dashboard_base_url: str) -> None:
    first = [row for row in rows if row["case_study_id"] in FIRST_REVIEW]
    first.sort(key=lambda row: FIRST_REVIEW.index(row["case_study_id"]))
    rows_html = []
    for row in rows:
        priority = " priority" if row["case_study_id"] in FIRST_REVIEW else ""
        rows_html.append(
            "<tr class=\"{}\"><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
            "<td>{}</td><td>{}</td><td>{} {} {}</td><td>{}</td></tr>".format(
                priority.strip(),
                html.escape(row["case_study_id"]),
                html.escape(row["image_id"]),
                html.escape(row["prompt_label"]),
                html.escape(row["main_pattern"]),
                html.escape(row["quadrant"]),
                html.escape(row["cluster_id"]),
                html.escape(row["is_outlier"]),
                a("case", link_url(row, "dashboard_case_url", dashboard_base_url)),
                a("matrix", link_url(row, "dashboard_matrix_url", dashboard_base_url)),
                a("compare", link_url(row, "dashboard_compare_url", dashboard_base_url)),
                html.escape(row["recommended_dashboard_action"]),
            )
        )
    first_links = "\n".join(
        f'<a class="button" href="{html.escape(link_url(row, "dashboard_case_url", dashboard_base_url))}">{html.escape(row["case_study_id"])} {html.escape(row["image_id"])} / {html.escape(row["prompt_label"])}</a>'
        for row in first
    )
    dashboard_note = dashboard_base_url or "http://127.0.0.1:4321/"
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Selected Case Studies - qualitative review</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; background: #f7f7f5; }}
    h1 {{ font-size: 24px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ border: 1px solid #d8dde3; padding: 8px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #eef1f4; }}
    a {{ color: #145ea8; margin-right: 8px; }}
    .button {{ display: inline-block; padding: 7px 9px; margin: 0 8px 8px 0; background: #e9eef2; border: 1px solid #b9c0c8; text-decoration: none; color: #1f2933; }}
    .priority {{ background: #fff9e8; }}
    .note {{ color: #687380; }}
  </style>
</head>
<body>
  <h1>Selected Case Studies - qualitative review</h1>
  <p class="note">Static navigation page for Batch 3. Start/open the dashboard at <code>{html.escape(dashboard_note)}</code> before opening links.</p>
  <h2>Recommended first review</h2>
  <p>{first_links}</p>
  <h2>All selected cases</h2>
  <table>
    <thead><tr><th>ID</th><th>Image</th><th>Prompt</th><th>Pattern</th><th>Quadrant</th><th>Cluster</th><th>Outlier</th><th>Links</th><th>Cosa controllare</th></tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</body>
</html>
"""
    (out_dir / "case_studies_dashboard_index.html").write_text(page, encoding="utf-8")


def write_report(rows: list[dict[str, str]], out_dir: Path, dashboard_base_url: str) -> None:
    missing = [
        {
            "case_study_id": row["case_study_id"],
            "missing": [
                key
                for key in ["dashboard_case_url", "dashboard_matrix_url", "dashboard_compare_url"]
                if not row.get(key)
            ],
        }
        for row in rows
        if any(not row.get(key) for key in ["dashboard_case_url", "dashboard_matrix_url", "dashboard_compare_url"])
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(SOURCE_CSV),
        "case_count": len(rows),
        "dashboard_base_url": dashboard_base_url,
        "first_review": FIRST_REVIEW,
        "missing_links": missing,
        "outputs": [
            "case_studies_dashboard_index.md",
            "case_studies_dashboard_index.html",
            "dashboard_index_update_report.md",
        ],
    }
    lines = [
        "# Dashboard Index Update Report",
        "",
        "- Added a dynamic Selected Case Studies section to the dashboard home template.",
        "- Generated a static Markdown/HTML navigation page under `outputs/analysis/v4_qualitative_review/`.",
        f"- Case count: `{len(rows)}`.",
        f"- Missing link rows: `{len(missing)}`.",
        "",
        "```json",
        json.dumps(report, indent=2),
        "```",
    ]
    (out_dir / "dashboard_index_update_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.source_csv.resolve())
    write_markdown(rows, out_dir, args.dashboard_base_url)
    write_html(rows, out_dir, args.dashboard_base_url)
    write_report(rows, out_dir, args.dashboard_base_url)
    print(f"Wrote qualitative review index for {len(rows)} cases to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
