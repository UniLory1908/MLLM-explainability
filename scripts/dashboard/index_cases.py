from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dashboard.config import DashboardConfig, resolve_project_path
from scripts.dashboard.data_access import build_case_record, discover_metadata, insert_case
from scripts.dashboard.db import connect, initialize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index TAM dashboard cases from outputs/prompt_sensitivity.")
    parser.add_argument("--rebuild", action="store_true", help="Drop and recreate dashboard index tables.")
    parser.add_argument("--dry-run", action="store_true", help="Print discovered cases without writing SQLite.")
    parser.add_argument("--case-filter", help="Substring filter matched against metadata paths.")
    parser.add_argument(
        "--metadata-root",
        action="append",
        help=(
            "Directory or metadata.json file to scan instead of the full source root. "
            "Repeat for multiple pilot/image roots."
        ),
    )
    parser.add_argument("--official-only", action="store_true", help="Index only official V3/fix256 cases.")
    parser.add_argument("--limit-cases", type=int, default=0, help="Limit number of metadata files processed.")
    return parser


def discover_metadata_from_roots(config: DashboardConfig, roots: list[str], case_filter: str | None = None) -> list[Path]:
    paths: list[Path] = []
    needle = case_filter.lower() if case_filter else None
    for raw_root in roots:
        root = resolve_project_path(raw_root, config.project_root)
        if root.is_file():
            candidates = [root] if root.name == "metadata.json" else []
        elif root.is_dir():
            direct = root / "metadata.json"
            candidates = [direct] if direct.exists() else sorted(root.rglob("metadata.json"))
        else:
            raise FileNotFoundError(f"metadata root not found: {raw_root}")
        for path in candidates:
            if needle and needle not in path.as_posix().lower():
                continue
            paths.append(path)
    return sorted(dict.fromkeys(paths))


def main() -> int:
    args = build_parser().parse_args()
    config = DashboardConfig.default()
    if args.metadata_root:
        paths = discover_metadata_from_roots(config, args.metadata_root, args.case_filter)
    else:
        paths = discover_metadata(config, args.case_filter)
    if args.limit_cases > 0:
        paths = paths[: args.limit_cases]

    records = []
    for path in paths:
        record = build_case_record(path, config)
        if record is None:
            continue
        if args.official_only and not record.is_official:
            continue
        records.append(record)

    print(f"[discover] metadata={len(paths)} cases={len(records)} source={config.source_root}")
    for record in records[:10]:
        print(f"[case] {record.case_id} official={record.is_official} words={len(record.words)} layers={len(record.layers)}")
    if len(records) > 10:
        print(f"[case] ... {len(records) - 10} more")

    if args.dry_run:
        return 0

    config.ensure_dirs()
    conn = connect(config.db_path)
    initialize(conn, rebuild=args.rebuild)
    for record in records:
        insert_case(conn, config, record)
    conn.commit()
    print(f"[done] indexed={len(records)} db={config.db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
