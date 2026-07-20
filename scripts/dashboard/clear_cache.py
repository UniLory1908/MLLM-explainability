from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dashboard.cache import clear_dashboard_cache
from scripts.dashboard.config import DashboardConfig
from scripts.dashboard.db import connect, initialize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safely clear generated dashboard cache artifacts.")
    parser.add_argument("--artifact-type", help="Optional exact artifact_type from cache_manifest.")
    parser.add_argument("--yes", action="store_true", help="Actually delete cached dashboard artifacts.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = DashboardConfig.default()
    conn = connect(config.db_path)
    initialize(conn, rebuild=False)
    count = conn.execute(
        "SELECT COUNT(*) AS n, COALESCE(SUM(size_bytes), 0) AS bytes FROM cache_manifest WHERE (? IS NULL OR artifact_type=?)",
        (args.artifact_type, args.artifact_type),
    ).fetchone()
    print(f"[cache] matching_artifacts={count['n']} size_mb={float(count['bytes']) / 1024 / 1024:.2f}")
    print(f"[cache] cache_dir={config.cache_dir}")
    if not args.yes:
        print("[cache] dry run only; pass --yes to delete dashboard cache artifacts")
        return 0
    removed, bytes_removed = clear_dashboard_cache(conn, config, artifact_type=args.artifact_type)
    print(f"[cache] removed={removed} size_mb={bytes_removed / 1024 / 1024:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
