from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.dashboard.config import DashboardConfig, to_project_rel


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_cache_key(kind: str, params: dict[str, Any], source_signature: str, version: str) -> str:
    payload = json.dumps(
        {"kind": kind, "params": params, "source_signature": source_signature, "version": version},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def cache_path(config: DashboardConfig, key: str, suffix: str) -> Path:
    return config.cache_dir / key[:2] / f"{key}{suffix}"


def get_cached(conn: sqlite3.Connection, config: DashboardConfig, key: str) -> Path | None:
    row = conn.execute("SELECT artifact_path FROM cache_manifest WHERE cache_key=?", (key,)).fetchone()
    if not row:
        return None
    path = config.project_root / row["artifact_path"]
    if path.exists():
        conn.execute("UPDATE cache_manifest SET last_accessed_at=? WHERE cache_key=?", (now_iso(), key))
        conn.commit()
        return path
    return None


def put_cached(
    conn: sqlite3.Connection,
    config: DashboardConfig,
    key: str,
    artifact_path: Path,
    artifact_type: str,
    params: dict[str, Any],
    source_paths: list[Path],
    source_signature: str,
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    size = artifact_path.stat().st_size if artifact_path.exists() else 0
    rel_artifact = to_project_rel(artifact_path, config.project_root)
    rel_sources = [to_project_rel(path, config.project_root) for path in source_paths]
    stamp = now_iso()
    conn.execute(
        """
        INSERT OR REPLACE INTO cache_manifest VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (
            key,
            rel_artifact,
            artifact_type,
            json.dumps(params, sort_keys=True),
            json.dumps(rel_sources),
            source_signature,
            stamp,
            size,
            stamp,
        ),
    )
    conn.commit()


def clear_dashboard_cache(conn: sqlite3.Connection, config: DashboardConfig, artifact_type: str | None = None) -> tuple[int, int]:
    rows = conn.execute(
        "SELECT cache_key, artifact_path, size_bytes FROM cache_manifest WHERE (? IS NULL OR artifact_type=?)",
        (artifact_type, artifact_type),
    ).fetchall()
    removed = 0
    bytes_removed = 0
    for row in rows:
        path = config.project_root / row["artifact_path"]
        if path.exists() and path.is_file():
            bytes_removed += int(path.stat().st_size)
            path.unlink()
            removed += 1
        conn.execute("DELETE FROM cache_manifest WHERE cache_key=?", (row["cache_key"],))
    conn.commit()
    # Remove empty two-character cache subdirectories only under dashboard_cache.
    if config.cache_dir.exists():
        for child in sorted(config.cache_dir.iterdir()):
            if child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass
    return removed, bytes_removed
