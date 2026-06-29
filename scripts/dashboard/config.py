from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DashboardConfig:
    project_root: Path
    source_root: Path
    index_dir: Path
    cache_dir: Path
    db_path: Path
    cache_version: str = "dashboard-v1"

    @classmethod
    def default(cls) -> "DashboardConfig":
        root = find_project_root()
        index_dir = root / "outputs" / "dashboard_index"
        cache_dir = root / "outputs" / "dashboard_cache"
        return cls(
            project_root=root,
            source_root=root / "outputs" / "prompt_sensitivity",
            index_dir=index_dir,
            cache_dir=cache_dir,
            db_path=index_dir / "tam_index.sqlite",
        )

    def ensure_dirs(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


def find_project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "README.md").exists() and (candidate / "scripts").exists():
            return candidate
    return Path(__file__).resolve().parents[2]


def to_project_rel(path: str | Path, project_root: Path | None = None) -> str:
    root = project_root or find_project_root()
    value = Path(path)
    try:
        return value.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def resolve_project_path(path: str | Path, project_root: Path | None = None) -> Path:
    root = project_root or find_project_root()
    value = Path(path)
    if value.is_absolute() and value.exists():
        return value
    candidate = root / value
    if candidate.exists():
        return candidate
    return value
