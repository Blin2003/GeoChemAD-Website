from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    return project_root().parent / "all_data"


@dataclass
class ProjectPaths:
    root: Path = field(default_factory=project_root)
    data_dir: Path = field(default_factory=default_data_dir)
    artifact_dir: Path = field(default_factory=lambda: project_root() / "artifacts")
    web_dir: Path = field(default_factory=lambda: project_root() / "web")
    upload_dir: Path = field(default_factory=lambda: project_root() / "artifacts" / "uploads")

    def ensure(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        (self.artifact_dir / "runs").mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
