from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class ProjectPaths:
    root: Path = field(default_factory=project_root)
    artifact_dir: Path = field(default_factory=lambda: project_root() / "artifacts")
    web_dir: Path = field(default_factory=lambda: project_root() / "web")

    def ensure(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
