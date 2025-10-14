from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".rst", ".txt"}


class ExistingDocsCollector:
    """Collects existing documentation to feed into the pipeline as supplemental context."""

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root

    def collect(self) -> Dict[Path, str]:
        docs: dict[Path, str] = {}
        for ext in SUPPORTED_EXTENSIONS:
            for path in self._repo_root.rglob(f"*{ext}"):
                if "generated" in path.parts:
                    continue
                try:
                    docs[path] = path.read_text(encoding="utf-8")
                except OSError as exc:
                    logger.warning("Failed to read documentation file %s: %s", path, exc)
        return docs
