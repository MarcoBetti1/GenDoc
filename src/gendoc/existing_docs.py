from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from .config import GenDocConfig

logger = logging.getLogger(__name__)


class ExistingDocsCollector:
    """Collects existing documentation to feed into the pipeline as supplemental context."""

    def __init__(self, config: GenDocConfig) -> None:
        self._repo_root = config.repo_path
        self._extensions = tuple(config.existing_docs_settings.extensions)
        self._exclude_markers = tuple(marker.lower() for marker in config.existing_docs_settings.exclude_substrings)
        self._max_excerpt = int(config.existing_docs_settings.max_excerpt_chars)

    def collect(self) -> Dict[Path, str]:
        docs: dict[Path, str] = {}
        for ext in self._extensions:
            pattern = f"*{ext}"
            for path in self._repo_root.rglob(pattern):
                if self._should_skip(path):
                    continue
                try:
                    content = path.read_text(encoding="utf-8")
                except OSError as exc:
                    logger.warning("Failed to read documentation file %s: %s", path, exc)
                    continue
                docs[path] = content[: self._max_excerpt] if self._max_excerpt > 0 else content
        return docs

    def _should_skip(self, path: Path) -> bool:
        normalized = str(path).lower()
        return any(marker in normalized for marker in self._exclude_markers)
