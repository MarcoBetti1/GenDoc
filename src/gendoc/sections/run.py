from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from ..config import GenDocConfig
from ..prompting import PromptOrchestrator

logger = logging.getLogger(__name__)


def generate_to_run_section(
    *,
    config: GenDocConfig,
    orchestrator: PromptOrchestrator,
    existing_docs: Optional[Dict[Path, str]] = None,
    goal: Optional[str] = None,
) -> str:
    """Compose the "To Run" section using repo context and prompt orchestration."""

    context_entries = build_run_context(config=config, existing_docs=existing_docs or {})
    if not context_entries:
        return ""

    effective_goal = (goal or "Provide actionable run instructions aligned with the project.").strip()

    try:
        draft = orchestrator.collect_run_steps(goal=effective_goal, context_entries=context_entries)
    except Exception as exc:  # pragma: no cover - resilience guard
        logger.warning("Failed to draft run instructions: %s", exc)
        return ""

    sanitized = draft.strip()
    normalized = _normalize_run_draft(sanitized)

    try:
        refined = orchestrator.refine_run_steps(goal=effective_goal, draft=normalized)
    except Exception as exc:  # pragma: no cover - resilience guard
        logger.warning("Failed to refine run instructions: %s", exc)
        return sanitized

    return refined.strip()


def build_run_context(*, config: GenDocConfig, existing_docs: Dict[Path, str]) -> List[Dict[str, str]]:
    """Collect candidate snippets that describe how to install or run the project."""

    repo_root = config.repo_path
    settings = config.run_section_settings
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    max_entries = max(0, settings.max_context_entries)
    excerpt_limit = max(200, settings.max_excerpt_chars)

    def add_entry(path: Path, content: str) -> None:
        if max_entries and len(entries) >= max_entries:
            return
        try:
            relative = path.relative_to(repo_root)
        except ValueError:
            relative = path
        key = str(relative).replace("\\", "/")
        if key in seen:
            return
        excerpt = (content or "").strip()
        if not excerpt:
            return
        if "llmchess" in excerpt.lower():
            return
        entries.append({"path": key, "excerpt": excerpt[:excerpt_limit]})
        seen.add(key)

    doc_keywords = tuple(settings.doc_keywords) or (
        "readme",
        "usage",
        "getting-started",
        "setup",
        "install",
        "run",
        "quickstart",
    )
    for path, content in existing_docs.items():
        name = path.name.lower()
        if any(keyword in name for keyword in doc_keywords):
            add_entry(path, content)

    fallback_docs = [repo_root / Path(doc_path) for doc_path in settings.fallback_docs]
    for doc_path in fallback_docs:
        if not doc_path.exists():
            continue
        try:
            add_entry(doc_path, doc_path.read_text(encoding="utf-8"))
        except OSError as exc:  # pragma: no cover - filesystem guard
            logger.debug("Failed to read fallback doc %s: %s", doc_path, exc)

    source_paths = [repo_root / Path(source) for source in settings.source_paths]
    for source_path in source_paths:
        if not source_path.exists():
            continue
        try:
            add_entry(source_path, source_path.read_text(encoding="utf-8"))
        except OSError as exc:  # pragma: no cover - filesystem guard
            logger.debug("Failed to read source context %s: %s", source_path, exc)

    return entries


def _normalize_run_draft(draft: str) -> str:
    if not draft:
        return ""
    try:
        parsed = json.loads(draft)
    except json.JSONDecodeError:
        return draft
    return json.dumps(parsed, ensure_ascii=False, indent=2)
