from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .analysis import CodeElement, ProjectAnalyzer, ProjectStructure
from .config import GenDocConfig, RunContext
from .existing_docs import ExistingDocsCollector
from .prompting import DEFAULT_TEMPLATES, MockLLM, OpenAIClient, PromptLedger, PromptOrchestrator

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Section:
    title: str
    body: str
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class RunSummary:
    sections: List[Section]
    element_count: int
    existing_doc_count: int
    ledger_path: Path
    cross_reference_path: Path
    document_path: Path
    project_goal: str
    component_documents: List[Path]


@dataclass(slots=True)
class ComponentDocument:
    name: str
    path: Path
    relative_path: str
    summary: str
    content: str


class Pipeline:
    """Coordinates the end-to-end documentation workflow."""

    def __init__(self, config: GenDocConfig) -> None:
        self._config = config
        self._context = RunContext(config=config)
        self._config.ensure_output_dirs()
        self._context.init_paths()
        ledger_path = config.ledger_path or (config.output_path.parent / "prompt-ledger.jsonl")
        self._ledger = PromptLedger(ledger_path)
        self._analyzer = ProjectAnalyzer(config)
        self._docs_collector = ExistingDocsCollector(config.repo_path)
        self._orchestrator = self._build_orchestrator()

    def _build_orchestrator(self) -> PromptOrchestrator:
        if self._config.llm_provider == "openai":
            client = OpenAIClient()
        else:
            client = MockLLM()
        return PromptOrchestrator(client=client, ledger=self._ledger, templates=DEFAULT_TEMPLATES)

    def run(self) -> RunSummary:
        logger.info("Starting GenDoc pipeline")
        structure = self._analyzer.analyze()
        existing_docs = self._docs_collector.collect() if self._config.use_existing_docs else {}
        element_sections = self._document_elements(structure.elements, existing_docs)
        project_goal = self._derive_project_goal(element_sections)
        refined_sections = self._refine_sections(element_sections, project_goal)
        if not refined_sections:
            refined_sections = element_sections
        cross_reference_path = self._write_cross_reference(existing_docs, refined_sections, project_goal)
        document_text = self._compose_document(structure, refined_sections, project_goal)
        run_section = self._generate_run_section(existing_docs=existing_docs, goal=project_goal)
        if run_section:
            document_text = self._insert_run_section(document_text, run_section)
        component_docs = self._generate_component_documents(
            element_sections=element_sections,
            goal=project_goal,
        )
        if component_docs:
            document_text = self._integrate_component_docs(
                document_text=document_text,
                component_docs=component_docs,
            )
        self._write_output(document_text)

        summary = RunSummary(
            sections=refined_sections,
            element_count=len(structure.elements),
            existing_doc_count=len(existing_docs),
            ledger_path=self._config.ledger_path or (self._config.output_path.parent / "prompt-ledger.jsonl"),
            cross_reference_path=cross_reference_path,
            document_path=self._config.output_path,
            project_goal=project_goal,
            component_documents=[doc.path for doc in component_docs],
        )
        logger.info("GenDoc pipeline finished: %d elements", summary.element_count)
        return summary

    def _document_elements(self, elements: List[CodeElement], existing_docs: Dict[Path, str]) -> List[Section]:
        sections: list[Section] = []
        for element in elements:
            metadata = {
                "identifier": element.identifier,
                "kind": element.kind,
                "path": str(element.file_path.relative_to(self._config.repo_path)),
                "start_line": element.start_line,
                "end_line": element.end_line,
                "dependencies": element.dependencies,
            }
            supplemental = self._find_existing_doc_snippets(element, existing_docs)
            supplemental_text = None
            if supplemental:
                metadata["supplemental_docs"] = [
                    {"path": str(path), "excerpt": excerpt} for path, excerpt in supplemental
                ]
                supplemental_text = "\n\n".join(
                    f"Source: {path}\n{excerpt}" for path, excerpt in supplemental
                )

            analyst_result = self._orchestrator.summarize_element(
                element_source=element.source,
                metadata=metadata,
                supplemental_text=supplemental_text,
            )

            if self._config.enable_reviewer:
                analyst_result = self._orchestrator.review_summary(
                    element_source=element.source,
                    summary=analyst_result,
                    metadata=metadata,
                    supplemental_text=supplemental_text,
                )

            sections.append(
                Section(
                    title=element.identifier,
                    body=analyst_result,
                    metadata=metadata,
                )
            )
        return sections

    def _find_existing_doc_snippets(self, element: CodeElement, existing_docs: Dict[Path, str]) -> list[tuple[Path, str]]:
        related: list[tuple[Path, str]] = []
        element_rel = element.file_path.relative_to(self._config.repo_path)
        element_parent = element_rel.parent
        for path, content in existing_docs.items():
            rel_path = path.relative_to(self._config.repo_path)
            same_directory = rel_path.parent == element_parent
            name_matches = path.stem in element.identifier
            if same_directory or name_matches:
                excerpt = content[:500]
                related.append((path, excerpt))
        return related

    def _derive_project_goal(self, sections: List[Section]) -> str:
        summaries = [section.body for section in sections if section.body.strip()]
        if not summaries:
            return "Project goal unavailable."
        goal = self._orchestrator.derive_project_goal(section_summaries=summaries).strip()
        return goal or "Project goal unavailable."

    def _refine_sections(self, sections: List[Section], goal: str) -> List[Section]:
        refined: list[Section] = []
        for section in sections:
            refined_text = self._orchestrator.refine_section(
                section_body=section.body,
                goal=goal,
                metadata=section.metadata,
            ).strip()
            if not refined_text or refined_text.lower().startswith("omit"):
                continue
            refined.append(
                Section(
                    title=section.title,
                    body=refined_text,
                    metadata=section.metadata,
                )
            )
        return refined

    def _write_cross_reference(self, existing_docs: Dict[Path, str], sections: List[Section], goal: str) -> Path:
        cross_ref = {}
        for section in sections:
            supplemental = section.metadata.get("supplemental_docs", [])
            if supplemental:
                cross_ref[section.title] = supplemental
        if existing_docs:
            cross_ref["_doc_sources"] = [str(path) for path in existing_docs.keys()]
        cross_ref["_project_goal"] = goal
        output_path = self._config.output_path.parent / "cross-reference.json"
        output_path.write_text(json.dumps(cross_ref, indent=2), encoding="utf-8")
        return output_path

    def _compose_document(self, structure: ProjectStructure, sections: List[Section], goal: str) -> str:
        payload: list[dict[str, object]] = []
        for section in sections:
            entry: dict[str, object] = {"title": section.title, "summary": section.body}
            for key in ("identifier", "kind", "path", "dependencies"):
                value = section.metadata.get(key)
                if value:
                    entry[key] = value
            payload.append(entry)
        document = self._orchestrator.compose_document(
            goal=goal,
            project_tree=structure.tree_repr,
            repo_name=self._config.repo_path.name,
            section_payload=payload,
        )
        return document.strip()

    def _write_output(self, document: str) -> None:
        self._config.output_path.write_text(document.strip() + "\n", encoding="utf-8")

    def _generate_component_documents(
        self,
        *,
        element_sections: List[Section],
        goal: str,
    ) -> List[ComponentDocument]:
        if not element_sections:
            return []
        component_dir = self._config.output_path.parent / f"{self._config.output_path.stem}_components"
        component_dir.mkdir(parents=True, exist_ok=True)
        grouped: Dict[str, List[Section]] = {}
        for section in element_sections:
            component_key = str(section.metadata.get("path") or section.title).replace("\\", "/")
            grouped.setdefault(component_key, []).append(section)

        generated: list[ComponentDocument] = []
        slug_counts: dict[str, int] = {}
        repo_name = self._config.repo_path.name
        for component_key, sections in grouped.items():
            raw_notes_parts = []
            for section in sections:
                raw_notes_parts.append(f"### {section.title}\n{section.body.strip()}")
            raw_notes = "\n\n".join(raw_notes_parts).strip()
            if not raw_notes:
                continue
            highlight_snippets = self._select_highlight_snippets(sections)
            condensed = self._orchestrator.reduce_component_notes(
                goal=goal,
                component_name=component_key,
                raw_notes=raw_notes,
            ).strip()
            if not condensed:
                continue
            polished = self._orchestrator.polish_component_document(
                goal=goal,
                component_name=component_key,
                repo_name=repo_name,
                condensed_notes=condensed,
                highlight_snippets=highlight_snippets,
            ).strip()
            polished = self._strip_wrapping_code_fence(polished)
            if not polished:
                continue
            slug_base = self._slugify_component(component_key)
            count = slug_counts.get(slug_base, 0)
            slug_counts[slug_base] = count + 1
            filename = f"{slug_base}{'' if count == 0 else f'-{count+1}'}.md"
            path = component_dir / filename
            path.write_text(polished + "\n", encoding="utf-8")
            relative_path = f"{component_dir.name}/{filename}"
            summary_line = "Further details available."
            for line in condensed.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("- "):
                    stripped = stripped[2:].strip()
                summary_line = stripped or summary_line
                break
            summary_line = summary_line.replace("\\", "/")
            generated.append(
                ComponentDocument(
                    name=component_key,
                    path=path,
                    relative_path=relative_path,
                    summary=summary_line.lstrip("- "),
                    content=polished,
                )
            )
        return generated

    def _integrate_component_docs(
        self,
        *,
        document_text: str,
        component_docs: List[ComponentDocument],
    ) -> str:
        return self._merge_component_deep_links(document_text, component_docs)

    def _slugify_component(self, component_key: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "-", component_key.strip().lower()).strip("-")
        return slug or "component"

    def _select_highlight_snippets(self, sections: List[Section]) -> List[Dict[str, object]]:
        candidates: list[tuple[int, Dict[str, object]]] = []
        for section in sections:
            metadata = section.metadata
            path_str = metadata.get("path")
            if not isinstance(path_str, str):
                continue
            try:
                start_line = int(metadata.get("start_line", 0))
                end_line = int(metadata.get("end_line", 0))
            except (TypeError, ValueError):
                continue
            if start_line <= 0 or end_line <= 0 or end_line < start_line:
                continue
            line_count = end_line - start_line + 1
            dependencies = metadata.get("dependencies") or []
            dep_count = len(dependencies) if isinstance(dependencies, list) else 0
            if line_count < 18 and dep_count < 4:
                continue
            file_path = self._config.repo_path / path_str
            if not file_path.exists():
                continue
            try:
                code_lines = file_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            start_idx = max(start_line - 1, 0)
            end_idx = min(len(code_lines), start_idx + 20, end_line)
            if end_idx <= start_idx:
                continue
            snippet_lines = code_lines[start_idx:end_idx]
            code_excerpt = "\n".join(snippet_lines).strip()
            if not code_excerpt:
                continue
            score = line_count + dep_count * 5
            candidates.append(
                (
                    score,
                    {
                        "title": section.title,
                        "path": path_str,
                        "start_line": start_line,
                        "end_line": start_idx + len(snippet_lines),
                        "code": code_excerpt,
                    },
                )
            )
        if not candidates:
            return []
        candidates.sort(key=lambda item: item[0], reverse=True)
        top_snippets: List[Dict[str, object]] = []
        for _, snippet in candidates[:2]:
            top_snippets.append(snippet)
        return top_snippets

    def _strip_wrapping_code_fence(self, text: str) -> str:
        if not text:
            return text
        stripped = text.strip()
        fence_match = re.match(r"^```([^\n]*)\n(?P<body>.*)\n```$", stripped, re.DOTALL)
        if not fence_match:
            return stripped
        language = (fence_match.group(1) or "").strip().lower()
        if language and language not in {"markdown", "md", "text", "txt"}:
            return stripped
        body = fence_match.group("body").rstrip()
        return body or stripped

    def _generate_run_section(self, *, existing_docs: Dict[Path, str], goal: str) -> str:
        context_entries = self._build_run_context(existing_docs)
        if not context_entries:
            return ""
        try:
            draft = self._orchestrator.collect_run_steps(goal=goal, context_entries=context_entries)
        except Exception as exc:  # pragma: no cover - resilience guard
            logger.warning("Failed to draft run instructions: %s", exc)
            return ""
        sanitized = draft.strip()
        try:
            parsed = json.loads(sanitized)
        except json.JSONDecodeError:
            normalized = sanitized
        else:
            normalized = json.dumps(parsed, ensure_ascii=False, indent=2)
        try:
            refined = self._orchestrator.refine_run_steps(goal=goal, draft=normalized)
        except Exception as exc:  # pragma: no cover - resilience guard
            logger.warning("Failed to refine run instructions: %s", exc)
            return ""
        return refined.strip()

    def _build_run_context(self, existing_docs: Dict[Path, str]) -> List[Dict[str, str]]:
        repo_root = self._config.repo_path
        entries: list[dict[str, str]] = []
        seen: set[str] = set()

        def add_entry(path: Path, content: str) -> None:
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
            entries.append({"path": key, "excerpt": excerpt[:4000]})
            seen.add(key)

        doc_keywords = ("readme", "usage", "getting-started", "setup", "install", "run", "quickstart")
        for path, content in existing_docs.items():
            name = path.name.lower()
            if any(keyword in name for keyword in doc_keywords):
                add_entry(path, content)

        fallback_docs = [
            repo_root / "README.md",
            repo_root / "docs" / "README.md",
            repo_root / "samples" / "demo_app" / "README.md",
        ]
        for doc_path in fallback_docs:
            if not doc_path.exists():
                continue
            try:
                add_entry(doc_path, doc_path.read_text(encoding="utf-8"))
            except OSError as exc:  # pragma: no cover - filesystem guard
                logger.debug("Failed to read fallback doc %s: %s", doc_path, exc)

        source_paths = [
            repo_root / "pyproject.toml",
            repo_root / "src" / "gendoc" / "cli.py",
        ]
        for source_path in source_paths:
            if not source_path.exists():
                continue
            try:
                add_entry(source_path, source_path.read_text(encoding="utf-8"))
            except OSError as exc:  # pragma: no cover - filesystem guard
                logger.debug("Failed to read source context %s: %s", source_path, exc)

        return entries[:8]

    def _insert_run_section(self, document: str, run_section: str) -> str:
        section_text = run_section.strip()
        if not section_text:
            return document
        if "## To Run" not in section_text:
            section_text = "## To Run\n" + section_text
        cleaned = re.sub(r"\n## To Run\b.*?(?=\n## |\Z)", "", document, flags=re.S)
        flow_match = re.search(r"(## Functional Flow\b.*?)(?=\n## |\Z)", cleaned, flags=re.S)
        if not flow_match:
            return cleaned.rstrip() + "\n\n" + section_text + "\n"
        insert_pos = flow_match.end()
        return (
            cleaned[:insert_pos].rstrip()
            + "\n\n"
            + section_text
            + "\n\n"
            + cleaned[insert_pos:].lstrip()
        )

    def _merge_component_deep_links(self, document_text: str, component_docs: List[ComponentDocument]) -> str:
        if not component_docs:
            return document_text
        pattern = re.compile(r"(## Component Breakdown\b)(?P<body>.*?)(?=\n## |\Z)", re.S)
        match = pattern.search(document_text)
        if not match:
            return document_text
        section_text = match.group(0)
        merged_section = self._augment_component_breakdown(section_text, component_docs)
        return document_text[:match.start()] + merged_section + document_text[match.end():]

    def _augment_component_breakdown(self, section_text: str, component_docs: List[ComponentDocument]) -> str:
        lines = section_text.splitlines()
        if not lines:
            return section_text
        header = lines[0]
        remainder = lines[1:]
        grouped: Dict[str, List[ComponentDocument]] = {}
        for doc in component_docs:
            key = self._normalize_component_key(doc.name)
            grouped.setdefault(key, []).append(doc)

        result_lines: list[str] = [header]
        pending_docs: List[ComponentDocument] = []
        for idx, line in enumerate(remainder):
            stripped = line.strip()
            if stripped.startswith("### "):
                if pending_docs:
                    if result_lines and result_lines[-1].strip():
                        result_lines.append("")
                    result_lines.extend(self._format_component_link_lines(pending_docs))
                    pending_docs = []
                result_lines.append(line)
                heading = stripped[4:].strip()
                key = self._normalize_component_key(heading)
                pending_docs = grouped.pop(key, [])
            else:
                result_lines.append(line)
            next_line = remainder[idx + 1] if idx + 1 < len(remainder) else None
            if pending_docs and (next_line is None or next_line.strip().startswith("### ")):
                if result_lines and result_lines[-1].strip():
                    result_lines.append("")
                result_lines.extend(self._format_component_link_lines(pending_docs))
                pending_docs = []
        if pending_docs:
            if result_lines and result_lines[-1].strip():
                result_lines.append("")
            result_lines.extend(self._format_component_link_lines(pending_docs))

        leftover_docs: list[ComponentDocument] = []
        for docs in grouped.values():
            leftover_docs.extend(docs)
        if leftover_docs:
            if result_lines and result_lines[-1].strip():
                result_lines.append("")
            result_lines.append("### Additional Components")
            result_lines.extend(self._format_component_link_lines(leftover_docs))

        if result_lines and result_lines[-1].strip():
            result_lines.append("")
        return "\n".join(result_lines)

    def _normalize_component_key(self, value: str) -> str:
        cleaned = (value or "").replace("\\", "/").strip()
        if "/" in cleaned:
            cleaned = cleaned.split("/")[-1]
        return cleaned.lower()

    def _component_display_name(self, doc: ComponentDocument) -> str:
        base = doc.name.replace("\\", "/").split("/")[-1]
        return base or doc.name

    def _format_component_link_lines(self, docs: List[ComponentDocument]) -> List[str]:
        lines: list[str] = []
        for doc in docs:
            display = self._component_display_name(doc)
            summary = self._clean_summary(doc.summary)
            link = f"[{display}]({doc.relative_path})"
            lines.append(f"- Deep dive: {link} — {summary}")
        return lines

    def _clean_summary(self, summary: str | None) -> str:
        if not summary:
            return "Further details available."
        cleaned = summary.strip().lstrip("- ").strip()
        return cleaned or "Further details available."

