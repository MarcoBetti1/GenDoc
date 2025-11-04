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
        component_docs = self._generate_component_documents(
            element_sections=element_sections,
            goal=project_goal,
        )
        if component_docs:
            document_text = self._integrate_component_docs(document_text, component_docs, project_goal)
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
            ).strip()
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
        document_text: str,
        component_docs: List[ComponentDocument],
        goal: str,
    ) -> str:
        payload = [
            {
                "name": doc.name,
                "relative_path": doc.relative_path,
                "summary": doc.summary,
            }
            for doc in component_docs
        ]
        integrated = self._orchestrator.integrate_component_links(
            base_document=document_text,
            component_payload=payload,
            goal=goal,
        )
        return integrated.strip()

    def _slugify_component(self, component_key: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "-", component_key.strip().lower()).strip("-")
        return slug or "component"