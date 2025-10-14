from __future__ import annotations

import json
import logging
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
        synthesis_section = self._synthesize(structure, element_sections)
        cross_reference_path = self._write_cross_reference(existing_docs, element_sections)
        sections = element_sections + [synthesis_section]
        self._write_output(sections)

        summary = RunSummary(
            sections=sections,
            element_count=len(structure.elements),
            existing_doc_count=len(existing_docs),
            ledger_path=self._config.ledger_path or (self._config.output_path.parent / "prompt-ledger.jsonl"),
            cross_reference_path=cross_reference_path,
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

    def _synthesize(self, structure: ProjectStructure, element_sections: List[Section]) -> Section:
        summaries = [section.body for section in element_sections]
        metadata = {"element_count": len(element_sections), "tree": structure.tree_repr}
        combined = self._orchestrator.synthesize(summaries=summaries, metadata=metadata)
        return Section(title="Project Overview", body=combined, metadata=metadata)

    def _write_cross_reference(self, existing_docs: Dict[Path, str], sections: List[Section]) -> Path:
        cross_ref = {}
        for section in sections:
            supplemental = section.metadata.get("supplemental_docs", [])
            if supplemental:
                cross_ref[section.title] = supplemental
        if existing_docs:
            cross_ref["_doc_sources"] = [str(path) for path in existing_docs.keys()]
        output_path = self._config.output_path.parent / "cross-reference.json"
        output_path.write_text(json.dumps(cross_ref, indent=2), encoding="utf-8")
        return output_path

    def _write_output(self, sections: List[Section]) -> None:
        lines: list[str] = ["# GenDoc Prototype Output", ""]
        for section in sections:
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.body)
            lines.append("")
        self._config.output_path.write_text("\n".join(lines), encoding="utf-8")