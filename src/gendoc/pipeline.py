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
    identifiers: List[str] = field(default_factory=list)


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
            identifiers: list[str] = []
            for section in sections:
                raw_notes_parts.append(f"### {section.title}\n{section.body.strip()}")
                identifier = section.metadata.get("identifier")
                if identifier:
                    identifiers.append(str(identifier))
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
                    identifiers=sorted(set(identifiers)),
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

    def _insert_run_section(self, document_text: str, run_section: str) -> str:
        cleaned_section = run_section.strip()
        if not cleaned_section:
            return document_text
        if not cleaned_section.lower().startswith("## to run"):
            cleaned_section = f"## To Run\n\n{cleaned_section}"

        to_run_pattern = re.compile(r"^## To Run.*?(?=^\s*## |\Z)", re.DOTALL | re.MULTILINE)
        document_body = to_run_pattern.sub("", document_text).strip("\n")

        insertion_candidates = [
            re.compile(r"(^## Functional Flow.*?)(?=^\s*## |\Z)", re.DOTALL | re.MULTILINE),
            re.compile(r"(^## At a Glance.*?)(?=^\s*## |\Z)", re.DOTALL | re.MULTILINE),
        ]

        insert_pos: int | None = None
        for pattern in insertion_candidates:
            match = pattern.search(document_body)
            if match:
                insert_pos = match.end()
                break

        if insert_pos is None:
            insert_pos = document_body.find("\n## ")
            if insert_pos == -1:
                insert_pos = len(document_body)

        before = document_body[:insert_pos].rstrip("\n")
        after = document_body[insert_pos:].lstrip("\n")

        segments = []
        if before:
            segments.append(before)
        segments.append(cleaned_section)
        if after:
            segments.append(after)

        return "\n\n".join(segments).strip("\n") + "\n"

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
        return entries

    def _augment_component_breakdown(
        self,
        *,
        section_text: str,
        component_docs: List[ComponentDocument],
    ) -> str:
        lines = section_text.splitlines()
        if not lines:
            return section_text
        header = lines[0]
        remainder = lines[1:]
        result_lines: list[str] = [header]
        used_docs: set[int] = set()
        idx = 0
        while idx < len(remainder):
            line = remainder[idx]
            stripped = line.strip()
            if stripped.startswith("### "):
                heading_text = stripped[4:].strip()
                result_lines.append(line)
                idx += 1
                block_lines: list[str] = []
                while idx < len(remainder) and not remainder[idx].strip().startswith("### "):
                    block_lines.append(remainder[idx])
                    idx += 1
                matched_indices = self._match_component_doc_indices(heading_text, component_docs)
                matched_docs = [component_docs[i] for i in matched_indices]
                used_docs.update(matched_indices)
                updated_block = self._attach_component_links(block_lines, matched_docs)
                result_lines.extend(updated_block)
                if updated_block and updated_block[-1].strip():
                    result_lines.append("")
                continue
            result_lines.append(line)
            idx += 1

        leftover_docs = [component_docs[i] for i in range(len(component_docs)) if i not in used_docs]
        if leftover_docs:
            if result_lines and result_lines[-1].strip():
                result_lines.append("")
            result_lines.extend(self._attach_component_links([], leftover_docs, allow_inline=False))

        if result_lines and result_lines[-1].strip():
            result_lines.append("")
        return "\n".join(line for line in result_lines)

    def _merge_component_deep_links(
        self,
        document_text: str,
        component_docs: List[ComponentDocument],
    ) -> str:
        if not component_docs:
            return document_text
        pattern = re.compile(r"(^## Component Breakdown.*?)(?=^\s*## |\Z)", re.DOTALL | re.MULTILINE)
        match = pattern.search(document_text)
        if not match:
            references = self._attach_component_links([], component_docs, allow_inline=False)
            base_text = document_text.rstrip()
            reference_block = "\n".join(references)
            return f"{base_text}\n\n{reference_block}\n"
        section_text = match.group(0)
        augmented = self._augment_component_breakdown(section_text=section_text, component_docs=component_docs)
        start, end = match.span(0)
        return f"{document_text[:start]}{augmented}{document_text[end:]}"

        if result_lines and result_lines[-1].strip():
            result_lines.append("")
        return "\n".join(line for line in result_lines if line is not None)

    def _match_component_doc_indices(self, heading: str, component_docs: List[ComponentDocument]) -> List[int]:
        heading_key = self._normalize_component_key(heading)
        heading_tokens = self._tokenize_key(heading)
        matches: list[tuple[int, int]] = []
        for idx, doc in enumerate(component_docs):
            doc_keys = self._component_doc_keys(doc)
            doc_tokens = self._component_doc_tokens(doc)
            score = 0
            if heading_key and heading_key in doc_keys:
                score = 3
            elif heading_key and any(heading_key in key or key in heading_key for key in doc_keys if key):
                score = 2
            elif heading_tokens and doc_tokens and heading_tokens & doc_tokens:
                score = 1
            if score:
                matches.append((score, idx))
        if not matches:
            return []
        max_score = max(score for score, _ in matches)
        return [idx for score, idx in matches if score == max_score]

    def _normalize_component_key(self, value: str) -> str:
        if not value:
            return ""
        cleaned = value.strip().replace("\\", "/")
        if "/" in cleaned:
            cleaned = cleaned.split("/")[-1]
        cleaned = re.sub(r"\.(py|pyi|md|rst|txt)$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^A-Za-z0-9]+", "", cleaned.lower())
        return cleaned

    def _component_doc_keys(self, doc: ComponentDocument) -> set[str]:
        keys: set[str] = set()
        keys.add(self._normalize_component_key(doc.name))
        display = self._component_display_name(doc)
        keys.add(self._normalize_component_key(display))
        base = display.rsplit(".", 1)[0]
        keys.add(self._normalize_component_key(base))
        for identifier in doc.identifiers:
            keys.add(self._normalize_component_key(identifier))
            parts = re.split(r"[.:]+", identifier)
            if parts:
                keys.add(self._normalize_component_key(parts[-1]))
        return {key for key in keys if key}

    def _component_doc_tokens(self, doc: ComponentDocument) -> set[str]:
        tokens = set()
        tokens.update(self._tokenize_key(doc.name))
        display = self._component_display_name(doc)
        tokens.update(self._tokenize_key(display))
        base = display.rsplit(".", 1)[0]
        tokens.update(self._tokenize_key(base))
        for identifier in doc.identifiers:
            tokens.update(self._tokenize_key(identifier))
            parts = re.split(r"[.:]+", identifier)
            if parts:
                tokens.update(self._tokenize_key(parts[-1]))
        if doc.summary:
            tokens.update(self._tokenize_key(doc.summary))
        return {token for token in tokens if token}

    def _tokenize_key(self, value: str) -> set[str]:
        if not value:
            return set()
        cleaned = value.replace("\\", "/").replace(".", " ")
        parts = re.split(r"[^A-Za-z0-9]+", cleaned)
        tokens: list[str] = []
        for part in parts:
            if not part:
                continue
            camel_tokens = re.findall(r"[A-Z]?[a-z]+|[0-9]+", part)
            if camel_tokens:
                tokens.extend(camel_tokens)
            else:
                tokens.append(part)
        return {token.lower() for token in tokens if token}

    def _component_display_name(self, doc: ComponentDocument) -> str:
        base = doc.name.replace("\\", "/").split("/")[-1]
        return base or doc.name

    def _format_doc_link(self, doc: ComponentDocument) -> str:
        display = self._component_display_name(doc)
        return f"[{display}]({doc.relative_path})"

    def _attach_component_links(
        self,
        block_lines: List[str],
        docs: List[ComponentDocument],
        *,
        allow_inline: bool = True,
    ) -> List[str]:
        if not docs:
            return block_lines
        lines = list(block_lines)
        primary_doc = docs[0]
        if allow_inline:
            attach_index = next((idx for idx, value in enumerate(lines) if value.strip()), None)
            reference_text = self._format_doc_link(primary_doc)
            if attach_index is not None:
                lines[attach_index] = self._append_inline_reference(lines[attach_index], reference_text)
            else:
                lines.append(f"- Reference: {reference_text}")
        else:
            lines.append(f"- Reference: {self._format_doc_link(primary_doc)} — {self._clean_summary(primary_doc.summary)}")
        for doc in docs[1:]:
            lines.append(f"- Also see {self._format_doc_link(doc)} — {self._clean_summary(doc.summary)}")
        return lines

    def _append_inline_reference(self, line: str, reference_text: str) -> str:
        stripped = line.rstrip()
        suffix = f" (See {reference_text} for details.)"
        if stripped.endswith(('.', '!', '?')):
            return stripped + suffix
        return stripped + suffix

    def _clean_summary(self, summary: str | None) -> str:
        if not summary:
            return "Further details available."
        cleaned = summary.strip().lstrip("- ").strip()
        return cleaned or "Further details available."

