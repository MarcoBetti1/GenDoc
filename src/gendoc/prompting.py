from __future__ import annotations

import ast
import importlib
import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Iterable, List, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PromptRecord:
    role: str
    prompt: str
    response: str
    timestamp: datetime
    metadata: Dict[str, object]

    def to_json(self) -> str:
        payload = {
            "role": self.role,
            "prompt": self.prompt,
            "response": self.response,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
        return json.dumps(payload, ensure_ascii=False)


class PromptLedger:
    """Persists prompts/responses for auditability."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: PromptRecord) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_json() + "\n")


class LLMClient(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str, metadata: Optional[Dict[str, object]] = None) -> str:
        ...


class MockLLM:
    """Heuristic LLM used for offline development."""

    def complete(self, *, system_prompt: str, user_prompt: str, metadata: Optional[Dict[str, object]] = None) -> str:  # noqa: D401
        _ = system_prompt
        metadata = metadata or {}

        identifier = metadata.get("identifier") or self._extract_identifier(user_prompt) or "section"

        if "Given the following code snippet" in user_prompt:
            meta_obj = self._extract_json_block(user_prompt, "Metadata:")
            code = self._extract_code_block(user_prompt)
            source_identifier = (meta_obj or {}).get("identifier") or identifier
            return self._summarize_code_block(code, source_identifier, meta_obj)

        if "Rewrite this section" in user_prompt:
            goal = self._extract_goal(user_prompt)
            section_meta = self._extract_json_block(user_prompt, "Section metadata:") or {}
            section_identifier = section_meta.get("identifier") or identifier
            section_text = self._extract_section_content(user_prompt)
            return self._refine_section(section_text, goal, section_identifier)

        if "Section summaries:" in user_prompt:
            return self._derive_project_goal_text(user_prompt)

        if "Review the provided code" in user_prompt:
            return f"Approve: summary for {identifier} appears thorough."

        if "Combine the following component summaries" in user_prompt:
            return self._combine_summaries(user_prompt)

        if "Produce a standalone Markdown document" in user_prompt and "Section digests" in user_prompt:
            return self._compose_document_overview(user_prompt)

        if "Overly descriptive notes" in user_prompt and "Component:" in user_prompt:
            return self._trim_component_notes(user_prompt)

        if "Condensed notes" in user_prompt and "Component:" in user_prompt:
            return self._polish_component_notes(user_prompt)

        if "Return a JSON object with keys \"prerequisites\", \"windows\", \"mac\"" in user_prompt:
            return self._mock_run_steps_context()

        if "Raw run instructions (JSON):" in user_prompt and "Convert the JSON" in user_prompt:
            return self._mock_run_markdown(user_prompt)

        if "Component documents (JSON):" in user_prompt and "Base document:" in user_prompt:
            return self._integrate_component_section(user_prompt)

        lines = [line.strip() for line in user_prompt.splitlines() if line.strip()]
        if not lines:
            return "No content provided."
        code_preview = ""
        if "Code:" in user_prompt:
            code_preview = user_prompt.split("Code:", 1)[-1].strip().splitlines()[0].strip()
        description = f"{identifier} behavior" if identifier else "Element behavior"
        if code_preview:
            return f"{description}: {code_preview[:200]}"
        return f"{description}: summarized"

    # -- helpers -----------------------------------------------------------------

    def _extract_identifier(self, prompt: str) -> Optional[str]:
        meta = self._extract_json_block(prompt, "Metadata:")
        if meta and meta.get("identifier"):
            return str(meta["identifier"])
        section_meta = self._extract_json_block(prompt, "Section metadata:")
        if section_meta and section_meta.get("identifier"):
            return str(section_meta["identifier"])
        return None

    def _extract_json_block(self, prompt: str, label: str) -> Optional[Dict[str, Any]]:
        if label not in prompt:
            return None
        after = prompt.split(label, 1)[1]
        start = after.find("{")
        if start == -1:
            return None
        depth = 0
        for index, char in enumerate(after[start:]):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = start + index + 1
                    try:
                        return json.loads(after[start:end])
                    except json.JSONDecodeError:
                        return None
        return None

    def _extract_code_block(self, prompt: str) -> str:
        if "Code:" not in prompt:
            return ""
        return dedent(prompt.split("Code:", 1)[-1]).strip()

    def _summarize_code_block(
        self,
        code: str,
        identifier: str,
        metadata: Optional[Dict[str, Any]],
    ) -> str:
        if not code:
            return f"{identifier} behavior: source code unavailable."

        parsed = self._safe_parse(code)
        if not parsed or not parsed.body:
            return self._fallback_summary(identifier, code)

        node = parsed.body[0]
        if isinstance(node, ast.ClassDef):
            return self._summarize_class(node, identifier, metadata, code)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self._summarize_function(node, identifier, metadata, code)
        return self._fallback_summary(identifier, code)

    def _safe_parse(self, code: str) -> Optional[ast.Module]:
        try:
            return ast.parse(dedent(code))
        except SyntaxError:
            return None

    def _fallback_summary(self, identifier: str, code: str) -> str:
        first_line = code.strip().splitlines()[0] if code.strip() else ""
        snippet = first_line[:160]
        return f"{identifier} behavior: {snippet}".strip()

    def _summarize_class(
        self,
        node: ast.ClassDef,
        identifier: str,
        metadata: Optional[Dict[str, Any]],
        code: str,
    ) -> str:
        class_name = node.name
        fields: set[str] = set()
        methods: list[str] = []
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                methods.append(stmt.name)
                for inner in ast.walk(stmt):
                    if isinstance(inner, ast.Assign):
                        for target in inner.targets:
                            attr = self._attr_name(target)
                            if attr:
                                fields.add(attr)
                    elif isinstance(inner, ast.AnnAssign):
                        attr = self._attr_name(inner.target)
                        if attr:
                            fields.add(attr)
        is_dataclass = any(
            isinstance(dec, ast.Name) and dec.id == "dataclass"
            or isinstance(dec, ast.Attribute) and dec.attr == "dataclass"
            for dec in node.decorator_list
        )
        parts: list[str] = [f"{identifier} defines the {class_name} class."]
        if is_dataclass:
            parts.append("It is a dataclass for structured data.")
        if fields:
            parts.append(f"Instances track {self._format_list(sorted(fields))}.")
        if methods:
            parts.append(f"Key methods include {self._format_list(methods, limit=3)}.")
        behavior = self._infer_behavior_from_source(code)
        if behavior:
            parts.append(behavior)
        return " ".join(parts)

    def _attr_name(self, target: ast.AST) -> Optional[str]:
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
            return target.attr
        return None

    def _summarize_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        identifier: str,
        metadata: Optional[Dict[str, Any]],
        code: str,
    ) -> str:
        params = [arg.arg for arg in node.args.args]
        is_method = bool(params and params[0] == "self")
        public_params = [p for p in params if p != "self"]
        signature = f"{node.name}({', '.join(params)})"
        parts: list[str] = []
        role = "method" if is_method else "function"
        parts.append(f"{identifier} implements the {role} {signature}.")
        if public_params:
            parts.append(f"It accepts {self._format_list(public_params)}.")
        return_annotation = self._get_return_annotation(node)
        if return_annotation:
            parts.append(f"It returns {return_annotation}.")
        behavior = self._infer_behavior_from_source(code)
        if behavior:
            parts.append(behavior)
        return " ".join(parts)

    def _get_return_annotation(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> Optional[str]:
        if node.returns is None:
            return None
        try:
            return ast.unparse(node.returns)
        except Exception:  # pragma: no cover - ast.unparse fallback
            return None

    def _infer_behavior_from_source(self, code: str) -> Optional[str]:
        text = code.lower()
        observations: list[str] = []
        if "return [" in text and "if task.completed" in text:
            observations.append("It filters tasks flagged as completed.")
        if "if not task.completed" in text:
            observations.append("It collects tasks that are still pending.")
        if "len(self._tasks)" in text and "if total == 0" in text:
            observations.append("It guards against division by zero when the list is empty.")
        if "len(self.completed())" in text and " / total" in text:
            observations.append("It computes the fraction of tasks that are complete.")
        if "self._tasks.append" in text:
            observations.append("It appends a new task to the internal list and returns it.")
        if "mark_done" in text and "bootstrap" in text:
            observations.append("It seeds a demo list and marks some tasks complete for examples.")
        if "mark_done" in text and "self.completed = true" in text:
            observations.append("It marks a task as completed by toggling its flag.")
        observations = list(dict.fromkeys(observations))
        if not observations and "return" in text:
            observations.append("It returns the computed result based on the current task state.")
        if not observations:
            return None
        return " ".join(observations)

    def _format_list(self, items: Iterable[str], limit: int | None = None) -> str:
        unique = list(dict.fromkeys(items))
        if limit and len(unique) > limit:
            unique = unique[:limit]
        if not unique:
            return ""
        if len(unique) == 1:
            return unique[0]
        return ", ".join(unique[:-1]) + f" and {unique[-1]}"

    def _extract_goal(self, prompt: str) -> str:
        if "Project goal:" not in prompt:
            return ""
        after = prompt.split("Project goal:", 1)[1]
        if "Section metadata" in after:
            goal_text = after.split("Section metadata", 1)[0]
        else:
            goal_text = after
        return goal_text.strip()

    def _extract_section_content(self, prompt: str) -> str:
        if "Section content:" not in prompt:
            return ""
        after = prompt.split("Section content:", 1)[1]
        return after.split("Rewrite this section", 1)[0].strip()

    def _refine_section(self, section_text: str, goal: str, identifier: str) -> str:
        if not section_text:
            return "OMIT"
        sentences = re.split(r"(?<=[.!?])\s+", section_text.strip())
        if not sentences:
            return section_text.strip()
        keywords = {word.lower() for word in re.findall(r"[A-Za-z0-9_]+", goal) if len(word) > 3}
        keywords.update(word.lower() for word in re.findall(r"[A-Za-z0-9_]+", identifier))
        filtered = [sent for sent in sentences if any(word in sent.lower() for word in keywords)]
        if not filtered:
            filtered = sentences[:2]
        refined = " ".join(filtered).strip()
        if refined and not refined.endswith(('.', '!', '?')):
            refined += "."
        return refined or "OMIT"

    def _derive_project_goal_text(self, prompt: str) -> str:
        summaries_block = prompt.split("Section summaries:", 1)[-1].strip()
        identifiers = []
        for match in re.finditer(r"([\w\.]+::[\w_]+)", summaries_block):
            identifiers.append(match.group(1))
        modules = sorted({ident.split("::")[0] for ident in identifiers if "::" in ident})
        entities = sorted({ident.split("::")[-1] for ident in identifiers if "::" in ident})
        module_phrase = self._format_list(modules) if modules else "the codebase"
        entity_phrase = self._format_list(entities, limit=4) if entities else "its core components"
        return (
            f"Project goal: Provide clear documentation for {entity_phrase} in {module_phrase}, highlighting how the pieces collaborate."
        )

    def _combine_summaries(self, prompt: str) -> str:
        meta = self._extract_json_block(prompt, "Metadata:") or {}
        element_count = meta.get("element_count")
        project_goal = meta.get("project_goal") or "the documented goal"
        tree = meta.get("tree", "")
        modules = [line.strip("- ") for line in tree.splitlines() if line.strip()]
        module_phrase = self._format_list(modules) if modules else "the analysed modules"
        if isinstance(project_goal, str) and project_goal.lower().startswith("project goal:"):
            goal_text = project_goal.split(":", 1)[-1].strip()
        else:
            goal_text = str(project_goal)
        goal_text = goal_text.rstrip(".")
        count_text = f"{element_count}" if element_count is not None else "several"
        return (
            f"This overview weaves together {count_text} focused summaries across {module_phrase} to support the goal of {goal_text}."
        )

    def _compose_document_overview(self, prompt: str) -> str:
        goal = self._extract_between(prompt, "Project goal:", "Repository name:") or "Project goal unavailable."
        repo = self._extract_between(prompt, "Repository name:", "Project tree:") or "Repository"
        tree = self._extract_between(prompt, "Project tree:", "Section digests (JSON):") or ""
        sections_json = self._extract_between(prompt, "Section digests (JSON):", "Produce a standalone")
        sections: list[dict[str, Any]] = []
        if sections_json:
            try:
                sections = json.loads(sections_json)
            except json.JSONDecodeError:
                sections = []

        module_counts: dict[str, int] = {}
        highlight_summaries: list[str] = []
        for entry in sections:
            path = entry.get("path") or entry.get("title", "")
            module = str(path).split("/")[0] if path else "module"
            module = module or "module"
            module_counts[module] = module_counts.get(module, 0) + 1
            summary = entry.get("summary")
            if isinstance(summary, str) and summary.strip():
                highlight_summaries.append(summary.strip())

        modules_sorted = sorted(module_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        module_names = [name for name, _ in modules_sorted]
        module_phrase = self._format_list(module_names, limit=4) if module_names else "the key modules"
        section_total = sum(module_counts.values()) or len(sections) or 0

        at_a_glance = [
            f"Mission: {goal.strip().rstrip('.')}.",
            f"Coverage spans {section_total or 'several'} documented sections across {module_phrase}.",
            "LLM integration: content composed via the document prompt using structured section summaries.",
            "Differentiators: persistent prompt ledger and cross-reference artefacts for traceability.",
        ]
        if tree.strip():
            at_a_glance.append("Architecture footprint:\n" + "  " + tree.strip().splitlines()[0])
        while len(at_a_glance) < 4:
            at_a_glance.append("Additional insight captured during analysis.")
        at_a_glance = at_a_glance[:6]

        functional_flow = [
            "Inspect repository structure and gather code elements for analysis.",
            "Generate and refine per-element summaries aligned with the project goal.",
            "Compose the final narrative and export markdown plus traceable artefacts.",
        ]

        component_sections: list[str] = []
        for name, count in modules_sorted[:5]:
            component_sections.append(
                f"### {name}\n- Summarizes {count} element{'s' if count != 1 else ''} from this module."
            )
        if not component_sections:
            component_sections.append("### Components\n- Summaries highlight core behaviors across the repository.")

        outputs = [
            "Markdown documentation at the requested output path.",
            "JSON cross-reference linking sections to supplemental sources.",
            "Prompt ledger (`prompt-ledger.jsonl`) capturing LLM interactions.",
        ]

        issues = [
            "No issues were identified during mock generation; review real LLM output for accuracy.",
        ]

        lines = [
            f"# {repo} - Mock Overview",
            "",
            "## At a Glance",
        ]
        lines.extend(f"- {item}" for item in at_a_glance)

        lines.append("")
        lines.append("## Functional Flow")
        for idx, step in enumerate(functional_flow, start=1):
            lines.append(f"{idx}. {step}")

        lines.append("")
        lines.append("## Component Breakdown")
        lines.extend(component_sections)

        lines.append("")
        lines.append("## Outputs & Observability")
        lines.extend(f"- {item}" for item in outputs)

        lines.append("")
        lines.append("## Known Issues & Bugs")
        lines.extend(f"- {item}" for item in issues)

        return "\n".join(lines).strip() + "\n"

    def _extract_between(self, prompt: str, start_label: str, end_label: str) -> Optional[str]:
        if start_label not in prompt or end_label not in prompt:
            return None
        after = prompt.split(start_label, 1)[1]
        before = after.split(end_label, 1)[0]
        return before.strip() or None

    def _trim_component_notes(self, prompt: str) -> str:
        content = self._extract_between(prompt, "Overly descriptive notes:", "\n\n")
        if not content:
            content = prompt.split("Overly descriptive notes:", 1)[-1].strip()
        entries: list[tuple[str, str]] = []
        for block in content.split("### "):
            if not block.strip():
                continue
            title, _, remainder = block.partition("\n")
            title = title.strip()
            summary_line = ""
            for raw_line in remainder.splitlines():
                cleaned = raw_line.strip()
                if cleaned:
                    summary_line = cleaned
                    break
            entries.append((title, summary_line))
        if not entries:
            return "### Essential Responsibilities\n- Essential responsibilities will be extracted from the source summaries."
        essential_lines = [
            f"- Focus: {title}" + (f" — {summary}" if summary else "")
            for title, summary in entries[:5]
        ]
        interaction_lines = [
            f"- Interaction cue: {summary}"
            for _, summary in entries[5:9]
            if summary
        ]
        sections: list[str] = ["### Essential Responsibilities", *essential_lines]
        if interaction_lines:
            sections.append("### Key Interactions")
            sections.extend(interaction_lines)
        return "\n".join(sections).strip()

    def _polish_component_notes(self, prompt: str) -> str:
        component = self._extract_between(prompt, "Component:", "\n") or "Component"
        notes = self._extract_between(prompt, "Condensed notes:", "\n\n")
        if not notes:
            notes = prompt.split("Condensed notes:", 1)[-1].strip()
        overview_lines = [line.strip("- ") for line in notes.splitlines() if line.strip()]
        intro = next((line for line in overview_lines if "Focus:" in line), overview_lines[0] if overview_lines else "Purpose pending analysis.")
        responsibilities = [line for line in overview_lines if line.startswith("Focus:")]
        interactions = [line for line in overview_lines if "Interaction cue" in line]
        snippet_matches = re.findall(r"```python\n(.*?)\n```", prompt, flags=re.DOTALL)
        snippet_body = snippet_matches[0].strip() if snippet_matches else ""
        lines = [
            f"# {component.strip()} Component Guide",
            "",
            "## Overview",
            intro,
            "",
            "## Key Responsibilities",
        ]
        if responsibilities:
            lines.extend(f"- {item}" for item in responsibilities)
        else:
            lines.append("- Responsibilities captured during analysis.")
        lines.extend(["", "## Collaboration Points"])
        if interactions:
            lines.extend(f"- {item}" for item in interactions[:5])
        else:
            lines.append("- Interacts with surrounding modules as orchestrated by the game runner.")
        lines.extend(["", "## Implementation Notes", "- Refer to source summaries for concrete behaviors."])
        if snippet_body:
            lines.extend(["", "```python", snippet_body, "```"])
        return "\n".join(lines).strip() + "\n"

    def _mock_run_steps_context(self) -> str:
        payload = {
            "prerequisites": [
                "Python 3.11 or newer",
                "Set OPENAI_API_KEY when invoking the OpenAI provider",
            ],
            "windows": [
                "python -m venv .venv",
                r".venv\\Scripts\\Activate.ps1",
                "pip install -e .",
                "gendoc run --repo samples/demo_app --out docs/generated/demo-output.md --llm-provider mock",
            ],
            "mac": [
                "python3 -m venv .venv",
                "source .venv/bin/activate",
                "pip install -e .",
                "gendoc run --repo samples/demo_app --out docs/generated/demo-output.md --llm-provider mock",
            ],
            "notes": "Switch to --llm-provider openai after exporting OPENAI_API_KEY for live runs.",
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _mock_run_markdown(self, prompt: str) -> str:
        data = self._extract_json_block(prompt, "Raw run instructions (JSON):") or {}
        if not data:
            try:
                data = json.loads(self._mock_run_steps_context())
            except json.JSONDecodeError:
                data = {}
        prerequisites = data.get("prerequisites") or []
        windows_steps = data.get("windows") or []
        mac_steps = data.get("mac") or []
        notes = data.get("notes")

        lines = ["## To Run", ""]
        if prerequisites:
            lines.append("**Prerequisites**")
            lines.extend(f"- {item}" for item in prerequisites)
            lines.append("")
        lines.append("### Windows")
        lines.extend(self._format_numbered_steps(windows_steps))
        lines.append("")
        lines.append("### macOS")
        lines.extend(self._format_numbered_steps(mac_steps))
        if notes:
            lines.extend(["", "**Notes**", f"- {notes}"])
        return "\n".join(lines).strip() + "\n"

    def _format_numbered_steps(self, steps: List[str]) -> List[str]:
        if not steps:
            return ["1. Refer to the README for the latest setup guidance."]
        return [f"{idx}. {step}" for idx, step in enumerate(steps, start=1)]

    def _integrate_component_section(self, prompt: str) -> str:
        base = self._extract_between(prompt, "Base document:", "Component documents (JSON):") or ""
        components_json = self._extract_between(prompt, "Component documents (JSON):", "Update the base document")
        if not components_json:
            components_json = self._extract_between(prompt, "Component documents (JSON):", "Append a new concluding section")
        if not components_json:
            components_json = prompt.split("Component documents (JSON):", 1)[-1].split("Append a new concluding section", 1)[0].strip()
        try:
            entries = json.loads(components_json)
        except json.JSONDecodeError:
            entries = []
        lines = [base.strip(), "", "## Component Deep Dives"]
        if not entries:
            lines.append("- Detailed component documents will be published soon.")
        else:
            for entry in entries:
                name = entry.get("name", "Component")
                path = entry.get("relative_path", "")
                summary = entry.get("summary", "Further details available.")
                link_text = f"[{name}]({path})"
                lines.append(f"- {link_text}: {summary}")
        return "\n".join(lines).strip() + "\n"


class OpenAIClient:
    """Thin wrapper around the OpenAI Chat Completions API."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        try:
            openai = importlib.import_module("openai")
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise RuntimeError("openai package is not installed") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        self._client: Any = openai.OpenAI(api_key=api_key)
        self._model = model

    def complete(self, *, system_prompt: str, user_prompt: str, metadata: Optional[Dict[str, object]] = None) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content or ""


@dataclass(slots=True)
class PromptTemplates:
    system_analyst: str
    prompt_analyst: str
    system_reviewer: str
    prompt_reviewer: str
    system_synthesizer: str
    prompt_synthesizer: str
    system_goal: str
    prompt_goal: str
    system_filter: str
    prompt_filter: str
    system_document: str
    prompt_document: str
    system_component_prune: str
    prompt_component_prune: str
    system_component_polish: str
    prompt_component_polish: str
    system_document_integrator: str
    prompt_document_integrator: str
    system_run_collector: str
    prompt_run_collector: str
    system_run_refiner: str
    prompt_run_refiner: str


class PromptOrchestrator:
    """Coordinates the prompt workflow across agents."""

    def __init__(self, client: LLMClient, ledger: PromptLedger, templates: PromptTemplates) -> None:
        self._client = client
        self._ledger = ledger
        self._templates = templates

    def summarize_element(
        self,
        *,
        element_source: str,
        metadata: Dict[str, object],
        supplemental_text: Optional[str] = None,
    ) -> str:
        supplemental_block = (
            f"\n\nExisting documentation excerpts:\n{supplemental_text}"
            if supplemental_text
            else ""
        )
        prompt = self._templates.prompt_analyst.format(
            source=element_source,
            metadata=json.dumps(metadata, ensure_ascii=False),
            supplemental=supplemental_block,
        )
        response = self._client.complete(
            system_prompt=self._templates.system_analyst,
            user_prompt=prompt,
            metadata={
                "stage": "element",
                "identifier": metadata.get("identifier"),
            },
        )
        self._ledger.log(
            PromptRecord(
                role="analyst",
                prompt=prompt,
                response=response,
                timestamp=datetime.utcnow(),
                metadata={
                    "stage": "element",
                    "identifier": metadata.get("identifier"),
                    "supplemental_used": bool(supplemental_text),
                },
            )
        )
        return response

    def review_summary(
        self,
        *,
        element_source: str,
        summary: str,
        metadata: Dict[str, object],
        supplemental_text: Optional[str] = None,
    ) -> str:
        supplemental_block = (
            f"\n\nExisting documentation excerpts:\n{supplemental_text}"
            if supplemental_text
            else ""
        )
        prompt = self._templates.prompt_reviewer.format(
            source=element_source,
            summary=summary,
            metadata=json.dumps(metadata, ensure_ascii=False),
            supplemental=supplemental_block,
        )
        response = self._client.complete(
            system_prompt=self._templates.system_reviewer,
            user_prompt=prompt,
            metadata={
                "stage": "review",
                "identifier": metadata.get("identifier"),
            },
        )
        self._ledger.log(
            PromptRecord(
                role="reviewer",
                prompt=prompt,
                response=response,
                timestamp=datetime.utcnow(),
                metadata={
                    "stage": "review",
                    "identifier": metadata.get("identifier"),
                    "supplemental_used": bool(supplemental_text),
                },
            )
        )
        if response.lower().startswith("approve"):
            return summary
        return response

    def synthesize(self, *, summaries: Iterable[str], metadata: Dict[str, object]) -> str:
        joined = "\n".join(summaries)
        prompt = self._templates.prompt_synthesizer.format(summaries=joined, metadata=json.dumps(metadata, ensure_ascii=False))
        response = self._client.complete(system_prompt=self._templates.system_synthesizer, user_prompt=prompt, metadata={"stage": "synthesis"})
        self._ledger.log(
            PromptRecord(
                role="synthesizer",
                prompt=prompt,
                response=response,
                timestamp=datetime.utcnow(),
                metadata={"stage": "synthesis"},
            )
        )
        return response

    def derive_project_goal(self, *, section_summaries: Iterable[str]) -> str:
        joined = "\n".join(section_summaries)
        prompt = self._templates.prompt_goal.format(summaries=joined)
        response = self._client.complete(
            system_prompt=self._templates.system_goal,
            user_prompt=prompt,
            metadata={"stage": "goal"},
        )
        self._ledger.log(
            PromptRecord(
                role="goal",
                prompt=prompt,
                response=response,
                timestamp=datetime.utcnow(),
                metadata={"stage": "goal"},
            )
        )
        return response

    def refine_section(self, *, section_body: str, goal: str, metadata: Dict[str, object]) -> str:
        prompt = self._templates.prompt_filter.format(
            goal=goal,
            section=section_body,
            metadata=json.dumps(metadata, ensure_ascii=False),
        )
        response = self._client.complete(
            system_prompt=self._templates.system_filter,
            user_prompt=prompt,
            metadata={"stage": "refine", "identifier": metadata.get("identifier")},
        )
        self._ledger.log(
            PromptRecord(
                role="refiner",
                prompt=prompt,
                response=response,
                timestamp=datetime.utcnow(),
                metadata={"stage": "refine", "identifier": metadata.get("identifier")},
            )
        )
        return response

    def compose_document(
        self,
        *,
        goal: str,
        project_tree: str,
        repo_name: str,
        section_payload: List[Dict[str, object]],
    ) -> str:
        prompt = self._templates.prompt_document.format(
            goal=goal.strip(),
            repo=repo_name,
            tree=project_tree.strip() or "(structure unavailable)",
            sections=json.dumps(section_payload, ensure_ascii=False, indent=2),
        )
        response = self._client.complete(
            system_prompt=self._templates.system_document,
            user_prompt=prompt,
            metadata={"stage": "document"},
        )
        self._ledger.log(
            PromptRecord(
                role="document",
                prompt=prompt,
                response=response,
                timestamp=datetime.utcnow(),
                metadata={"stage": "document"},
            )
        )
        return response

    def collect_run_steps(
        self,
        *,
        goal: str,
        context_entries: List[Dict[str, str]],
    ) -> str:
        prompt = self._templates.prompt_run_collector.format(
            goal=goal,
            context=json.dumps(context_entries, ensure_ascii=False, indent=2),
        )
        response = self._client.complete(
            system_prompt=self._templates.system_run_collector,
            user_prompt=prompt,
            metadata={"stage": "run_collect"},
        )
        self._ledger.log(
            PromptRecord(
                role="run_collect",
                prompt=prompt,
                response=response,
                timestamp=datetime.utcnow(),
                metadata={"stage": "run_collect"},
            )
        )
        return response

    def refine_run_steps(
        self,
        *,
        goal: str,
        draft: str,
    ) -> str:
        prompt = self._templates.prompt_run_refiner.format(
            goal=goal,
            draft=draft,
        )
        response = self._client.complete(
            system_prompt=self._templates.system_run_refiner,
            user_prompt=prompt,
            metadata={"stage": "run_refine"},
        )
        self._ledger.log(
            PromptRecord(
                role="run_refine",
                prompt=prompt,
                response=response,
                timestamp=datetime.utcnow(),
                metadata={"stage": "run_refine"},
            )
        )
        return response

    def reduce_component_notes(
        self,
        *,
        goal: str,
        component_name: str,
        raw_notes: str,
    ) -> str:
        prompt = self._templates.prompt_component_prune.format(
            goal=goal,
            component=component_name,
            raw_notes=raw_notes,
        )
        response = self._client.complete(
            system_prompt=self._templates.system_component_prune,
            user_prompt=prompt,
            metadata={"stage": "component_prune", "component": component_name},
        )
        self._ledger.log(
            PromptRecord(
                role="component_prune",
                prompt=prompt,
                response=response,
                timestamp=datetime.utcnow(),
                metadata={"stage": "component_prune", "component": component_name},
            )
        )
        return response

    def polish_component_document(
        self,
        *,
        goal: str,
        component_name: str,
        repo_name: str,
        condensed_notes: str,
        highlight_snippets: Optional[List[Dict[str, object]]] = None,
    ) -> str:
        snippet_block = ""
        if highlight_snippets:
            snippet_parts: list[str] = []
            for snippet in highlight_snippets:
                code = str(snippet.get("code", "")).strip()
                if not code:
                    continue
                title = str(snippet.get("title", "Key excerpt")).strip()
                path = str(snippet.get("path", ""))
                start_line = snippet.get("start_line")
                end_line = snippet.get("end_line")
                location_parts: list[str] = []
                if path:
                    location_parts.append(path)
                if isinstance(start_line, int) and isinstance(end_line, int):
                    location_parts.append(f"L{start_line}-{end_line}")
                location = " ".join(location_parts)
                header = f"- {title}" + (f" ({location})" if location else "")
                snippet_parts.append(f"{header}\n```python\n{code}\n```")
            if snippet_parts:
                snippet_block = "\n\nHighlighted code excerpts:\n" + "\n\n".join(snippet_parts)
        prompt = self._templates.prompt_component_polish.format(
            goal=goal,
            component=component_name,
            repo=repo_name,
            condensed=condensed_notes,
            snippet_block=snippet_block,
        )
        response = self._client.complete(
            system_prompt=self._templates.system_component_polish,
            user_prompt=prompt,
            metadata={"stage": "component_polish", "component": component_name},
        )
        self._ledger.log(
            PromptRecord(
                role="component_polish",
                prompt=prompt,
                response=response,
                timestamp=datetime.utcnow(),
                metadata={
                    "stage": "component_polish",
                    "component": component_name,
                    "highlight_count": len(highlight_snippets or []),
                },
            )
        )
        return response

    def integrate_component_links(
        self,
        *,
        base_document: str,
        component_payload: List[Dict[str, str]],
        goal: str,
    ) -> str:
        prompt = self._templates.prompt_document_integrator.format(
            goal=goal,
            document=base_document,
            components=json.dumps(component_payload, ensure_ascii=False, indent=2),
        )
        response = self._client.complete(
            system_prompt=self._templates.system_document_integrator,
            user_prompt=prompt,
            metadata={"stage": "document_integrator"},
        )
        self._ledger.log(
            PromptRecord(
                role="document_integrator",
                prompt=prompt,
                response=response,
                timestamp=datetime.utcnow(),
                metadata={"stage": "document_integrator"},
            )
        )
        return response


DEFAULT_TEMPLATES = PromptTemplates(
    system_analyst="You are an expert software analyst who explains code precisely.",
    prompt_analyst="""Given the following code snippet, metadata, and optional supplemental documentation, describe in detail what it does, its inputs, outputs, side effects, and error handling.\n\nMetadata: {metadata}{supplemental}\n\nCode:\n{source}\n""",
    system_reviewer="You are a meticulous reviewer ensuring the explanation covers all important behaviors.",
    prompt_reviewer="""Review the provided code, summary, and supplemental documentation (if present). If the summary is sufficient, respond with 'Approve' followed by optional minor notes. Otherwise, rewrite the summary to address the gaps.\n\nMetadata: {metadata}\nSummary:\n{summary}{supplemental}\n\nCode:\n{source}\n""",
    system_synthesizer="You are a technical writer composing cohesive documentation.",
    prompt_synthesizer="""Combine the following component summaries into a higher-level narrative that explains the purpose of the parent component.\n\nMetadata: {metadata}\nSummaries:\n{summaries}\n""",
    system_goal="You are a senior product strategist deriving precise project goals from technical documentation.",
    prompt_goal="""Using the section summaries below, craft a concise yet detailed project goal statement that captures the overall intent of the codebase.\n\nSection summaries:\n{summaries}\n""",
    system_filter="You are an editor who preserves only information relevant to the project goal.",
    prompt_filter="""Project goal:\n{goal}\n\nSection metadata: {metadata}\nSection content:\n{section}\n\nRewrite this section so it only contains information that directly supports the project goal. If the section is irrelevant, respond with 'OMIT'.""",
    system_document="You are a principal technical writer who produces clear, top-down documentation for engineering stakeholders.",
    prompt_document="""Project goal:\n{goal}\n\nRepository name: {repo}\nProject tree:\n{tree}\n\nSection digests (JSON):\n{sections}\n\nProduce a standalone Markdown document that explains the repository from high level to implementation. Requirements:\n- Title the document with an H1 that includes the repository name and a concise tagline.\n- Provide a section `## At a Glance` with 4-6 bullets covering mission, architecture, LLM integration, and distinguishing traits.\n- Provide `## Functional Flow` with a numbered sequence describing how the system operates end-to-end.\n- Provide `## Component Breakdown` with subsections for the major modules or services, summarizing responsibilities and collaboration.\n- Provide `## Outputs & Observability` as bullets describing generated artefacts, logs, or metrics.\n- Provide `## Known Issues & Bugs` capturing risks, limitations, or TODOs; if none are apparent, write a single bullet stating that no issues were identified.\n- Avoid referencing the JSON directly; convert insights into prose.\n- Keep the tone confident and instructive, and keep the document concise and scannable.\n""",
    system_component_prune="You are a ruthless technical editor who extracts only critical facts for component documentation. Always return plain Markdown without wrapping the entire response in code fences.",
    prompt_component_prune="""Project goal: {goal}\nComponent: {component}\n\nOverly descriptive notes:\n{raw_notes}\n\nCondense these notes to only the essential responsibilities, interactions, and edge considerations for the component. Return succinct Markdown bullets that can feed a polished document. Do not surround the response with ``` fences.""",
    system_component_polish="You are a senior technical writer crafting polished component guides for engineers. Deliver polished Markdown directly—never wrap the whole response in a fenced code block. Use language-specific fenced code snippets only when they clarify a critical implementation point, and keep such snippets short (<=10 lines).",
    prompt_component_polish="""Project goal: {goal}\nRepository: {repo}\nComponent: {component}\n\nCondensed notes:\n{condensed}{snippet_block}\n\nProduce a clean Markdown document for the component with sections: `## Overview`, `## Key Responsibilities`, `## Collaboration Points`, and `## Implementation Notes`. Keep it scannable and confident. Only include a ```python``` snippet when highlighting an important code path, and otherwise avoid code fences entirely.""",
    system_document_integrator="You are a documentation curator who weaves component deep dives into a primary reference.",
    prompt_document_integrator="""Project goal: {goal}\n\nBase document:\n{document}\n\nComponent documents (JSON):\n{components}\n\nUpdate the base document so that the `## Component Breakdown` section references the component documents directly. For each component payload entry, add or augment a bullet beneath the best matching subsection that links to the deep-dive file and incorporates the friendly summary. Preserve every other section exactly as written, including any run guides that may already exist, and do not create a standalone `## Component Deep Dives` section.""",
    system_run_collector="You are a release engineer who extracts concise cross-platform run instructions from technical materials.",
    prompt_run_collector="""Project goal: {goal}\n\nContext entries (JSON array):\n{context}\n\nIdentify prerequisites, environment setup tasks, and execution commands required to run the project on Windows (PowerShell) and macOS (zsh or bash). Return a JSON object with keys \"prerequisites\", \"windows\", \"mac\", and optional \"notes\". Each value must be an array of strings except \"notes\", which may be a string. Focus on actionable steps including virtual environment creation, dependency installation, environment variables, and CLI invocation. Respond with JSON only.""",
    system_run_refiner="You are a technical writer who formats platform-specific run instructions for engineers.",
    prompt_run_refiner="""Project goal: {goal}\n\nRaw run instructions (JSON):\n{draft}\n\nConvert the JSON into a Markdown section titled `## To Run`. Present prerequisites as a bullet list when provided, followed by `### Windows` and `### macOS` subsections that each use ordered steps. Close with a short notes block if a \"notes\" field exists. Mention required environment variables explicitly and keep the tone reassuring and direct.""",
)
