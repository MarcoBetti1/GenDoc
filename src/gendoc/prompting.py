from __future__ import annotations

import importlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Protocol

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
        lines = [line.strip() for line in user_prompt.splitlines() if line.strip()]
        if not lines:
            return "No content provided."
        preview = lines[: min(5, len(lines))]
        summary = " ".join(preview)
        return f"[mock-summary] {summary[:500]}"


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
        prompt = self._templates.prompt_analyst.format(
            source=element_source, metadata=json.dumps(metadata, ensure_ascii=False)
        )
        if supplemental_text:
            prompt += f"\n\nSupplemental documentation:\n{supplemental_text}\n"
        response = self._client.complete(system_prompt=self._templates.system_analyst, user_prompt=prompt, metadata={"stage": "element"})
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
        prompt = self._templates.prompt_reviewer.format(
            source=element_source, summary=summary, metadata=json.dumps(metadata, ensure_ascii=False)
        )
        if supplemental_text:
            prompt += f"\n\nSupplemental documentation:\n{supplemental_text}\n"
        response = self._client.complete(system_prompt=self._templates.system_reviewer, user_prompt=prompt, metadata={"stage": "review"})
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


DEFAULT_TEMPLATES = PromptTemplates(
    system_analyst="You are an expert software analyst who explains code precisely.",
    prompt_analyst="""Given the following code snippet and metadata, describe in detail what it does, its inputs, outputs, side effects, and error handling.\n\nMetadata: {metadata}\n\nCode:\n{source}\n""",
    system_reviewer="You are a meticulous reviewer ensuring the explanation covers all important behaviors.",
    prompt_reviewer="""Review the provided code and summary. If the summary is sufficient, respond with 'Approve' followed by optional minor notes. Otherwise, rewrite the summary to address the gaps.\n\nMetadata: {metadata}\nSummary:\n{summary}\n\nCode:\n{source}\n""",
    system_synthesizer="You are a technical writer composing cohesive documentation.",
    prompt_synthesizer="""Combine the following component summaries into a higher-level narrative that explains the purpose of the parent component.\n\nMetadata: {metadata}\nSummaries:\n{summaries}\n""",
)
