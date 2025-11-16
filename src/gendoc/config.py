from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(slots=True)
class LLMSettings:
    provider: str = "mock"
    default_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_output_tokens: Optional[int] = None
    timeout_seconds: int = 60
    max_retries: int = 3
    stage_models: dict[str, str] = field(default_factory=dict)
    stage_temperatures: dict[str, float] = field(default_factory=dict)

    def model_for(self, stage: Optional[str]) -> Optional[str]:
        if stage and stage in self.stage_models:
            return self.stage_models[stage]
        return self.default_model

    def temperature_for(self, stage: Optional[str]) -> float:
        if stage and stage in self.stage_temperatures:
            return self.stage_temperatures[stage]
        return self.temperature


@dataclass(slots=True)
class FeatureToggles:
    include_run_section: bool = True
    include_repo_layout: bool = True
    include_cli_reference: bool = True
    include_component_docs: bool = True
    generalize_document_text: bool = True
    component_doc_limit: int = 50


@dataclass(slots=True)
class AnalyzerSettings:
    ignore_dirs: tuple[str, ...] = (".git", "venv", "__pycache__", ".mypy_cache")


@dataclass(slots=True)
class ExistingDocsSettings:
    extensions: tuple[str, ...] = (".md", ".rst", ".txt")
    exclude_substrings: tuple[str, ...] = ("generated",)
    max_excerpt_chars: int = 500


@dataclass(slots=True)
class RunSectionSettings:
    enabled: bool = True
    doc_keywords: tuple[str, ...] = (
        "readme",
        "usage",
        "getting-started",
        "setup",
        "install",
        "run",
        "quickstart",
    )
    fallback_docs: tuple[str, ...] = ("README.md", "docs/README.md", "samples/demo_app/README.md")
    source_paths: tuple[str, ...] = ("pyproject.toml", "src/gendoc/cli.py")
    max_context_entries: int = 12
    max_excerpt_chars: int = 4000


@dataclass(slots=True)
class OutputSettings:
    cross_reference_filename: str = "cross-reference.json"
    artefacts_dir_name: str = "artefacts"
    component_dir_suffix: str = "_components"
    default_ledger_filename: str = "prompt-ledger.jsonl"


@dataclass(slots=True)
class GenDocConfig:
    """Runtime configuration for the GenDoc pipeline."""

    repo_path: Path
    output_path: Path
    use_existing_docs: bool = False
    enable_batching: bool = False
    enable_reviewer: bool = True
    max_chunk_tokens: int = 1800
    ledger_path: Optional[Path] = None
    project_tree_prompt_tokens: int = 2000
    sections: tuple[str, ...] = field(default_factory=tuple)
    llm_settings: LLMSettings = field(default_factory=LLMSettings)
    feature_toggles: FeatureToggles = field(default_factory=FeatureToggles)
    analyzer_settings: AnalyzerSettings = field(default_factory=AnalyzerSettings)
    existing_docs_settings: ExistingDocsSettings = field(default_factory=ExistingDocsSettings)
    run_section_settings: RunSectionSettings = field(default_factory=RunSectionSettings)
    output_settings: OutputSettings = field(default_factory=OutputSettings)
    config_path: Optional[Path] = None

    @property
    def llm_provider(self) -> str:
        return self.llm_settings.provider

    @classmethod
    def from_args(cls, args: "argparse.Namespace") -> "GenDocConfig":
        file_data, config_path = cls._load_file_blob(args)
        repo_value = cls._coalesce(
            args.repo,
            file_data.get("paths", {}).get("repo") if isinstance(file_data.get("paths"), dict) else None,
            default="samples/demo_app",
        )
        output_value = cls._coalesce(
            args.out,
            file_data.get("paths", {}).get("output") if isinstance(file_data.get("paths"), dict) else None,
            default="docs/generated/output.md",
        )
        ledger_value = cls._coalesce(
            args.ledger,
            file_data.get("paths", {}).get("ledger") if isinstance(file_data.get("paths"), dict) else None,
            default=None,
        )

        repo_path = Path(str(repo_value)).expanduser().resolve()
        output_path = Path(str(output_value)).expanduser().resolve()
        ledger_path = Path(str(ledger_value)).expanduser().resolve() if ledger_value else None

        pipeline_section = file_data.get("pipeline", {}) if isinstance(file_data.get("pipeline"), dict) else {}
        sections_raw = args.sections if args.sections is not None else pipeline_section.get("sections")
        sections = tuple(str(section).strip() for section in (sections_raw or []) if str(section).strip())

        config = cls(
            repo_path=repo_path,
            output_path=output_path,
            use_existing_docs=cls._coalesce(args.use_existing_docs, pipeline_section.get("use_existing_docs"), default=False),
            enable_batching=cls._coalesce(args.enable_batching, pipeline_section.get("enable_batching"), default=False),
            enable_reviewer=cls._coalesce(args.enable_reviewer, pipeline_section.get("enable_reviewer"), default=True),
            max_chunk_tokens=cls._coalesce(args.max_chunk_tokens, pipeline_section.get("max_chunk_tokens"), default=1800),
            ledger_path=ledger_path,
            project_tree_prompt_tokens=int(pipeline_section.get("project_tree_prompt_tokens", 2000)),
            sections=sections,
            llm_settings=cls._parse_llm_settings(args, file_data.get("llm", {}) if isinstance(file_data.get("llm"), dict) else {}),
            feature_toggles=cls._parse_feature_toggles(file_data.get("features", {}) if isinstance(file_data.get("features"), dict) else {}),
            analyzer_settings=cls._parse_analyzer_settings(file_data.get("analysis", {}) if isinstance(file_data.get("analysis"), dict) else {}),
            existing_docs_settings=cls._parse_existing_docs_settings(file_data.get("existing_docs", {}) if isinstance(file_data.get("existing_docs"), dict) else {}),
            run_section_settings=cls._parse_run_section_settings(file_data.get("run_section", {}) if isinstance(file_data.get("run_section"), dict) else {}),
            output_settings=cls._parse_output_settings(file_data.get("output", {}) if isinstance(file_data.get("output"), dict) else {}),
            config_path=config_path,
        )
        config.validate()
        return config

    @staticmethod
    def _parse_llm_settings(args: "argparse.Namespace", llm_section: Dict[str, Any]) -> LLMSettings:
        provider = GenDocConfig._coalesce(args.llm_provider, llm_section.get("provider"), default="mock")
        default_model = GenDocConfig._coalesce(args.llm_model, llm_section.get("default_model"), default="gpt-4o-mini")
        temperature = float(llm_section.get("temperature", 0.0))
        max_output = llm_section.get("max_output_tokens")
        timeout_seconds = int(llm_section.get("timeout_seconds", 60))
        max_retries = int(llm_section.get("max_retries", 3))
        stage_models = {
            str(key): str(value)
            for key, value in (llm_section.get("stage_models") or {}).items()
            if value
        }
        stage_temperatures = {
            str(key): float(value)
            for key, value in (llm_section.get("stage_temperatures") or {}).items()
        }
        max_output_int = int(max_output) if max_output is not None else None
        return LLMSettings(
            provider=provider,
            default_model=default_model,
            temperature=temperature,
            max_output_tokens=max_output_int,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            stage_models=stage_models,
            stage_temperatures=stage_temperatures,
        )

    @staticmethod
    def _parse_feature_toggles(section: Dict[str, Any]) -> FeatureToggles:
        return FeatureToggles(
            include_run_section=bool(section.get("include_run_section", True)),
            include_repo_layout=bool(section.get("include_repo_layout", True)),
            include_cli_reference=bool(section.get("include_cli_reference", True)),
            include_component_docs=bool(section.get("include_component_docs", True)),
            generalize_document_text=bool(section.get("generalize_document_text", True)),
            component_doc_limit=int(section.get("component_doc_limit", 50)),
        )

    @staticmethod
    def _parse_analyzer_settings(section: Dict[str, Any]) -> AnalyzerSettings:
        ignore_dirs = tuple(str(item).strip() for item in section.get("ignore_dirs", [".git", "venv", "__pycache__", ".mypy_cache"]) if str(item).strip())
        return AnalyzerSettings(ignore_dirs=ignore_dirs)

    @staticmethod
    def _parse_existing_docs_settings(section: Dict[str, Any]) -> ExistingDocsSettings:
        extensions = tuple(str(item).strip() for item in section.get("extensions", [".md", ".rst", ".txt"]) if str(item).strip())
        exclude = tuple(str(item).strip() for item in section.get("exclude", ["generated"]) if str(item).strip())
        max_excerpt_chars = int(section.get("max_excerpt_chars", 500))
        return ExistingDocsSettings(extensions=extensions, exclude_substrings=exclude, max_excerpt_chars=max_excerpt_chars)

    @staticmethod
    def _parse_run_section_settings(section: Dict[str, Any]) -> RunSectionSettings:
        keywords = tuple(str(item).strip() for item in section.get("doc_keywords", RunSectionSettings.doc_keywords) if str(item).strip())
        fallback_docs = tuple(str(item).strip() for item in section.get("fallback_docs", RunSectionSettings.fallback_docs) if str(item).strip())
        source_paths = tuple(str(item).strip() for item in section.get("source_paths", RunSectionSettings.source_paths) if str(item).strip())
        return RunSectionSettings(
            enabled=bool(section.get("enabled", True)),
            doc_keywords=keywords or RunSectionSettings.doc_keywords,
            fallback_docs=fallback_docs or RunSectionSettings.fallback_docs,
            source_paths=source_paths or RunSectionSettings.source_paths,
            max_context_entries=int(section.get("max_context_entries", RunSectionSettings.max_context_entries)),
            max_excerpt_chars=int(section.get("max_excerpt_chars", RunSectionSettings.max_excerpt_chars)),
        )

    @staticmethod
    def _parse_output_settings(section: Dict[str, Any]) -> OutputSettings:
        return OutputSettings(
            cross_reference_filename=str(section.get("cross_reference_filename", "cross-reference.json")),
            artefacts_dir_name=str(section.get("artefacts_dir_name", "artefacts")),
            component_dir_suffix=str(section.get("component_dir_suffix", "_components")),
            default_ledger_filename=str(section.get("default_ledger_filename", "prompt-ledger.jsonl")),
        )

    @staticmethod
    def _load_file_blob(args: "argparse.Namespace") -> tuple[Dict[str, Any], Optional[Path]]:
        candidate_paths: list[Path] = []
        if getattr(args, "config", None):
            candidate_paths.append(Path(args.config).expanduser())
        default_path = Path("gendoc.config.toml")
        if default_path.exists():
            candidate_paths.append(default_path.resolve())
        if getattr(args, "repo", None):
            repo_candidate = Path(str(args.repo)).expanduser()
            repo_file = repo_candidate / "gendoc.config.toml"
            if repo_file.exists():
                candidate_paths.append(repo_file.resolve())
        for path in candidate_paths:
            if path.exists() and path.is_file():
                with path.open("rb") as handle:
                    return tomllib.load(handle), path
        return {}, None

    @staticmethod
    def _coalesce(*values: Any, default: Any) -> Any:
        for value in values:
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
        return default

    def validate(self) -> None:
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {self.repo_path}")
        if not self.repo_path.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {self.repo_path}")
        if self.llm_settings.provider not in {"mock", "openai"}:
            raise ValueError("llm_provider must be 'mock' or 'openai'")

    def ensure_output_dirs(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.ledger_path:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            default_ledger = self.output_path.parent / self.output_settings.default_ledger_filename
            default_ledger.parent.mkdir(parents=True, exist_ok=True)
            self.ledger_path = default_ledger


@dataclass(slots=True)
class RunContext:
    """Mutable state shared across pipeline stages."""

    config: GenDocConfig
    prompt_count: int = 0
    token_count: int = 0
    batches_executed: int = 0
    artefacts_dir: Path = field(default_factory=Path)

    def init_paths(self) -> None:
        artefacts_dir = self.config.output_path.parent / self.config.output_settings.artefacts_dir_name
        artefacts_dir.mkdir(parents=True, exist_ok=True)
        self.artefacts_dir = artefacts_dir
