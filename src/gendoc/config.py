from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class GenDocConfig:
    """Runtime configuration for the GenDoc pipeline."""

    repo_path: Path
    output_path: Path
    use_existing_docs: bool = False
    enable_batching: bool = False
    enable_reviewer: bool = True
    llm_provider: str = "mock"
    max_chunk_tokens: int = 1800
    ledger_path: Optional[Path] = None
    project_tree_prompt_tokens: int = 2000

    @classmethod
    def from_args(cls, args: "argparse.Namespace") -> "GenDocConfig":
        repo_path = Path(args.repo).expanduser().resolve()
        output_path = Path(args.out).expanduser().resolve()
        ledger_path = Path(args.ledger).expanduser().resolve() if args.ledger else None

        config = cls(
            repo_path=repo_path,
            output_path=output_path,
            use_existing_docs=args.use_existing_docs,
            enable_batching=args.enable_batching,
            enable_reviewer=args.enable_reviewer,
            llm_provider=args.llm_provider,
            max_chunk_tokens=args.max_chunk_tokens,
            ledger_path=ledger_path,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {self.repo_path}")
        if not self.repo_path.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {self.repo_path}")
        if self.llm_provider not in {"mock", "openai"}:
            raise ValueError("llm_provider must be 'mock' or 'openai'")

    def ensure_output_dirs(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.ledger_path:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            default_ledger = self.output_path.parent / "prompt-ledger.jsonl"
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
        artefacts_dir = self.config.output_path.parent / "artefacts"
        artefacts_dir.mkdir(parents=True, exist_ok=True)
        self.artefacts_dir = artefacts_dir
