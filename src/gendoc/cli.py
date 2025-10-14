from __future__ import annotations

import argparse
import logging
from typing import Optional

from .config import GenDocConfig
from .pipeline import Pipeline

LOG_FORMAT = "[%(levelname)s] %(message)s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate layered documentation for a repository.")
    parser.add_argument("run", nargs="?", help="Execute the documentation pipeline", default="run")
    parser.add_argument("--repo", default="samples/demo_app", help="Path to the repository to document.")
    parser.add_argument("--out", default="docs/generated/output.md", help="Destination Markdown file.")
    parser.add_argument("--use-existing-docs", action="store_true", help="Include existing documentation as supplemental context.")
    parser.add_argument("--enable-batching", dest="enable_batching", action="store_true", help="Enable prompt batching where supported.")
    parser.add_argument("--disable-batching", dest="enable_batching", action="store_false", help="Disable prompt batching.")
    parser.add_argument("--enable-reviewer", dest="enable_reviewer", action="store_true", help="Enable reviewer agent loops.")
    parser.add_argument("--disable-reviewer", dest="enable_reviewer", action="store_false", help="Disable reviewer agent loops.")
    parser.set_defaults(enable_batching=False, enable_reviewer=True)
    parser.set_defaults(enable_reviewer=True)
    parser.add_argument("--llm-provider", choices=["mock", "openai"], default="mock", help="LLM backend to use.")
    parser.add_argument("--max-chunk-tokens", type=int, default=1800, help="Maximum token budget per code chunk.")
    parser.add_argument("--ledger", help="Optional path for the prompt ledger JSONL file.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING).")
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=LOG_FORMAT)


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    if args.run not in {"run", None}:
        parser.error(f"Unknown command '{args.run}'. Only 'run' is supported in v0.1.")

    config = GenDocConfig.from_args(args)
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
