# GenDoc Prototype

GenDoc is an experimental pipeline that turns an existing codebase into layered documentation, starting from ultra low-level method walk-throughs and climbing to narrative project overviews via multi-agent prompt orchestration.

## Project Status
- **Stage**: v0.1 planning and scaffold
- **Focus**: Validate the end-to-end workflow on a curated sample project with mock prompts (real OpenAI integration optional).

## Getting Started

### 1. Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install the package in editable mode
```powershell
pip install -e .
```

### 3. Generate layered documentation
```powershell
gendoc run --repo samples/demo_app --out docs/generated/demo-output.md
```

The default run uses a mock LLM so the pipeline can execute without external credentials. To create production-strength docs for the chess harness sample with the real OpenAI API, provide an API key and run:

```powershell
$Env:OPENAI_API_KEY = "sk-..."
gendoc run --repo samples/llmchess --out docs/generated/demo-output.md --llm-provider openai
```

Each run produces a polished Markdown overview plus supporting artefacts (`cross-reference.json`, `prompt-ledger.jsonl`, and per-run assets under `docs/generated/artefacts/`).

## Key CLI Flags
- `--use-existing-docs`: Blend in any Markdown/RST files from the repo as supplemental context.
- `--enable-batching / --disable-batching`: Toggle prompt batching where applicable.
- `--enable-reviewer / --disable-reviewer`: Control whether reviewer agents critique first-pass summaries.
- `--llm-provider`: Choose `mock` (default) or `openai`.

Run `gendoc run --help` for the full list of options.

## Repository Layout
```
GenDoc/
├── docs/              # Generated docs and prototype notes
├── samples/           # Curated sample projects for repeatable runs
├── src/gendoc/        # Source code for the CLI and pipeline
├── project-goal.md    # Vision statement
└── first-version-plan.md
```

## Next Steps
- Implement real OpenAI prompt calls with retry logic and cost accounting.
- Expand the demo suite with multilingual or framework-specific repositories.
