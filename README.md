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

### 3. Run the CLI against the bundled sample project
```powershell
gendoc run --repo samples/demo_app --out docs/generated/demo-output.md
```

The default run uses a mock LLM so the pipeline can execute without external credentials. To enable real OpenAI calls later, set the environment variables referenced in `docs/prototype-notes.md` and pass `--llm-provider openai`.

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
- Flesh out the analysis layer to support deeper language features.
- Implement real OpenAI prompt calls with retry logic and cost accounting.
- Expand the demo suite with multilingual or framework-specific repositories.

## License
TBD
