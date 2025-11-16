## Overview

`src/gendoc/config.py` now centralizes every knob used by the pipeline. It loads a layered configuration stack (default values → `gendoc.config.toml` → CLI overrides), validates the result, and hands richly typed settings to the rest of the system. This keeps the CLI lean while giving power users a single file to tweak models, artefact names, or optional documentation sections.

## Core Dataclasses

- **`GenDocConfig`** – top-level state: repo/output paths, batching/reviewer flags, chunk/token limits, requested sections, plus nested settings objects. It exposes helpers like `ensure_output_dirs()` and `validate()`.
- **`LLMSettings`** – provider, base model, retry policy, temperature, optional per-stage overrides via `stage_models` and `stage_temperatures`, plus helpers (`model_for`, `temperature_for`).
- **`FeatureToggles`** – switches for including the run instructions, repo layout, CLI reference, component deep dives, and whether to generalize sample-specific text. Also caps the number of component docs emitted.
- **`AnalyzerSettings`** – directory ignore list for the AST walk, allowing per-repo overrides (e.g., skip `node_modules`).
- **`ExistingDocsSettings`** – allowed extensions, exclusion markers, and maximum excerpt length when harvesting supplemental docs.
- **`RunSectionSettings`** – keyword filters, fallback doc paths, source snippets, context limits, and enablement for the “To Run” generator.
- **`OutputSettings`** – names for artefact directories, cross-reference manifests, component subfolders, and the default prompt ledger file.
- **`RunContext`** – unchanged in spirit but now derives the artefact directory name from `OutputSettings`.

## File Loading Flow

1. The CLI parser accepts `--config` (optional). If omitted, `GenDocConfig.from_args` searches for `gendoc.config.toml` in:
   - the working directory
   - the provided repo path (so configs can ship with the target project)
2. The TOML blob is parsed via `tomllib` and merged with CLI flags. Flags always win.
3. Nested sections are converted into their respective dataclasses (`LLMSettings`, `FeatureToggles`, etc.), each with sane defaults when the TOML omits fields.
4. `validate()` ensures the repo exists and the selected LLM provider is supported; additional checks can be added per section as the surface grows.

## Collaboration Points

- **CLI (`src/gendoc/cli.py`)** – the parser now feeds a `Namespace` straight into `GenDocConfig.from_args`, relying on the config module to merge TOML + overrides.
- **Pipeline** – consumes feature toggles (`include_run_section`, `generalize_document_text`, etc.), analyzer ignore lists, and the strongly typed LLM settings when constructing orchestrators.
- **Prompting layer** – reads `LLMSettings` to determine per-stage model/temperature routing, ensuring every downstream call is consistent.
- **Sections/Run guide** – uses `RunSectionSettings` to decide which files to mine for context and how many excerpts to keep.

## Example Usage

```python
parser = build_parser()
args = parser.parse_args()
config = GenDocConfig.from_args(args)
config.ensure_output_dirs()
pipeline = Pipeline(config)
pipeline.run()
```

With this structure a user can, for example, create `gendoc.config.toml` with a custom OpenAI model map, disable component docs, and raise the excerpt budget—all without changing code. Individual runs can still flip switches (`--no-existing-docs`, `--llm-model`, `--disable-reviewer`) when experimentation is needed.
