# GenDoc v0.1 Design Spec

## Architectural Overview
GenDoc orchestrates a multi-pass documentation workflow composed of five core subsystems:

1. **Intake & Structural Mapping**
   - Discovers source files, applies ignore rules, and builds a hierarchical tree of modules, classes, and functions.
   - Extracts metadata such as docstrings, import relationships, and inferred dependencies.
   - Optionally gathers existing documentation artifacts (Markdown, reStructuredText, Jupyter notebooks converted to Markdown) and maps them to nearby code elements.

2. **Ultra Low-Level Narration**
   - Iterates over each code element, emitting a first-pass “explain what this does” prompt that includes source code and metadata.
   - A reviewer agent inspects the response, compares it against the original code, and issues follow-up prompts until coverage criteria are met (e.g., branches described, side-effects captured, docstring parity maintained).
   - All prompts, responses, and reviewer decisions are logged to a ledger for traceability.

3. **Contextual Purpose Derivation**
   - Groups elements according to the structural map (module, feature slice, or LLM-suggested cluster) and asks “What role does this grouping play within its parent context?”
   - Incorporates relevant excerpts from existing documentation when available, clearly labeling their provenance so the LLM can reconcile discrepancies.
   - Supports backtracking: if the summary is vague or contradictory, the system requests clarifications from lower-level agents or replays earlier prompts with expanded context.

4. **Hierarchical Synthesis & Assembly**
   - Aggregates mid-level summaries into subsystem narratives, then into a project-wide overview.
   - Produces a final Markdown artifact that begins at the highest abstraction layer and drills down, embedding links to low-level notes.
   - Emits supplemental JSON/Markdown files capturing the grouping decisions, prompt ledger, and cross-reference map linking generated sections to code and existing docs.

5. **Goal Derivation & Editorial Refinement**
   - Synthesizes section-level summaries into an explicit project goal statement using a higher-context model.
   - Revisits each section with the derived goal, pruning details that do not advance the overarching objective and optionally omitting irrelevant sections entirely.
   - Records the final goal alongside cross-reference metadata so downstream consumers can trace why a section was retained.

## Key Modules
- `analysis.ProjectAnalyzer`: Handles file discovery, dependency graph construction, AST inspection, and code element extraction.
- `existing_docs.ExistingDocsCollector`: Finds supplemental documentation, ranks relevance, and associates snippets with code elements.
- `prompting.PromptOrchestrator`: Coordinates low-level, reviewer, contextual, and synthesis prompts. Interfaces with the LLM client, prompt templates, and batching machinery.
- `prompting.LLMClient` implementations:
  - `MockLLM`: Deterministic heuristic responses for offline development.
  - `OpenAIClient`: Thin wrapper over `openai` Chat Completions with retry, exponential back-off, and token accounting (stubbed in v0.1).
- `pipeline.Pipeline`: Ties together analysis, prompting, and assembly. Maintains run state, handles configuration flags, and emits artifacts.
- `artifacts.DocumentAssembler`: Composes Markdown and manifest files from intermediate results.

## Data Artifacts
| Artifact | Format | Purpose |
| --- | --- | --- |
| Prompt Ledger | JSON Lines | Auditable record of prompts/responses, reviewer actions, batching metadata |
| Code Element Notes | Markdown per element | Ultra low-level narration with code excerpts |
| Module Summaries | Markdown/JSON | Contextual role narratives referencing child elements |
| Project Overview | Markdown | Final human-facing narrative |
| Cross-Reference Map | JSON | Links generated sections to code paths, existing documentation, and the derived project goal |
| Run Manifest | JSON | Records config values, timestamps, LLM provider, cost estimates |

## Prompt Workflow
1. **Structure Proposal Prompt**
   - Inputs: Directory tree, file metrics, existing doc titles.
   - Outputs: Suggested clusters, documentation entry points, and priority order.
2. **Element Narration Prompt**
   - Inputs: Code snippet, metadata, relevant doc excerpts.
   - Outputs: Detailed explanation of behavior, inputs/outputs, side effects.
3. **Reviewer Prompt**
   - Inputs: Code, first-pass summary, checklist of required coverage.
   - Outputs: Critique + revision instructions or acceptance notice.
4. **Contextual Purpose Prompt**
   - Inputs: Child element notes, cluster metadata, supplemental docs.
   - Outputs: Summary of the cluster’s role, dependencies, and interactions.
5. **Synthesis Prompt**
   - Inputs: Collection of contextual summaries.
   - Outputs: Higher-level narrative suitable for top-level documentation.
6. **Project Goal Prompt**
   - Inputs: All section summaries (post-review).
   - Outputs: Concise goal statement describing the repository’s purpose.
7. **Relevance Filter Prompt**
   - Inputs: Project goal, section metadata, and section content.
   - Outputs: Refined section text containing only goal-aligned details or an `OMIT` directive if irrelevant.

All prompts must carry a `role` tag (`analyst`, `reviewer`, `synthesizer`, etc.) to aid audit trails and future fine-tuning.

## Configuration Surface (`GenDocConfig`)
- `repo_path`: Path to target repository.
- `output_path`: Destination Markdown file.
- `use_existing_docs`: Toggle supplemental doc ingestion.
- `enable_batching`: Allow batching of independent prompts.
- `enable_reviewer`: Activate reviewer agent loops.
- `llm_provider`: `mock` (default) or `openai`.
- `max_chunk_tokens`: Token budget for element prompts.
- `ledger_path`: Optional custom location for the prompt ledger.

Future options (not in v0.1):
- `language_overrides`: Force specific analyzers per directory.
- `quality_threshold`: Numeric grade to require before finalizing a summary.

## Error Handling & Telemetry
- Analyzer emits warnings for unsupported languages or unparsable files but continues processing others.
- Prompt orchestrator retries transient LLM errors with exponential backoff; on repeated failures it downgraded to a mock response and flags the section in the manifest.
- All exceptions propagate to the CLI with actionable messages; partial artifacts include failure annotations.
- Telemetry counters (prompt count, tokens used, retry attempts) are collected in the run manifest for post-run analysis.

## Open Questions (tracked)
- Optimal heuristics for mapping existing docs to code elements.
- Criteria for terminating reviewer loops automatically vs. requiring manual inspection.
- Strategies for managing very large files (split by class vs. by logical block).

This spec should be treated as the implementation contract for v0.1. As features land, update the document to reflect actual behavior and decisions.
