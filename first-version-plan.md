# GenDoc v0.1 Working Example Plan

## Objective
Deliver a demonstrable prototype that ingests a smaller repository and produces layered documentation (ultra low-level notes → contextual summaries → project overview) using OpenAI prompts. The prototype should emphasize a modular orchestration capable of traversing up and down abstraction levels, incorporating iterative agent-driven refinement loops, so we can validate the core workflow before investing in automation or scale.

## MVP Scope
- **Target Repo**: One curated sample project (e.g., a simple Python CLI or Node.js service) kept alongside the prototype for repeatable runs.
- **Documentation Depth**: Multi-layer, driven by modular passes.
   1. Ultra low-level narration for every method/class, grounded in source and inline comments.
   2. Context-aware summaries that answer “What is the purpose of this in the context of X?” for progressively larger components.
   3. Final project-wide synopsis assembled from the higher-level summaries.
- **Interface**: A command-line script that accepts a path to the sample repo and emits Markdown files.
- **LLM Usage**: Deterministic prompt templates orchestrated by cooperating agents; no retrieval or fine-tuning yet.
- **Output**: A single consolidated Markdown document plus structured intermediate artifacts (per-method notes, module summaries, orchestration manifest) and optional reconciled references to any existing documentation supplied.
- **Optional Inputs**: Ability to ingest pre-existing documentation (README files, design docs) as supplemental context without treating it as authoritative.

## Guardrails & Assumptions
- Runs on a developer workstation with valid OpenAI API credentials set via environment variables.
- Repository size capped (~<2000 LOC) to avoid prompt/token limits in the prototype.
- Optional ingestion of existing documentation is best-effort; conflicts default to source-of-truth in code.
- Errors are surfaced as readable logs; graceful degradation over perfect UX.
- Multi-language repos, CI integration, version history, and automated regeneration are out-of-scope for v0.1.

## Core Workflow (Happy Path)
1. **Project Intake & Structural Mapping**
   - Validate path, detect dominant language, gather file list (respecting ignore rules).
   - Build a lightweight dependency graph (imports, directory hierarchy, call signatures).
   - Present a summarized project tree to an LLM to propose initial groupings or entry points for documentation passes.
2. **Ultra Low-Level Extraction**
   - Traverse each method/class to capture source, docstrings, comments, and metadata (file path, dependencies, call sites).
   - Generate detailed explanations per element, explicitly describing control flow, side effects, and data contracts.
   - Introduce an iterative review loop where an “analyst” agent critiques first-pass summaries and requests clarifications or re-prompts when needed.
3. **Contextual Purpose Derivation**
   - For each grouping proposed earlier, ask “What is the purpose of this in the context of <parent grouping>?” using the collected low-level notes and any relevant existing documentation excerpts as evidence.
   - Allow stepping up or down the hierarchy when prompts reveal gaps (e.g., re-query a subgroup for clarity) and support agent-to-agent handoffs (e.g., delegate refinement tasks to a reviewer agent).
4. **Hierarchical Synthesis**
   - Iterate upward by merging related purpose statements into module/component briefs, then subsystem narratives.
   - Maintain traceability from high-level statements back to the underlying notes to support revisions and permit targeted re-prompts when discrepancies appear.
5. **Assembly & Export**
   - Organize outputs into Markdown: start with the highest abstraction, then embed/append the subordinate layers.
   - Emit all intermediate artifacts plus a run manifest (inputs, timestamps, prompt templates, grouping decisions).

## Key Questions to Resolve During Implementation
- How should we chunk code to balance context richness with token limits? Focus on chunking strategy. We can prompt an llm with the project directory tree and some instructions to split the code into related different parts, classes, folders (Not sure, keep this modular but choose some instruction for a starting example). Then we can see if any of those parts is too large (code tokens will exceed a threshold we will create) then we will split furthur.
- Which metadata (file path, dependencies, docstrings) most improves LLM accuracy? 
- What formatting keeps the generated Markdown readable while signaling which sections are AI-authored? It should all follow the same format, all ai-authored. Whatever the most common project documentation format is.
- How do we cache or deduplicate prompts to control latency and cost in iterative runs? We will save all raw prompts and responses. Also we will use batching where necessary. But since this is for cost saving it should be an option to do batching (when at a step with multiple prompts that dont depend on the output of another, send all in batch). We should have an option to run without batching for testing. Iterative review loops must still log their intermediate artifacts.

## Implementation Milestones
1. **Prototype Scaffold** (Day 0-1)
   - Set up repo structure (src/, samples/, docs/).
   - Add configuration loading (API keys, run options).
2. **Static Analysis Layer** (Day 1-2)
   - Implement file discovery with ignore patterns.
   - Parse the sample language (start with Python `ast` or JavaScript `acorn`/`esprima`).
   - Produce chunk objects with source text + metadata plus optional existing-document excerpts tied to each chunk.
3. **Prompt Engine** (Day 2-3)
   - Draft prompt templates for low-level extraction, contextual purpose probing, synthesis passes, reviewer/critic loops, and sanity-check backs.
   - Implement reusable wrapper for OpenAI chat completions (retry, rate-limit handling, cost tracking, prompt/response ledger, batching toggles).
4. **Generation Pipeline** (Day 3-4)
   - Orchestrate flow: structure heuristics → per-element notes → contextual roll-ups → global synopsis, with agents delegating between passes.
   - Implement the ability to revisit lower levels when higher-level prompts indicate missing detail, including targeted re-prompts that incorporate reviewer feedback.
   - Blend existing documentation snippets where helpful and annotate provenance in outputs.
   - Assemble Markdown using templates that expose the hierarchy and cite reviewer/batching decisions where relevant.
5. **CLI Entry Point** (Day 4)
   - `gendoc run --repo samples/demo-app --out docs/generated.md` style command.
   - Basic logging, progress feedback, and insights into grouping/selection decisions.
   - Flags to enable/disable existing-document ingestion, batching, and reviewer loops.
6. **Demo & Validation** (Day 5)
   - Run against the sample repo.
   - Manual review for coherence, factual accuracy, traceability, and structure.
   - Capture learnings for next iteration (e.g., grouping heuristics, prompt tuning, reviewer effectiveness, backtracking triggers).

## Deliverables for v0.1
- Executable CLI script or package with instructions in `README.md`.
- Sample repository bundled under `samples/` for deterministic evaluation.
- Generated documentation artifact stored in `docs/generated/`.
- Intermediate artifacts (per-method notes, grouping manifests, synthesis transcripts) stored for inspection.
- Ledger of prompts/responses (including reviewer iterations) with optional batching metadata.
- Cross-reference map showing where existing documentation informed generated sections.
- Notes on prompt templates, grouping heuristics, reviewer agents, and configuration, living in a `docs/prototype-notes.md` file.

## Immediate Next Actions
- Choose the sample project (language + repository).
- Define initial prompt templates for ultra low-level, contextual purpose, synthesis, reviewer/critic, and sanity-check passes.
- Draft repository structure and initialize version control settings.
- Enumerate required environment variables and dependencies for onboarding.
- Outline heuristics for LLM-guided grouping, criteria for when to recurse back into lower levels, and rules for blending optional existing documentation with code-derived insight.
