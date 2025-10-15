# Prototype Notes

## Environment Variables
- `OPENAI_API_KEY`: Secret key used when `--llm-provider openai` is selected.
- `OPENAI_ORG_ID` (optional): Override organization for API usage.

## Prompt Templates (Draft)
- Element narration template emphasizing control flow, inputs/outputs, side effects.
- Reviewer checklist template verifying coverage of branches, error handling, and dependencies.
- Contextual purpose template tying elements back to parent scope.
- Synthesis template requesting narrative cohesion and references to lower layers.
- Project-goal synthesis template that compresses section summaries into a single objective statement.
- Relevance filter template that trims or omits sections based on goal alignment.

## Batch Controls
- Batching is disabled by default; enable with `--enable-batching` once prompt templates are deterministic.
- Ledger records `batch_id` fields to correlate grouped prompts.

## Outstanding Tasks
- Finalize prompt wordings and acceptance criteria.
- Implement OpenAI client with retry/back-off and cost estimation.
- Expand sample library beyond Python once the pipeline stabilizes.
