## Overview

The `src/gendoc/config.py` module is a critical component of the GenDoc documentation generation pipeline. It defines the configuration management and runtime state necessary for the pipeline's operation, ensuring that all settings are validated and maintained throughout the documentation process.

## Key Responsibilities

### GenDocConfig
- **Configuration Management**: Manages and validates essential configuration settings for the GenDoc pipeline.
- **Attributes**:
  - `repo_path`: Specifies the path to the source repository.
  - `output_path`: Designates the path for generated output files.
  - `use_existing_docs`: A boolean flag indicating whether to utilize existing documentation (default: `False`).
  - `enable_batching`: A boolean flag to enable batching of requests (default: `False`).
  - `enable_reviewer`: A boolean flag to activate the reviewer feature (default: `True`).
  - `llm_provider`: Specifies the language model provider (default: `"mock"`).
  - `max_chunk_tokens`: Sets the maximum number of tokens per chunk (default: `1800`).
  - `ledger_path`: An optional path for the ledger file (default: `None`).
  - `project_tree_prompt_tokens`: Defines the token limit for project tree prompts (default: `2000`).

### RunContext
- **State Management**: Maintains mutable state across different stages of the pipeline, tracking metrics such as prompt and token counts.
- **Attributes**:
  - `config`: An instance of `GenDocConfig`.
  - `prompt_count`: Tracks the number of prompts processed (default: `0`).
  - `token_count`: Records the total number of tokens processed (default: `0`).
  - `batches_executed`: Counts the number of executed batches (default: `0`).
  - `artefacts_dir`: Specifies the path for storing generated artefacts.

## Collaboration Points

- **Integration with Other Components**: The `GenDocConfig` class interacts with various components of the GenDoc pipeline, ensuring that configuration settings are consistently applied across the system.
- **Error Handling**: The methods within `GenDocConfig` raise specific exceptions for validation failures, which can be handled by upstream components to maintain robustness in the pipeline.

## Implementation Notes

- The `from_args` method is crucial for creating an instance of `GenDocConfig` from command-line arguments, validating paths, and raising exceptions for any invalid configurations. 

```python
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
```

- The `validate` method ensures that the `repo_path` exists and is of the correct type, while also validating the `llm_provider`.
- The `ensure_output_dirs` and `init_paths` methods create necessary directories for output and artefacts, handling existing paths gracefully to avoid errors.
