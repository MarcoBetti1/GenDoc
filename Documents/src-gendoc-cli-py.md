## Overview
The `src/gendoc/cli.py` component serves as the command-line interface (CLI) for the GenDoc project. It facilitates user interaction by allowing users to execute documentation generation tasks through command-line commands. The component is designed to be user-friendly while ensuring robust error handling and logging capabilities.

## Key Responsibilities
- **Argument Parsing**: The `build_parser` function creates a configured `ArgumentParser` that defines the command-line interface, including positional and optional arguments.
- **Logging Configuration**: The `configure_logging` function sets up the logging framework for the application, allowing for adjustable logging levels.
- **Main Execution Flow**: The `main` function acts as the entry point for the CLI, handling argument parsing, logging configuration, command validation, and the execution of the documentation generation pipeline.

## Collaboration Points
- **Integration with Other Components**: This CLI component interacts with the configuration and pipeline modules, specifically through the `GenDocConfig` and `Pipeline` classes, which are responsible for managing the documentation generation process.
- **User Feedback**: The CLI provides feedback to users based on their input, including error messages for invalid commands or arguments, enhancing the overall user experience.

## Implementation Notes
- The `build_parser` function does not take any inputs and returns a configured `ArgumentParser`. Beyond the standard `--repo` and `--out` flags, it now exposes `--config` (path to a TOML config), `--llm-model` (per-run model override), and paired toggles such as `--use-existing-docs` / `--no-existing-docs` so users can temporarily override the file-backed defaults.
- The `configure_logging` function modifies the global logging settings based on the provided logging level, defaulting to `INFO` for invalid inputs.
- The `main` function validates the command input, ensuring that only the "run" command is accepted. It initializes the configuration and executes the pipeline for documentation generation.

Important code excerpt from the `main` function:
```python
def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    if args.run not in {"run", None}:
        parser.error(f"Unknown command '{args.run}'. Only 'run' is supported in v0.1.")

    config = GenDocConfig.from_args(args)
    pipeline = Pipeline(config)
    pipeline.run()
``` 

This structure ensures that the CLI is both functional and user-friendly, providing a solid foundation for the documentation generation process within the GenDoc project.
