# GenDoc: Streamlining Documentation Generation for Software Projects

## At a Glance
- **Mission**: GenDoc aims to simplify the documentation process for software projects by automating the extraction of code elements and generating structured documentation.
- **Architecture**: The system is built around a modular architecture that includes components for code analysis, documentation generation, user interaction, and error handling.
- **LLM Integration**: Leveraging advanced language models, GenDoc generates human-readable summaries and instructions based on the analyzed code and existing documentation.
- **Distinguishing Traits**: The repository uniquely combines code analysis with user-friendly interfaces, allowing for seamless integration of existing documentation and robust error handling throughout the documentation generation process.

## Functional Flow
1. **User Input**: The user initiates the documentation generation process through a command-line interface, specifying the repository path and desired output location.
2. **Configuration Setup**: The system reads and validates configuration settings, ensuring all necessary directories exist for output files.
3. **Code Analysis**: The ProjectAnalyzer scans the specified repository, discovering Python files and extracting relevant code elements, such as classes and functions.
4. **Documentation Generation**: Using templates and prompts, the system generates structured documentation, including project goals, component breakdowns, and run instructions.
5. **User Interaction**: Users can refine the generated documentation through prompts, receiving feedback and summaries from the integrated language model.
6. **Output Creation**: The final documentation is written to the specified output location, along with logs of the interactions for auditing purposes.

## To Run

### Prerequisites
- Python 3.11 or higher installed
- pip package manager installed

### Windows
1. Open PowerShell.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   ```bash
   .\venv\Scripts\Activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the documentation generation:
   ```bash
   python -m gendoc.cli
   ```

### macOS
1. Open Terminal.
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```
3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the documentation generation:
   ```bash
   python -m gendoc.cli
   ```

### Notes
Ensure to replace 'requirements.txt' with the actual file if it exists, or install dependencies directly using pip. You're on the right track—just follow these steps, and you'll have your documentation generated in no time!

## Repository Layout

- **samples/**: Contains example files and templates that demonstrate the usage of the documentation generation features.
- **src/**: The main source directory housing the implementation code for the documentation generation tool.
  - **gendoc/**: A subdirectory within `src/` that includes the core classes and methods for code analysis and documentation generation.
- **design-spec.md**: A document outlining the design specifications and architectural decisions for the project.
- **first-version-plan.md**: A planning document detailing the objectives and milestones for the initial version of the project.
- **project-goal.md**: A file that articulates the overarching goals and objectives of the project.
- **pyproject.toml**: A configuration file for managing project dependencies and settings in a standardized format.
- **README.md**: The primary documentation file providing an overview of the project, installation instructions, and usage guidelines.

## CLI Reference

This section provides a detailed overview of the command-line interface (CLI) arguments available for the documentation generation tool. Each argument is described along with its purpose, default values, and any relevant notes.

- `run`: This positional argument is optional and defaults to `"run"`. It controls the execution of the documentation pipeline. If provided, it indicates the action to be performed.

- `--repo`: This argument specifies the path to the repository that you want to document. It defaults to `"samples/demo_app"`. You can change this to point to any valid repository path.

- `--out`: This argument defines the destination Markdown file where the generated documentation will be saved. The default value is `"docs/generated/output.md"`. You can specify a different file path if desired.

- `--use-existing-docs`: This flag, when included, enables the inclusion of existing documentation as supplemental context in the generated output. It does not require a value and is set to `False` by default.

- `--enable-batching`: This flag enables prompt batching where supported. It is set to `False` by default. If you want to enable batching, include this flag in your command.

- `--disable-batching`: This flag disables prompt batching. It is an alternative to `--enable-batching` and will set batching to `False`. 

- `--enable-reviewer`: This flag enables reviewer agent loops, allowing for additional review processes in the documentation generation. It is set to `True` by default. You can disable it by using the `--disable-reviewer` flag.

- `--disable-reviewer`: This flag disables the reviewer agent loops. It is an alternative to `--enable-reviewer` and will set the reviewer functionality to `False`.

- `--llm-provider`: This argument specifies the language model backend to use for generating documentation. It accepts two choices: `"mock"` and `"openai"`, with `"mock"` being the default option.

- `--max-chunk-tokens`: This argument sets the maximum token budget per code chunk during processing. The default value is `1800`. You can adjust this value based on your needs.

- `--ledger`: This optional argument allows you to specify a path for the prompt ledger JSONL file, which can be used for tracking and auditing purposes.

- `--log-level`: This argument sets the logging level for the application. It accepts values such as `DEBUG`, `INFO`, and `WARNING`, with `INFO` being the default.

For more detailed information on each argument and its usage, you can run the CLI with the `--help` flag to display the help message.

## Component Breakdown

### 1. Code Analysis
- **Module**: `analysis.py` (See [analysis.py](demo-output_components/src-gendoc-analysis-py.md) for details.)
- **Responsibilities**: This module is responsible for parsing Python code, extracting key components, and organizing project structure. It includes classes like `CodeElement`, `ProjectStructure`, and `ProjectAnalyzer`.
- **Collaboration**: Works closely with the `pipeline.py` module to provide structured data for documentation generation.

### 2. Documentation Generation
- **Module**: `pipeline.py` (See [analysis.py](demo-output_components/src-gendoc-analysis-py.md) for details.)
- **Responsibilities**: Coordinates the overall documentation generation process, integrating code analysis results and existing documentation. It manages the creation of sections, summaries, and final documents.
- **Collaboration**: Interacts with the `prompting.py` module to utilize language models for generating human-readable content.

- Also see [pipeline.py](demo-output_components/src-gendoc-pipeline-py.md) — Facilitate documentation generation and refinement for software projects.

### 3. User Interaction
- **Module**: `cli.py`
- **Responsibilities**: Provides a command-line interface for users to interact with the system, allowing them to specify parameters and initiate the documentation generation process.
- **Collaboration**: Interfaces with the `pipeline.py` module to execute the documentation generation workflow based on user inputs.

### 4. Prompt Management
- **Module**: `prompting.py` (See [prompting.py](demo-output_components/src-gendoc-prompting-py.md) for details.)
- **Responsibilities**: Manages interactions with the language model, including prompt creation, logging, and response handling. It includes classes like `PromptOrchestrator`, `PromptLedger`, and `MockLLM`.
- **Collaboration**: Works with the `pipeline.py` module to refine and generate documentation based on user prompts and existing content.

- Reference: [cli.py](demo-output_components/src-gendoc-cli-py.md) — **Component**: `src/gendoc/cli.py`
- Also see [config.py](demo-output_components/src-gendoc-config-py.md) — **Purpose**: Manages and validates configuration settings for the GenDoc pipeline.
- Also see [existing_docs.py](demo-output_components/src-gendoc-existing-docs-py.md) — **Class**: `ExistingDocsCollector`

## Outputs & Observability
- **Generated Artefacts**: The system produces structured documentation in Markdown format, including project goals, component breakdowns, and run instructions.
- **Logs**: Interaction logs are maintained for auditing purposes, capturing prompts, responses, and timestamps for each user interaction.
- **Metrics**: The system tracks the number of code elements processed, existing documentation integrated, and the overall success of the documentation generation process, providing insights into system performance and usage.
