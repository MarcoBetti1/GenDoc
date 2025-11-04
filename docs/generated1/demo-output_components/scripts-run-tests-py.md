```markdown
# scripts/run_tests.py

## Overview
The `run_tests.py` script is a crucial component of the chess game management system, designed to facilitate the execution of test scripts based on user-defined JSON configuration files. It supports both sequential and parallel execution modes, allowing for flexible testing scenarios. The script also includes functionality for collecting configuration files, validating moves, and summarizing test results.

## Key Responsibilities
- **Collect Configuration Files**: 
  - Gathers unique JSON configuration file paths from a comma-separated input string.
  
- **Run Tests Sequentially**: 
  - Executes a series of Python scripts in a specified order based on the provided configuration files.
  
- **Run Tests in Parallel**: 
  - Executes Python scripts concurrently, enhancing performance and reducing overall execution time.
  
- **Print Test Summary**: 
  - Summarizes and displays the results of the executed tests, including return codes, durations, and any skipped tests.
  
- **Main Execution Control**: 
  - Manages the entry point for the script, handling command-line arguments and determining the execution mode (sequential or parallel).

## Collaboration Points
- **Integration with Configuration Management**: 
  - Works closely with the configuration management system to ensure accurate and valid configuration file paths are utilized.
  
- **Error Handling Coordination**: 
  - Collaborates with other components to ensure robust error handling and reporting mechanisms are in place during script execution.
  
- **User Interface Interaction**: 
  - Provides feedback to the user interface regarding execution results and statuses, enhancing user experience.

## Implementation Notes
- **Function: `collect_config_files`**
  - Collects unique JSON configuration file paths from a comma-separated input string. Returns a sorted list of valid paths, skipping inaccessible ones.

- **Function: `_run_sequential`**
  - Executes Python scripts sequentially based on configuration files. Supports dry runs and can stop on the first error encountered.

- **Function: `_run_parallel`**
  - Executes Python scripts in parallel, with an option to limit the number of concurrent jobs. Handles process termination gracefully on errors.

- **Function: `_print_summary`**
  - Summarizes test results and prints them to the console. Assumes well-formed input and may raise exceptions for unexpected data structures.

- **Function: `main`**
  - The entry point for the script, managing command-line arguments and execution modes. Validates input and exits with specific status codes for errors.

This script is designed to be efficient and user-friendly, providing essential functionality for testing the chess game management system while ensuring robust performance and error handling.
```
