## Overview

The `ExistingDocsCollector` class is designed to streamline the process of gathering existing documentation files from a specified repository root. It efficiently identifies relevant documentation while excluding files from "generated" directories, ensuring that the collected data is both pertinent and manageable.

## Key Responsibilities

- **Initialization**: The constructor initializes the collector with the repository root path, setting the stage for documentation retrieval.
  
- **Documentation Collection**: The `collect` method is the core functionality, responsible for:
  - Iterating through supported file extensions to locate documentation files.
  - Recursively searching for files while skipping any located in "generated" directories.
  - Reading the contents of found files and storing them in a dictionary for easy access.

- **Error Handling**: The class logs warnings for any `OSError` encountered during file reading, allowing the collection process to continue smoothly despite potential issues.

## Collaboration Points

- **Integration with Other Components**: The `ExistingDocsCollector` works in conjunction with the `ProjectAnalyzer` and `PromptOrchestrator` to ensure that the documentation generated is comprehensive and contextually relevant.
  
- **Dependency on `SUPPORTED_EXTENSIONS`**: This class relies on the `SUPPORTED_EXTENSIONS` variable, which should be defined elsewhere in the codebase to specify the file types considered as documentation.

## Implementation Notes

- **Constructor**: 
  - The constructor does not perform type checking on the `repo_root` parameter. Providing an invalid path may lead to exceptions during execution.

- **Main Method**: 
  - The `collect` method returns a dictionary where the keys are file paths and the values are the corresponding file contents. This structure facilitates easy access to documentation data.

```python
def collect(self) -> Dict[Path, str]:
    # Iterates over SUPPORTED_EXTENSIONS to find documentation files
    # Skips files in "generated" directories
    # Logs warnings for unreadable files
```

- **Side Effects**: The class logs warnings for any unreadable files, which aids in debugging and ensures that users are informed of potential issues during the documentation collection process.
