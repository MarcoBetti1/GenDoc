```markdown
# Component Documentation: `llmchess_simple/move_validator.py`

## Overview
The `move_validator.py` component is a crucial part of the chess game management system, responsible for validating and normalizing chess moves. It ensures that all moves made by players, whether human or AI, adhere to the rules of chess. This component leverages the `chess` library for board state management and utilizes regular expressions for move format validation.

## Key Responsibilities
- **Move Extraction**: 
  - `_extract_candidate`: Extracts valid chess moves from input text in UCI (Universal Chess Interface) or SAN (Standard Algebraic Notation) format.

- **Legal Move Generation**: 
  - `_legal_moves_set`: Generates a set of legal moves in UCI format based on a given FEN (Forsyth-Edwards Notation) string.

- **Move Legality Check**: 
  - `is_legal_uci`: Validates whether a UCI move is legal according to the current FEN state.

- **Sorted Legal Moves**: 
  - `legal_moves`: Returns a sorted list of legal UCI moves for a specified FEN string.

- **Move Normalization**: 
  - `normalize_move`: Validates and normalizes user input for chess moves, providing structured feedback on validity and formats.

## Collaboration Points
- **Chess Library**: Utilizes the `chess` library for managing board states and performing legality checks on moves.
- **Regular Expressions**: Interacts with regex to validate UCI and SAN formats, ensuring that user inputs conform to expected patterns.
- **Legal Moves Retrieval**: Calls `_legal_moves_set` to obtain legal moves necessary for legality checks and move validation.

## Implementation Notes
- **Error Handling**: Implements basic error handling for invalid UCI formats and FEN strings, relying on the `chess` library to manage exceptions.
- **Input Validation**: Assumes that inputs are valid strings; non-string types may raise a `TypeError`.
- **No Side Effects**: Functions within this component do not modify any external state; they operate solely on the inputs provided to them, ensuring predictable behavior.
```
