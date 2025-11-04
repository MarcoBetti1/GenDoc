```markdown
# llmchess_simple/prompting.py

## Overview
The `prompting.py` module is a critical component of the `llmchess` chess game management system. It is designed to configure and generate prompts for a language model (LLM) to facilitate automated gameplay and analysis. The module supports various prompting modes, allowing for flexible interaction with the chess game based on user preferences and game state.

## Key Responsibilities
- **Prompt Configuration**: The `PromptConfig` class manages the structure and parameters of prompts used by the LLM, including modes such as plaintext, FEN, and combined formats.
- **Message Generation**: The module provides functions to build structured messages for the LLM based on the current game state:
  - `build_plaintext_messages`: Generates messages for a plaintext prompting system, incorporating move history and game context.
  - `build_fen_messages`: Constructs messages using FEN representation, recent moves in PGN format, and the current side to move.
  - `build_fen_plaintext_messages`: Creates messages that combine FEN data with plaintext, including move history and game initiation status.
- **Error Handling**: While the module does not implement extensive error handling, it assumes valid inputs for its operations, with specific checks in place for certain functions.

## Collaboration Points
- **Integration with Game Engine**: The prompting module interacts closely with the game execution engine, providing necessary prompts based on the current game state and player actions.
- **User Interface Coordination**: The module's output is utilized by the user interface to present game information and facilitate user interactions, ensuring a seamless experience for both human and AI players.
- **AI Decision-Making**: The prompts generated are critical for the AI's decision-making process, influencing the quality and relevance of the moves suggested by the LLM.

## Implementation Notes
- **PromptConfig Class**: 
  - Attributes include `mode`, `starting_context_enabled`, `request_format`, `system_plaintext`, `system_fen`, and `instruction_line`.
  - The class does not enforce strict validation on attribute values, relying on the user to provide valid configurations.

- **Message Building Functions**:
  - `build_plaintext_messages`: Assumes valid `cfg` attributes and does not validate `side` or `history_text`.
  - `build_fen_messages`: May raise `AttributeError` if the `cfg` instance is malformed; assumes valid inputs otherwise.
  - `build_fen_plaintext_messages`: Includes a check for empty `history_text` to ensure meaningful prompts.

This module is essential for enabling effective communication between the chess game management system and the LLM, enhancing the overall gameplay experience and analysis capabilities.
```
