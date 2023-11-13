### Development Progress:

**12.11.2023**
  - Implemented a reward/penalty system and end-game logic in the `Board` class:
    - Introduced a reward/penalty mechanism: This new functionality assigns rewards or penalties to the AI agent based on the moves it makes. This includes positive reinforcement for capturing opponent pieces and promoting pieces to kings, and penalties for illegal or non-strategic moves.
    - Added end-game conditions: The `Board` class now includes logic to determine the end of the game, considering conditions such as one player losing all pieces or a stalemate. This includes assigning final rewards or penalties based on these outcomes.
    - Revised `make_move` method: Updated to incorporate the reward system and check for game-ending conditions after each move.
  - Began development of the AI agent (`DQN_agent` class):
    - This agent utilizes a deep Q-network (DQN) to learn the optimal strategies for playing checkers. It is designed to interact with the game environment provided by the `Board` class.

**11.11.2023**
  - Enhanced the `Board` class:
    - Added `get_legal_moves` method: This function is pivotal in identifying all the permissible moves for the current player or a specified player. It first identifies all the pieces belonging to the player and then computes the legal moves for each piece, including regular and jump moves.
    - Introduced `get_piece_legal_moves` method: A specialized method for calculating the legal moves for a specific piece, considering its type (king or regular) and position. It accounts for both regular and jump moves, adhering to checkers' rules.
    - Implemented `is_move_within_board` method: A utility function to verify if a move stays within the boundaries of the checkers board, which is essential for determining the validity of the moves.
    - Updated `print_board` method: Enhanced the visual representation of the board. This method now prints the board with clearer formatting, making it more user-friendly and easier to interpret.
    - Developed `make_move` function: This key method updates the board state based on the given move. It includes logic to handle captures, multi-jumps, and piece promotion to king.

**10.11.2023**
  - Implemented foundational classes for the game:
    - `Piece`: Represents individual checker pieces. It includes player identity, king status, and a method for promoting a piece to a king.
    - `Board`: Initializes the checkers board with pieces in starting positions. Manages player turns and includes skeleton functions for legal move checking and executing moves.



