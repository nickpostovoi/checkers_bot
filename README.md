### Development Progress:

**10.11.2023**
  - Implemented foundational classes for the game:
    - `Piece`: Represents individual checker pieces. It includes player identity, king status, and a method for promoting a piece to a king.
    - `Board`: Initializes the checkers board with pieces in starting positions. Manages player turns and includes skeleton functions for legal move checking and executing moves.

**11.11.2023**
  - Enhanced the `Board` class:
    - Added `get_legal_moves` method: This function is pivotal in identifying all the permissible moves for the current player or a specified player. It first identifies all the pieces belonging to the player and then computes the legal moves for each piece, including regular and jump moves.
    - Introduced `get_piece_legal_moves` method: A specialized method for calculating the legal moves for a specific piece, considering its type (king or regular) and position. It accounts for both regular and jump moves, adhering to checkers' rules.
    - Implemented `is_move_within_board` method: A utility function to verify if a move stays within the boundaries of the checkers board, which is essential for determining the validity of the moves. 
