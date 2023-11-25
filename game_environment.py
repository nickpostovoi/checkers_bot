import numpy as np
from collections import deque

import config as cfg

# defining the piece class
class Piece:
    def __init__(self, player, king=False):
        # initialize a piece with player id and king status
        self.player = player
        self.king = king
    
    def make_king(self):
        # promote the piece to a king
        self.king = True

    def __repr__(self):
        # string represenation of a piece
        return f"{'K' if self.king else 'P'}{self.player}"

# defining the board class
class Board:
    def __init__(self):
        # initialize the board with pieces in their starting positions
        self.state = self.initialize_board()
        # start with the player 1 (white)
        self.current_player = 1
        # initialize the piece counts
        self.pieces_player_1 = 12
        self.pieces_player_2 = 12
        # initialize variable for storing the reward balance
        self.reward_count = {1: 0, 2: 0}
        # load the file with possible legal moves mapping to indices
        self.moves_mapping = self.load_legal_moves()
        # create reverse mapping from indices to moves
        self.reverse_moves_mapping = {index: move for move, index in self.moves_mapping.items()}
        # create state history buffers
        self.state_history = deque(maxlen = cfg.state_history_length)
        self.features_history = deque(maxlen = cfg.state_history_length)
        self.update_state_history()

    def initialize_board(self):
        # initialize an empty list to represent the board
        board = []

        # iterate over 8 rows to create the board
        for row in range(8):
            # initialize an empty list for the current row
            board_row = []

            # iterate over 8 columns to create the current row
            for col in range(8):

                # check if the square should have a P1 piece:
                # 1. the piece should be on rows 0, 1 or 2
                # 2. the piece have to be on dark squares 
                # the dark squares are those where the sum of the row and column indices is odd
                if row in [0, 1, 2] and (row + col) % 2 == 0:
                    #add a P1 piece to the row
                    board_row.append(Piece(1))

                # check if the square should have a P2 piece: 
                # 1. the piece should be on rows 5, 6 or 7
                # 2. the piece have to be on dark squares 
                elif row in [5, 6, 7] and (row + col) % 2 == 0:
                    #add a P2 piece to the row
                    board_row.append(Piece(2))

                # for squares that should be empty append None
                else:
                    board_row.append(None)
            
            # append the completed row to the board
            board.append(board_row)
        
        # return the initialized board
        return board

    def print_board(self):
        # display column headers with padding to align it to cells
        print("   " + "  ".join([f"{i}" for i in range(8)]))

        # iterate through each row
        for row_idx, row in enumerate(self.state):
            # start each row with index and space for alignment
            row_str = f"{row_idx} "

            # for each row construct a string
            for piece in row:
                cell = str(piece) if piece is not None else "-"
                row_str += f"{cell:^3}" # center each cell within a width of 3 char
                
            print(row_str)

    def get_board_representation(self, player=None):
        # if no player specified, use the current player
        if player is None:
            player = self.current_player

        # initialize a 3D array to represent the board, 8x8x4
        board_state = [[[0]*4 for _ in range(8)] for _ in range(8)]

        # encode the board representation using one-hot encoding
        # channel 0 for own piece
        # channel 1 for own king
        # channel 2 for opponents piece
        # channel 3 for opponents king
        for i in range(8):
            for j in range(8):
                piece = self.state[::-1][i][j] if player == 2 else self.state[i][j]
                if piece is not None:
                    if piece.player == player:
                        board_state[i][j][0 if not piece.king else 1] = 1  # own piece or king
                    else:
                        board_state[i][j][2 if not piece.king else 3] = 1  # opponent piece or king

        return np.array(board_state)

    def calculate_vertical_center_of_mass(self, player):
        total_height = 0
        total_pieces = 0

        for row_index, row in enumerate(self.state):
            for piece in row:
                if piece and piece.player == player:
                    total_height += row_index
                    total_pieces += 1

        # avoid division by zero if there are no pieces
        if total_pieces == 0:
            return 0

        # calculate and return the vertical center of mass
        return total_height / total_pieces
    
    def calculate_additional_features(self, player=None):
        # if no player specified, use the current player
        if player is None:
            player = self.current_player

        own_uncrowned, opponent_uncrowned, own_kings, opponent_kings, own_edge_pieces = 0, 0, 0, 0, 0

        # iterate over the board to count pieces and their positions
        for row_index, row in enumerate(self.state):
            for col_index, piece in enumerate(row):
                if piece:
                    if piece.player == player:
                        if piece.king:
                            own_kings += 1
                        else:
                            own_uncrowned += 1
                        if col_index in [0, 7]:
                            own_edge_pieces += 1
                    else:
                        if piece.king:
                            opponent_kings += 1
                        else:
                            opponent_uncrowned += 1

        # calculate the vertical centers of mass
        own_center_of_mass = self.calculate_vertical_center_of_mass(player)
        opponent_center_of_mass = self.calculate_vertical_center_of_mass(3 - player)

        features = [
            own_uncrowned,
            opponent_uncrowned,
            own_kings,
            opponent_kings,
            own_edge_pieces,
            own_center_of_mass,
            opponent_center_of_mass
        ]

        return features

    def update_state_history(self):
        current_state = self.get_board_representation()
        current_features = self.calculate_additional_features()
        self.state_history.appendleft(current_state)
        self.features_history.appendleft(current_features)

    def get_state_representation(self):
        # initialize a list to hold current and historical board states
        all_states = []
        # get current state and add to the list
        current_state = self.get_board_representation()
        all_states.append(current_state)
        # iterate over the state history and each state to the list
        for historical_state in list(self.state_history):
            all_states.append(historical_state)
        # #if there are fewer than required historical states, pad with zeros
        # while len(all_states) =< cfg.state_history_length + 1:
        return all_states

    @staticmethod
    def load_legal_moves():
        import ast
        # load all possible legal moves from .txt file and map them to indices
        with open("legal_moves.txt", "r") as file:
            moves = [ast.literal_eval(line.strip()) for line in file]
            return {move: index for index, move in enumerate(moves)}

    def get_legal_moves(self, player=None, return_indices=True):
        # if no player specified, use the current player
        if player is None:
            player = self.current_player

        legal_moves = []
        jump_moves = []  

        # mirror the orientation of the board for player 2
        state = self.state[::-1] if player == 2 else self.state

        # iterate over the board to find all pieces of the given player
        for row_index, row in enumerate(state):
            for col_index, piece in enumerate(row):
                if piece and piece.player == player:
                    # compute legal moves for a piece
                    piece_legal_moves = self.get_piece_legal_moves(piece, row_index, col_index, player)
                    for move, move_type in piece_legal_moves:
                        if move_type == 'jump':
                            jump_moves.append(move)
                        else:
                            legal_moves.append(move)

        # if any jump moves are available, they take precedence
        final_moves = jump_moves if jump_moves else legal_moves

        # return indices of moves if return_indices is True
        return [self.moves_mapping[move] for move in final_moves] if return_indices else final_moves

    def get_piece_legal_moves(self, piece, row, col, player):
        # returns a list of legal moves for a specific piece
        moves = []

        # mirror the orientation of the board for player 2
        state = self.state[::-1] if player == 2 else self.state
        
        # determine move directions based on piece type
        if piece.king:
            # diagonal in all directions
            move_directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            # diagonal in one direction
            move_directions = [(1, -1), (1, 1)]

        # check for moves and jump moves in each direction
        for d_row, d_col in move_directions:
            # define the position of the adjacent square in the direction of a movement
            adj_row, adj_col = row + d_row, col + d_col

            # check for jump moves
            # if move the adjacent square is within the board and not empty
            if self.is_move_within_board(adj_row, adj_col) and state[adj_row][adj_col] is not None:
                # if piece on the adjacent square belongs to the other player
                if state[adj_row][adj_col].player != piece.player:
                    # define potential jump landing square
                    jump_row, jump_col = adj_row + d_row, adj_col + d_col
                    # if landing square is within the board and empty
                    if self.is_move_within_board(jump_row, jump_col) and state[jump_row][jump_col] is None:
                        # add legal jump move
                        moves.append(((row, col, jump_row, jump_col), 'jump'))

            # check for regular moves
            elif self.is_move_within_board(adj_row, adj_col) and state[adj_row][adj_col] is None:
                moves.append(((row, col, adj_row, adj_col), 'regular'))

        return moves

    def is_move_within_board(self, row, col):
        # check if the given square is within the board
        return 0 <= row < 8 and 0 <= col < 8

    def make_move(self, move):
        #update the board state based on the given move

        # check if the game is already over
        if self.is_game_over():
            raise ValueError("Game is over")

        # check if move is given as an index and convert it to a tuple
        if isinstance(move, int) or isinstance(move, np.int64):
            move = self.reverse_moves_mapping.get(move, None)
            if move is None:
                raise ValueError("Invalid move index provided")

        # invert coordinates for player 2
        if self.current_player == 2:
            start_row, start_col, end_row, end_col = self.invert_coordinates(move)
        else:
            start_row, start_col, end_row, end_col = move

        # initialize the reward counter for this move
        current_player_balance = 0
        opponent_player_balance = 0

        # check if move is legal
        if move in self.get_legal_moves(return_indices=False):
            # move the piece
            piece = self.state[start_row][start_col]
            self.state[end_row][end_col] = piece
            self.state[start_row][start_col] = None

            # check if the move is a jump
            capture_made = False 
            if abs(start_row - end_row) == 2:
                # determine the position of the captured piece
                middle_row, middle_col = (start_row + end_row) // 2, (start_col + end_col) // 2
                
                # remove the captured piece and update the piece counts
                captured_piece = self.state[middle_row][middle_col]
                if captured_piece.player == 1:
                    self.pieces_player_1 -= 1
                elif captured_piece.player == 2:
                    self.pieces_player_2 -= 1
                
                self.state[middle_row][middle_col] = None
                capture_made = True

                # assign a reward for capturing an opponent piece
                current_player_balance += 1
                # assign a penalty to the opponent for losing a piece
                opponent_player_balance -= 0.5
            
            # check if piece has to be promoted to a king
            if (end_row == 0 and piece.player == 2) or (end_row == 7 and piece.player == 1):
                # promote a piece for a king
                piece.make_king()
                #assign a reward for promoting to a king
                current_player_balance += 0.5
                opponent_player_balance -= 0.2

            # check for additional captures
            if capture_made:
                all_legal_moves = self.get_legal_moves(return_indices=False)
                if self.current_player == 1:
                    additional_jumps = [m for m in all_legal_moves if m[0] == end_row and m[1] == end_col and abs(m[2] - (end_row)) == 2]
                else: 
                    additional_jumps = [m for m in all_legal_moves if m[0] == 7 - end_row and m[1] == end_col and abs(m[2] - (7-end_row)) == 2]
                if additional_jumps:
                    # assign an additional reward for a move leading to more captures
                    current_player_balance += 0.5
                    opponent_player_balance -= 0.5
                    
                    # update the reward count
                    self.reward_count[self.current_player] += current_player_balance
                    self.reward_count[3 - self.current_player] += opponent_player_balance

                    self.update_state_history()
                    
                    return # do not switch player turn since another jump is possible
            
            # small penalty for a normal move without immediate benefit
            if current_player_balance == 0:
                current_player_balance -= 0.05

        else:
            # penalty for an illegal move
            current_player_balance = -1

        # update the reward count
        self.reward_count[self.current_player] += current_player_balance
        self.reward_count[3 - self.current_player] += opponent_player_balance

        # check if game is over and assign additional rewards/penalties
        if self.is_game_over():
            if self.pieces_player_1 == 0:
                self.reward_count[2] += 50
                self.reward_count[1] -= 50
            elif self.pieces_player_2 == 0:
                self.reward_count[1] += 50
                self.reward_count[2] -= 50
            elif not self.get_legal_moves():
                # stalemate penalty
                self.reward_count[1] -= 1
                self.reward_count[2] -= 1
        else:
            self.update_state_history()
            # switch player turn if game is not over
            self.switch_player_turn()
        
        return

    def invert_coordinates(self, move):
        # helper to invert coordinates for player 2
        start_row, start_col, end_row, end_col = move
        return 7 - start_row, start_col, 7 - end_row, end_col
    
    def switch_player_turn(self):
        # switch turn between players
        self.current_player = 3 - self.current_player

    def is_game_over(self):
        if self.pieces_player_1 == 0 or self.pieces_player_2 == 0:
            # game ends if either player has no pieces left
            return True

        # check for stalemate
        if not self.get_legal_moves():
            # game ends in a stalemate if there are no legal moves left
            return True
        
        # if none of the above conditions are met, the game continues
        return False

if __name__ == "__main__":
    checkers_game = Board()
    checkers_game.print_board()