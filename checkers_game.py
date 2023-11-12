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
                if row in [0, 1, 2] and (row + col) % 2 == 1:
                    #add a P1 piece to the row
                    board_row.append(Piece(1))

                # check if the square should have a P2 piece: 
                # 1. the piece should be on rows 5, 6 or 7
                # 2. the piece have to be on dark squares 
                elif row in [5, 6, 7] and (row + col) % 2 == 1:
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

    def get_legal_moves(self, player=None):
        #if no player specified, use the current player
        if player is None:
            player = self.current_player

        #return a list of legal moves for a player
        legal_moves = []

        # iterate over the board to find all pieces of the given player
        for row_index, row in enumerate(self.state):
            for col_index, piece in enumerate(row):
                if piece and piece.player == player:
                    # compute legal moves for a piece
                    piece_legal_moves = self.get_piece_legal_moves(piece, row_index, col_index)
                    # add legal moves for this piece to all legal moves
                    legal_moves.extend(piece_legal_moves)

        #check if there are any jump moves, if yes then filter out regular ones
        if any(move_type == 'jump' for _, move_type in legal_moves):
            return [move for move, move_type in legal_moves if move_type == 'jump']
        
        return [move for move, move_type in legal_moves if move_type == 'regular']

    def get_piece_legal_moves(self, piece, row, col):
        # returns a list of legal moves for a specific piece
        moves = []

        # determine move directions based on piece type
        if piece.king:
            # diagonal in all directions
            move_directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            # diagonal in one direction
            move_directions = [(1, -1), (1, 1)] if piece.player == 1 else [(-1, -1), (-1, 1)]

        # check for moves and jump moves in each direction
        for d_row, d_col in move_directions:
            # define the position of the adjacent square in the direction of a movement
            adj_row, adj_col = row + d_row, col + d_col

            # check for jump moves
            # if move the adjacent square is within the board and not empty
            if self.is_move_within_board(adj_row, adj_col) and self.state[adj_row][adj_col] is not None:
                # if piece on the adjacent square belongs to the other player
                if self.state[adj_row][adj_col].player != piece.player:
                    # define potential jump landing square
                    jump_row, jump_col = adj_row + d_row, adj_col + d_col
                    # if landing square is within the board and empty
                    if self.is_move_within_board(jump_row, jump_col) and self.state[jump_row][jump_col] is None:
                        # add legal jump move
                        moves.append(((row, col, jump_row, jump_col), 'jump'))

            # check for regular moves
            elif self.is_move_within_board(adj_row, adj_col) and self.state[adj_row][adj_col] is None:
                moves.append(((row, col, adj_row, adj_col), 'regular'))

        return moves
            
    def is_move_within_board(self, row, col):
        # check if the given square is within the board
        return 0 <= row < 8 and 0 <= col < 8

    def make_move(self, move):
        #update the board state based on the given move
        start_row, start_col, end_row, end_col = move

        # initialize the reward counter for this move
        reward = 0

        # check if move is legal
        if move in self.get_legal_moves():
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
                reward += 10
                print(f'Reward for capturing an opponent piece applied to player {self.current_player}')
            
            # check if piece has to be promoted to a king
            if (end_row == 0 and piece.player == 2) or (end_row == 7 and piece.player == 1):
                piece.make_king()

                #assign a reward for promoting to a king
                reward += 5
                print(f'Reward for promoting to a king applied to player {self.current_player}')

            # check for additional captures
            if capture_made:
                additional_captures = self.get_piece_legal_moves(piece, end_row, end_col)
                if any(move_type == 'jump' for _, move_type in additional_captures):
                    # update the reward count
                    self.reward_count[self.current_player] += reward
                    return # do not switch player turn since another jump is possible
                # no additional jumps possible so switch turn

            # small penalty for a normal move without immediate benefit
            if reward == 0:
                reward -= 0.1
                print(f'Small penalty for regular move applied to player {self.current_player}')

            # update the reward count
            self.reward_count[self.current_player] += reward

            # check if game is over after each move
            if self.is_game_over():
                # update rewards based on game over conditions
                return
            else: 
                self.switch_player_turn()
        
        else:
            print('Illegal move')

            # penalty for an illegal move
            self.reward_count[self.current_player] -= 100
    
    def switch_player_turn(self):
        # switch turn between players
        self.current_player = 3 - self.current_player

    def is_game_over(self):
        # check if one of the players won and assign rewards/penalties
        if self.pieces_player_1 == 0:
            self.reward_count[2] += 50
            self.reward_count[1] -= 50
            print("Player 2 wins")
            return True
        elif self.pieces_player_2 == 0:
            self.reward_count[1] += 50
            self.reward_count[2] -= 50
            print("Player 1 wins")
            return True

        # check for stalemate and apply penalty
        if not self.get_legal_moves():
            self.reward_count[1] -= 5
            self.reward_count[2] -= 5
            print("Stalemate")
            return True
        
        return False