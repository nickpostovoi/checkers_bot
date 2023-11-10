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
        # initializing the board with pieces in their starting positions
        self.state = self.initialize_board()
        # start with the player 1 (white)
        self.current_player = 1

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
        # iterate through each row on the board
        for row in self.state:
            # for each row construct a string representing the row
            print(' '.join([str(piece) if piece is not None else '-' for piece in row]))

    def get_legal_moves(self, player):
        #return a list of legal moves for a player
        pass

    def is_legal_move(self, move):
        #check of the given move is legal for the current player
        pass

    def make_move(self, move):
        #update the board state based on the given move

        #make sure move is legal before updating a state
        if self.is_legal_move(move):
            #update the state based on the move
            pass
        else:
            #handle illegal move
            pass
    
    def switch_player_turn(self):
        #switch turn between players
        self.current_player = 3 - self.current_player
    
board_object = Board()


