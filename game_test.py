# try the game with random turns
import random
import time

from checkers_game_mirrored import Piece, Board

#create a board
board = Board()

#start a timer
start_time = time.time()

while not board.is_game_over():
    legal_moves = board.get_legal_moves()

    move = random.choice(legal_moves)
    board.make_move(move)
    board.print_board()
    print("Reward Count:", board.reward_count)
    print("Player 1 pieces: ", board.pieces_player_1)
    print("Player 2 pieces: ", board.pieces_player_2)
    print("\n-------------------------------------------\n")

# end the timer
end_time = time.time()
# calculate time taken to play a game
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")