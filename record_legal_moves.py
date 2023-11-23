import random
from game_environment import Piece, Board

# initialize a set to store all unique legal moves encountered
all_legal_moves = set()

# run the game 100000 times
for game in range(100000):
    board = Board()
    while not board.is_game_over():
        legal_moves = board.get_legal_moves()
        move = random.choice(legal_moves)

        # add the legal moves to the set
        for legal_move in legal_moves:
            all_legal_moves.add(legal_move)

        board.make_move(move)

# save a set of unique legal moves to a txt file
with open("legal_moves.txt", "w") as file:
    for move in all_legal_moves:
        file.write(f"{move}\n")

# print out number of legal moves (340)
print(f"Total unique legal moves recorded: {len(all_legal_moves)}")
