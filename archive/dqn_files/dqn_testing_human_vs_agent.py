import numpy as np
import random
import matplotlib.pyplot as plt
from game_environment import Board, Piece
from dqn_agent import DQN_agent

# load the trained model
checkers_game = Board()
action_size = 340
agent = DQN_agent(action_size, initial_epsilon=0)
# agent.load('model_checkpoints/checkers_model_cnn_9x9_episode_255000.h5')


# this command takes an action using the agent

# checkers_game.make_move(
#     agent.act(
#         checkers_game.get_state_representation(), 
#         checkers_game.get_legal_moves()
#         )
#     )
# print(checkers_game.current_player)
# checkers_game.print_board()



# checkers_game.make_move()
# print(checkers_game.current_player)
# checkers_game.print_board()