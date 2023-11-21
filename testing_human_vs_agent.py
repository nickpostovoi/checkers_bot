import numpy as np
import random
import matplotlib.pyplot as plt
from game_environment import Board, Piece
from dqn_agent import DQN_agent

# load the trained model
checkers_game = Board()
state_size = len(checkers_game.get_state_representation())
action_size = 340
agent = DQN_agent(state_size, action_size, initial_epsilon=0)
agent.load('model_checkpoints/checkers_model_episode_1481001.h5')

# this command takes an action using the agent
checkers_game.make_move(agent.act(np.reshape(np.array(checkers_game.get_state_representation()), [1, state_size]), checkers_game.get_legal_moves()))