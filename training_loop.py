from game_environment import Piece, Board
from dqn_agent import DQN_agent
import numpy as np

# maximum number of moves in any state is less than 32
max_possible_moves = 340

# initialize the game environment and the DQN agent
checkers_game = Board()
state_size = len(checkers_game.get_state_representation())
agent = DQN_agent(state_size, max_possible_moves)

# number of games to train on
episodes = 1000
# size of batch used in the replay
batch_size = 32

for episode in range(episodes):
    # reset the game environment at the start of each episode
    checkers_game = Board()
    state = np.array(checkers_game.get_state_representation())
    state = np.reshape(state, [1, state_size])

    while not checkers_game.is_game_over():
        # get current set of legal moves
        legal_moves = checkers_game.get_legal_moves()
        # make agent decide
        action = agent.act(state, legal_moves)
        # apply action to the environment
        checkers_game.make_move(action)
        
        # get next state, reward and check if the game is over
        next_state = np.array(checkers_game.get_state_representation())
        next_state = np.reshape(next_state, [1, state_size])
        reward = checkers_game.reward_count[checkers_game.current_player]
        done = checkers_game.is_game_over()

        agent.remember(state, action, reward, next_state, done)