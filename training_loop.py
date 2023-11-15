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
episodes = 20
# size of batch used in the replay
batch_size = 128

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
        
        # apply action to the environment and get an immediate reward
        try: 
            immediate_reward = checkers_game.make_move(action)
        except ValueError as e:
            print(f"Error occured: {e}")
            break
        
        print(checkers_game.print_board())
        # get next state, reward and check if the game is over
        next_state = np.array(checkers_game.get_state_representation())
        next_state = np.reshape(next_state, [1, state_size])
        done = checkers_game.is_game_over()

        # remember the current experience with the immediate reward
        agent.remember(state, action, immediate_reward, next_state, done)

        state = next_state

        if done:
            break

    # display total cumulative reward for the current player at the end of each episode
    total_cumulative_reward_1 = checkers_game.reward_count[1]
    total_cumulative_reward_2 = checkers_game.reward_count[2]
    print(f"Episode {episode} finished. Total reward (P1): {total_cumulative_reward_1}, Total reward (P1): {total_cumulative_reward_2}")

    # if enough experiences are gathered, start replay
    agent.replay(batch_size)

    # save the model every n episodes
    if episode % 100 == 0:
        agent.save(f"model_checkpoints/checkers_model_episode_{episode}.h5")
        