from game_environment import Piece, Board
from dqn_agent import DQN_agent
import numpy as np
import random
import gc
from keras import backend as K

# initialize the game environment and the DQN agent
checkers_game = Board()
state_size = len(checkers_game.get_state_representation())
agent = DQN_agent(340, initial_epsilon=1)

# weights_path = "model_checkpoints/checkers_model_tf_episode_42600.h5"

# # load the model weights and memory if the files exist
# try:
#     agent.load(weights_path)
#     print("Loaded model weights.")
# except Exception as e:
#     print(f"Error loading model weights: {e}")

episodes = 2000000
batch_size = 10000
save_interval = 10000
replay_interval = 500

# initialize a list to store cumulative rewards after each episode
cumulative_rewards_p1 = []
cumulative_rewards_p2 = []
 
for episode in range(episodes):
    checkers_game = Board()
    previous_reward_p1 = 0
    previous_reward_p2 = 0
    moves_count = 0

    while not checkers_game.is_game_over():
        state = np.array(checkers_game.get_state_representation())
        legal_moves = checkers_game.get_legal_moves()
        
        action = agent.act(state, legal_moves)
        checkers_game.make_move(action)

        next_state = np.array(checkers_game.get_state_representation())
        done = checkers_game.is_game_over()

        # calculate the reward for player current player
        current_reward = checkers_game.reward_count[checkers_game.current_player]
        if checkers_game.current_player == 1:
            reward = current_reward - previous_reward_p1
            previous_reward_p1 = current_reward
        else: 
            reward = current_reward - previous_reward_p2
            previous_reward_p2 = current_reward

        # remember the experience for player 1
        agent.remember(checkers_game.current_player, state, action, reward, next_state, done)

        moves_count += 1

    # update cumulative rewards
    total_cumulative_reward_p1 = checkers_game.reward_count[1]
    total_cumulative_reward_p2 = checkers_game.reward_count[2]

    # cumulative_rewards_p1.append(total_cumulative_reward_p1)
    # cumulative_rewards_p2.append(total_cumulative_reward_p2)

    # print(f"E{episode}. P1: {np.round(total_cumulative_reward_p1, 1)}, P2: {np.round(total_cumulative_reward_p2, 1)}, M: {moves_count}")

    # periodic replay experiences
    if episode % replay_interval == 0 and len(agent.memory_p1) > batch_size and len(agent.memory_p2) > batch_size:
        agent.replay(1, batch_size)
        agent.replay(2, batch_size)
        print('----------------')
        # K.clear_session()
        gc.collect()
        print(f"E{episode}")

    # periodic saving of weights and memory
    if episode % save_interval == 0 or episode == episodes - 1:
        agent.save(f"model_checkpoints/checkers_model_tf_episode_{episode}.h5")
        print(f"SW {episode}")
        K.clear_session()
        gc.collect()