from game_environment import Piece, Board
from dqn_agent import DQN_agent
import numpy as np
import random
import gc
from keras import backend as K

# initialize the game environment and the DQN agent
checkers_game = Board()
state_size = len(checkers_game.get_state_representation())
agent = DQN_agent(state_size, 340, initial_epsilon=1)

weights_path = 'model_checkpoints/checkers_model_episode_1481001.h5'
memory_path = 'model_checkpoints/checkers_memory.pkl'

# load the model weights and memory if the files exist
try:
    agent.load(weights_path)
    print("Loaded model weights.")
except Exception as e:
    print(f"Error loading model weights: {e}")

try:
    agent.load_memory(memory_path)
    print("Loaded memory.")
except Exception as e:
    print(f"Error loading memory: {e}")

episodes = 20000
batch_size = 2500
save_interval = 500
replay_interval = 10

# initialize a list to store cumulative rewards after each episode
cumulative_rewards_p1 = []
cumulative_rewards_p2 = []

for episode in range(episodes):
    checkers_game = Board()
    state = np.array(checkers_game.get_state_representation())
    state = np.reshape(state, [1, state_size])

    while not checkers_game.is_game_over():
        legal_moves = checkers_game.get_legal_moves()

        # agent makes a move
        action = agent.act(state, legal_moves)

        immediate_reward = checkers_game.make_move(action)
        next_state = np.array(checkers_game.get_state_representation())
        next_state = np.reshape(next_state, [1, state_size])
        done = checkers_game.is_game_over()

        # remember the experience
        agent.remember(state, action, immediate_reward, next_state, done)
        state = next_state

        state = next_state

    # update cumulative rewards
    total_cumulative_reward_p1 = checkers_game.reward_count[1]
    total_cumulative_reward_p2 = checkers_game.reward_count[2]
    cumulative_rewards_p1.append(total_cumulative_reward_p1)
    cumulative_rewards_p2.append(total_cumulative_reward_p2)
    print(f"E{episode}. P1: {total_cumulative_reward_p1}, P2: {total_cumulative_reward_p2}")

    # periodic replay experiences
    if episode % replay_interval == 0 and len(agent.memory) > batch_size:
        agent.replay(batch_size)
        gc.collect()
        
    # periodic saving of weights and memory
    if episode % save_interval == 0 or episode == episodes - 1:
        agent.save(f"model_checkpoints/checkers_model_episode_{episode+1481002}.h5")
        agent.save_memory(f"model_checkpoints/checkers_memory.pkl")
        print(f"Saved model and memory at episode {episode+1481002}")
        K.clear_session()