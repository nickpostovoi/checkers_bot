from game_environment import Piece, Board
from dqn_agent_with_lstm import DQN_agent
import numpy as np
import random
from collections import deque
import gc
from keras import backend as K

# initialize the game environment and the DQN agent
checkers_game = Board()
state_size = len(checkers_game.get_state_representation())
agent = DQN_agent(state_size, 340, num_past_states=6, initial_epsilon=1)

# weights_path = 'model_checkpoints_lstm/checkers_model_episode_.h5'
# memory_path = 'model_checkpoints_lstm/checkers_memory.pkl'

# # Load the model weights and memory if the files exist
# try:
#     agent.load(weights_path)
#     print("Loaded model weights.")
# except Exception as e:
#     print(f"Error loading model weights: {e}")

# try:
#     agent.load_memory(memory_path)
#     print("Loaded memory.")
# except Exception as e:
#     print(f"Error loading memory: {e}")

episodes = 200000
batch_size = 5000
save_interval = 10000
replay_interval = 100

# initialize a list to store cumulative rewards after each episode
cumulative_rewards_p1 = []
cumulative_rewards_p2 = []

for episode in range(episodes):
    checkers_game = Board()
    initial_state = np.array(checkers_game.get_state_representation())

    # clear and initialize past states for the new game
    past_states = deque(maxlen=agent.num_past_states)
    for _ in range(agent.num_past_states):
        past_states.append(initial_state)

    while not checkers_game.is_game_over():
        current_state = np.array(checkers_game.get_state_representation())

        # prepare the state input for the model (current + past states)
        model_state_input = np.concatenate(list(past_states) + [current_state])
        model_state_input = np.reshape(model_state_input, [1, agent.num_past_states + 1, state_size])

        legal_moves = checkers_game.get_legal_moves()
        # agent makes a move
        action = agent.act(model_state_input, legal_moves)

        immediate_reward = checkers_game.make_move(action)
        next_state = np.array(checkers_game.get_state_representation())
        done = checkers_game.is_game_over()

        # prepare next model state input
        next_model_state_input = np.concatenate(list(past_states)[1:] + [current_state, next_state])

        # remember the experience
        agent.remember(model_state_input, action, immediate_reward, next_model_state_input, done)

        # update current state and past states
        past_states.append(current_state)
    
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
        agent.save(f"model_checkpoints_lstm/checkers_model_episode_{episode}.h5")
        agent.save_memory(f"model_checkpoints_lstm/checkers_memory.pkl")
        print(f"Saved model and memory at episode {episode}")
        K.clear_session()

    # # plotting the bar chart after each episode
    # plt.figure(figsize=(18, 6))
    # bar_width = 0.35  # Width of the bars
    # episodes_axis = np.arange(1, episode + 2)  # Episode numbers

    # plt.bar(episodes_axis - bar_width/2, cumulative_rewards_p1, bar_width, color='blue', alpha=0.6, label='Player 1')
    # plt.bar(episodes_axis + bar_width/2, cumulative_rewards_p2, bar_width, color='red', alpha=0.6, label='Player 2')

    # plt.xlabel('Episode')
    # plt.ylabel('Total Cumulative Reward')
    # plt.title('Total Cumulative Reward After Each Episode')
    # plt.xticks(episodes_axis)  # Set x-ticks to be at the bar centers
    # plt.legend()
    # plt.show()