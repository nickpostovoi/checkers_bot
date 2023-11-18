from game_environment import Piece, Board
from dqn_agent import DQN_agent
import numpy as np
import random

# Initialize the game environment and the DQN agent
checkers_game = Board()
state_size = len(checkers_game.get_state_representation())
agent = DQN_agent(state_size, 340, initial_epsilon=1)

episodes = 2000
batch_size = 512
save_interval = 100
replay_interval = 10

# Initialize lists to store cumulative rewards after each episode
cumulative_rewards_p1 = []
cumulative_rewards_p2 = []

for episode in range(episodes):
    checkers_game = Board()
    state = np.array(checkers_game.get_state_representation())
    state = np.reshape(state, [1, state_size])

    while not checkers_game.is_game_over():
        legal_moves = checkers_game.get_legal_moves()

        # Random opponent plays as Player 1
        if checkers_game.current_player == 1:
            action = random.choice(legal_moves)
        else:
            # Agent plays as Player 2
            action = agent.act(state, legal_moves)

        immediate_reward = checkers_game.make_move(action)
        next_state = np.array(checkers_game.get_state_representation())
        next_state = np.reshape(next_state, [1, state_size])
        done = checkers_game.is_game_over()

        # Remember experience and train when the agent (Player 2) is playing
        if checkers_game.current_player == 2:
            agent.remember(state, action, immediate_reward, next_state, done)

        state = next_state

    # Update cumulative rewards
    total_cumulative_reward_p1 = checkers_game.reward_count[1]
    total_cumulative_reward_p2 = checkers_game.reward_count[2]
    cumulative_rewards_p1.append(total_cumulative_reward_p1)
    cumulative_rewards_p2.append(total_cumulative_reward_p2)
    print(f"E{episode}. P1: {total_cumulative_reward_p1}, P2: {total_cumulative_reward_p2}")

    # Periodic replay experiences
    if episode % replay_interval == 0 and len(agent.memory) > batch_size:
        agent.replay(batch_size)

    # Periodic saving of weights and memory
    if episode % save_interval == 0 or episode == episodes - 1:
        agent.save(f"model_checkpoints_p2/checkers_model_episode_{episode}.h5")
        agent.save_memory(f"model_checkpoints_p2/checkers_memory_episode_{episode}.pkl")
        print(f"Saved model and memory at episode {episode}")