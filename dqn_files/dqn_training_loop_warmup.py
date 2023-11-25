from game_environment import Piece, Board
from dqn_agent import DQN_agent
import numpy as np
import random
import gc
from keras import backend as K
from multiprocessing import Pool
from collections import deque

# initialize the game environment and the DQN agent
checkers_game = Board()
state_size = len(checkers_game.get_state_representation())
agent = DQN_agent(340, initial_epsilon=1)

weights_path = "model_checkpoints/checkers_model_cnn_9x9_episode_455000.h5"

# load the model weights and memory if the files exist
try:
    agent.load(weights_path)
    print("Loaded model weights.")
except Exception as e:
    print(f"Error loading model weights: {e}")

def play_game(episode):
    checkers_game = Board()
    previous_reward_p1 = 0
    previous_reward_p2 = 0
    moves_count = 0

    experiences = []
    capture_buffer = []

    while not checkers_game.is_game_over():
        add_capt_flag = 0
        player = checkers_game.current_player
        # get the current state and available moves
        state = np.array(checkers_game.get_state_representation())
        legal_moves = checkers_game.get_legal_moves()
        # evaluate the board and perform an action
        action = random.choice(legal_moves)
        # makes a move, switches the current_player, returns 5 if add captures available
        add_capt_flag = checkers_game.make_move(action) 
        #save state after the move and check if move ended the game
        next_state = np.array(checkers_game.get_state_representation())
        done = checkers_game.is_game_over()
        # calculate the reward
        current_reward = checkers_game.reward_count[player]
        if player == 1:
            reward = current_reward - previous_reward_p1
            previous_reward_p1 = current_reward
        else:
            reward = current_reward - previous_reward_p2
            previous_reward_p2 = current_reward
        # store experience in a local memory
        if add_capt_flag == 5:
            # store intermediate states and rewards for multiple captures
            capture_buffer.append((state, reward))            
        else:
            # check if there were multiple captures
            if capture_buffer:
                # use the state before the first capture and the final state
                initial_state, _ = capture_buffer[0]
                total_reward = sum(r for _, r in capture_buffer) + reward
                experiences.append((player, initial_state, action, total_reward, next_state, done))
                capture_buffer.clear()
            else:
                # regular move or signular capture
                experiences.append((player, state, action, reward, next_state, done))
        
        # increment the move count
        moves_count += 1
    
    return experiences

episodes = 2000000
batch_size = 100000
save_interval = 50000
replay_interval = 5000
parallel_games = 12

# pool for parallel game execution
with Pool(processes=parallel_games) as pool:
    for episode_batch in range(0, episodes, replay_interval):
        # prepare arguments for parallel game playing
        game_args = [(episode,) for episode in range(episode_batch, episode_batch + replay_interval)]
        
        # play games in parallel
        all_experiences = pool.starmap(play_game, game_args)

        for experiences in all_experiences:
            for experience in experiences:
                agent.remember(*experience)

        # after replay_interval number of games perform replay
        if len(agent.memory_p1) > batch_size and len(agent.memory_p2) > batch_size:
            agent.replay(1, batch_size)
            agent.replay(2, batch_size)
            gc.collect()
            print(f"E{episode_batch + replay_interval}")
            print('----------------')

        # periodic saving of weights
        if episode_batch % save_interval == 0 or episode_batch + replay_interval >= episodes:
            agent.save(f"model_checkpoints/checkers_model_cnn_9x9_episode_{episode_batch + replay_interval}.h5")
            print(f"SW {episode_batch + replay_interval}")
            K.clear_session()
            gc.collect()

