import numpy as np
import random
import matplotlib.pyplot as plt
from game_environment import Board, Piece
from dqn_agent import DQN_agent

# load the trained model
checkers_game = Board()
state_size = len(checkers_game.get_state_representation())
action_size = 340
agent1 = DQN_agent(state_size, action_size, initial_epsilon=0.05)
agent1.load('model_checkpoints/checkers_model_episode_1070000 EPS 1 FINISHED.h5')
agent2 = DQN_agent(state_size, action_size, initial_epsilon=0.05)
agent2.load('model_checkpoints/checkers_model_episode_1072500.h5')

episodes = 100
agent_1_wins = 0
agent_2_wins = 0
draws = 0
no_capture_moves = 0

for episode in range(episodes):
    checkers_game = Board()
    state = np.array(checkers_game.get_state_representation())
    state = np.reshape(state, [1, state_size])
    no_capture_moves = 0
    previous_pieces_p1 = checkers_game.pieces_player_1
    previous_pieces_p2 = checkers_game.pieces_player_2

    while not checkers_game.is_game_over():
        legal_moves = checkers_game.get_legal_moves()

        if checkers_game.current_player == 1:
            # agent 1 turn
            action = agent1.act(state, legal_moves)
        else:
            # agent 2 turn
            action = agent2.act(state, legal_moves)

        try:
            checkers_game.make_move(action)
        except ValueError as e:
            print(f"Error occured: {e}")
            break

        state = np.array(checkers_game.get_state_representation())
        state = np.reshape(state, [1, state_size])

        # check for a draw due to no progress made
        if checkers_game.pieces_player_1 == previous_pieces_p1 and checkers_game.pieces_player_2 == previous_pieces_p2:
            no_capture_moves += 1
        else:
            no_capture_moves = 0

        previous_pieces_p1 = checkers_game.pieces_player_1
        previous_pieces_p2 = checkers_game.pieces_player_2

        # declare draw if no captures in the last 50 moves
        if no_capture_moves >= 50:
            draws += 1
            print("Draw declared due to no captures in the last 50 moves")
            break

        print(checkers_game.print_board())

    # determine the winner
    if checkers_game.pieces_player_1 == 0 or not checkers_game.get_legal_moves(player=1):
        agent_2_wins += 1
    elif checkers_game.pieces_player_2 == 0 or not checkers_game.get_legal_moves(player=2):
        agent_1_wins += 1
    else: 
        draws += 1
    
    print(f"Episode {episode} finished.")

    # plot the results
    plt.figure(figsize=(10, 6))
    results = [agent_1_wins, agent_2_wins, draws]
    labels = ['DQN Agent 1 Wins', 'DQN Agent 2 Wins', 'Draws']
    plt.bar(labels, results, color=['blue', 'red', 'green'])
    plt.xlabel('Outcome')
    plt.ylabel('Number of Games')
    plt.title('Performance of DQN Agents Over 100 Games')
    plt.show()