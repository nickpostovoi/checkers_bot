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
agent.load('model_checkpoints/checkers_model_episode_99.h5')

episodes = 100
agent_wins = 0
opponent_wins = 0
draws = 0

for episode in range(episodes):
    checkers_game = Board()
    state = np.array(checkers_game.get_state_representation())
    state = np.reshape(state, [1, state_size])

    while not checkers_game.is_game_over():
        legal_moves = checkers_game.get_legal_moves()

        if checkers_game.current_player == 1:
            # agents turn
            action = agent.act(state, legal_moves)
        else:
            # random opponents turn
            action = random.choice(legal_moves)

        try:
            checkers_game.make_move(action)
        except ValueError as e:
            print(f"Error occured: {e}")
            break

        state = np.array(checkers_game.get_state_representation())
        state = np.reshape(state, [1, state_size])

    # determine the winner
    if checkers_game.pieces_player_1 == 0 or not checkers_game.get_legal_moves(player=1):
        opponent_wins += 1
    elif checkers_game.pieces_player_2 == 0 or not checkers_game.get_legal_moves(player=2):
        agent_wins += 1
    else:
        draws += 1
    
    print(f"Episode {episode} finished.")

    # plot the results
    plt.figure(figsize=(10, 6))
    results = [agent_wins, opponent_wins, draws]
    labels = ['DQN Agent Wins', 'Opponent Wins', 'Draws']
    plt.bar(labels, results, color=['blue', 'red', 'green'])
    plt.xlabel('Outcome')
    plt.ylabel('Number of Games')
    plt.title('Performance of DQN Agent Over 100 Games')
    plt.show()