# Deep Q-Learning with CNN for Checkers

## Introduction
Brief description of the project, its purpose, and what it achieves.

## Table of Contents
- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Methodology](#methodology)
  - [Game Environment Setup](#game-environment)
  - [Deep Q-Network Agent](#deep-q-network-agent)
  - [Training Process](#training-process)
- [Results and Observations](#results-and-observations)
- [Files Description](#files-description)
  - [game_environment.py](#game_environmentpy)
  - [dqn_agent.py](#dqn_agentpy)
  - [training_loop_warmup_parallelized.py](#training_loop_warmup_parallelizedpy)
  - [record_legal_moves.py](#record_legal_movespy)
- [Contact](#contact)

## Technologies Used

## Methodology
### Game Environment

#### Overview
<p> The game environment in this project serves as a simulation of a standard checkers game, crucial for training the agent. It includes the mechanics of a checkers game, such as the game board, piece movement, and game rules, implemented from scratch in Python. This environment is essential for the agent to understand the dynamics of the game and learn strategies through repeated play.</p>

#### Game Mechanics 
<p> The environment consists of two primary components: the `Piece` and `Board` classes. The `Piece` class represents individual checkers pieces, with attributes to identify the player it belongs to and whether it is a 'king'. The `Board` class simulates the checkers game board. It initializes the game state, handles piece movement, checks for legal moves, and manages game outcomes (win, loss, stalemate). </p>

#### Board Initialization
<p> At the start, the board is initialized with the standard checkers setup: pieces are placed on specific squares depending on their color (Player 1 or Player 2). The board also tracks the number of pieces each player has and the current player's turn. </p>

#### Game Progression
<p> The game progresses through the make_move function in the `Board` class, where pieces are moved according to the rules of checkers. This includes regular moves, jump moves, and kinging of pieces. The game state is updated after each move, keeping track of the pieces on the board, and the `current_player` is switched after a valid move. </p>

#### Legal Moves and Rewards
<p> The environment calculates legal moves for the current player and assigns rewards based on the actions taken. For instance, capturing an opponent's piece or kinging a piece provides positive rewards, while making illegal moves or losing pieces incurs penalties. This reward system is integral to the reinforcement learning process, guiding the agent towards beneficial strategies. </p>

### Deep Q-Network Agent

### Training Process

## Results and Observations


## Files Description
### `game_environment.py`

### `dqn_agent.py`

### `training_loop_warmup_parallelized.py`

### `record_legal_moves.py`



