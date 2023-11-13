# import the game environment
# from checkers_game import Piece, Board

# import necessary libraries
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQN_agent:
    def __init__(self, state_size, action_size):
        # initialize the agent
        
        # state representation from get_state_representation()
        self.state_size = state_size
        # possible actions from get_legal_moves()
        self.action_size = action_size
        # a double-ended queue to store experiences
        self.memory = deque(maxlen=100000)
        # discount rate (determines the importance of future rewards)
        # lower rate makes agent more short-sighted
        # higher rate makes agent value future rewards more significantly (far-sighted)
        self.gamma = 0.95 
        # exploration rate 
        # balances exploration (trying new actions) and exploitation (using the best-known action)
        self.epsilon = 1.0
        # the minimum value that epsilon can reach during the training process
        self.epsilon_min = 0.01
        # rate at which the epsilon value decreases over time
        self.epsilon_decay = 0.995
        # rathe at which the weights in the neural network are adjusted during each training iteration
        self.learning_rate = 0.001
        
        # initialize the model
        self.model = self._build_model()

        def _build_model(self):
            # compiles a neural net for DQ model
            model = Sequential()
            model.add(Dense(128, input_dim=self.state_size, activation='relu'))
            model.add(Dense(256, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
            return model
        
        def remember(self, state, action, reward, next_state, done):
            #enables the agent to store experiences and learn from them effectively through experience replay
            self.memory.append((
                state, # current state of the environment before the agent takes an action 
                action, # action taken by the agent in the current state
                reward, # immediate reward received after taking the action
                next_state, # state of the environment after the action is taken.
                done # boolean indicating whether this state-action pair led to the end of an episode
                ))