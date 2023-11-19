# import the game environment
# from checkers_game import Piece, Board

# import necessary libraries
import numpy as np
import random
import pickle
from collections import deque

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

class DQN_agent:
    def __init__(self, state_size, action_size, initial_epsilon=1.0):
        # initialize the agent
        
        # state representation from get_state_representation()
        self.state_size = state_size
        # possible actions from get_legal_moves()
        self.action_size = action_size
        # a double-ended queue to store experiences
        self.memory = deque(maxlen=15000)
        # discount rate (determines the importance of future rewards)
        # lower rate makes agent more short-sighted
        # higher rate makes agent value future rewards more significantly (far-sighted)
        self.gamma = 0.95 
        # exploration rate 
        # balances exploration (trying new actions) and exploitation (using the best-known action)
        self.epsilon = initial_epsilon
        # the minimum value that epsilon can reach during the training process
        self.epsilon_min = 0.1
        # rate at which the epsilon value decreases over time
        self.epsilon_decay = 0.999
        # rate at which the weights in the neural network are adjusted during each training iteration
        self.learning_rate = 0.001
        
        # initialize the main model
        self.model = self._build_model()
        # initialize the target model
        self.target_model = self._build_model()

        # step counter for updating target model
        self.update_counter = 0

    def _build_model(self):
        # compiles a neural net for DQ model
        model = Sequential()
        model.add(Dense(512, input_dim=self.state_size, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(2048, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(340, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        # method to update the target model
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        #enables the agent to store experiences and learn from them effectively through experience replay
        self.memory.append((
            state, # current state of the environment before the agent takes an action 
            action, # action taken by the agent in the current state
            reward, # immediate reward received after taking the action
            next_state, # state of the environment after the action is taken.
            done # boolean indicating whether this state-action pair led to the end of an episode
            ))
    
    def act(self, state, legal_moves):
        # deciding which action the agent should take in a given state

        # reshape state to match the model input shape
        state = np.reshape(state, [1, self.state_size])

        # checks if the agent should take a random action based on the current value of exploration rate
        if np.random.rand() <= self.epsilon:
            chosen_action = random.choice(legal_moves)
            # print(f"RA{chosen_action}")
            return chosen_action
        else:
            # use the model to make a prediction
            act_values = self.model.predict(state, verbose=0)[0]
            # set the Q-values of illegal moves to negative infinity
            all_possible_actions = set(range(self.action_size))  # create a set of all possible actions
            illegal_moves = all_possible_actions - set(legal_moves)  # determine illegal moves
            act_values[list(illegal_moves)] = float('-inf')
            chosen_action = np.argmax(act_values)
            # print(f"A{chosen_action}")
            return chosen_action

    # defining experience roleplay function (agent learns from a random sample of past experiences 
    # avoiding the pitfalls of strongly correlated sequential experiences)
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        # Separate the minibatch into states, actions, rewards, next_states, and dones
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        # Reshape states and next_states for batch processing
        states = np.reshape(states, [len(minibatch), self.state_size])
        next_states = np.reshape(next_states, [len(minibatch), self.state_size])

        # Batch prediction for current and next states
        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        # Update Q values for the actions taken
        targets = rewards + self.gamma * np.amax(next_q_values, axis=1) * (1 - dones)
        target_f = current_q_values
        for i in range(len(minibatch)):
            target_f[i][actions[i]] = targets[i]

        # Batch training
        self.model.fit(states, target_f, epochs=1, verbose=0)

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target model periodically
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.update_target_model()
    
    def load(self, name):
        # load the current weights of the neural network model from a file
        self.model.load_weights(name)
        self.target_model.load_weights(name)

    def save(self, name):
        # save the current weights of the neural network model to a file
        self.model.save_weights(name)
    
    def save_memory(self, filename):
        # serialise and save the memory
        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)

    def load_memory(self, filename):
        # deserialise and load the memory
        with open(filename, 'rb') as f:
            self.memory = pickle.load(f)