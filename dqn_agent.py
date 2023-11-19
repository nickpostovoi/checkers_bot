# import the game environment
# from checkers_game import Piece, Board

# import necessary libraries
import numpy as np
import random
import pickle
from collections import deque

# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         # memory growth must be set before GPUs have been initialized
#         print(e)

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
        self.memory = deque()
        # discount rate (determines the importance of future rewards)
        # lower rate makes agent more short-sighted
        # higher rate makes agent value future rewards more significantly (far-sighted)
        self.gamma = 0.95 
        # exploration rate 
        # balances exploration (trying new actions) and exploitation (using the best-known action)
        self.epsilon = initial_epsilon
        # the minimum value that epsilon can reach during the training process
        self.epsilon_min = 0.3
        # rate at which the epsilon value decreases over time
        self.epsilon_decay = 0.995
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
        if len(self.memory) % 100 == 0:  
            # print every 100 experiences
            print("Memory Buffer Size:", len(self.memory))
    
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
        # print('Replay is triggered')
        # randomly sample a minibatch of experiences from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        # iterate through minibatch of experiences and calculate target Q-value for the action taken
        for state, action, reward, next_state, done in minibatch:
            # if the episode is done then target is simply the observed reward
            target = reward
            if not done:
                next_state = np.reshape(next_state, [1, self.state_size])
                # if the episode is not done, the target Q-value is calculated using the Bellman equation
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            state = np.reshape(state, [1, self.state_size])
            # obtain the model prediction for the current state
            target_f = self.model.predict(state, verbose=0)
            # Q-value for the action taken is updated with the calculated target
            target_f[0][action] = target
            # the model is trained (updated) using this new target
            # this training step adjusts the model's weights to better predict the target Q-values in the future
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # check if the exploration rate is greater than a minimum value
        if self.epsilon > self.epsilon_min:
            # decay the exploration rate
            self.epsilon *= self.epsilon_decay
            print("Current exploration rate (epsilon):", self.epsilon)
        
        # increment the step counter and update the target model if needed
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