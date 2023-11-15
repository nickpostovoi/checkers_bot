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
        # rate at which the weights in the neural network are adjusted during each training iteration
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
    
    def act(self, state, legal_moves):
        # deciding which action the agent should take in a given state

        # reshape state to match the model input shape
        state = np.reshape(state, [1, self.state_size])

        # checks if the agent should take a random action based on the current value of exploration rate
        if np.random.rand() <= self.epsilon:
            return random.choice(legal_moves)

        # use the model to make a prediction
        act_values = self.model.predict(state)
        # filter out act_values to only include legal moves
        legal_act_values = act_values[0][legal_moves]
        # return the action with the highest Q-value
        return legal_moves[np.argmax(legal_act_values)]

    # defining experience roleplay function (agent learns from a random sample of past experiences 
    # avoiding the pitfalls of strongly correlated sequential experiences)
    def replay(self, batch_size):
        # randomly sample a minibatch of experiences from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        # iterate through minibatch of experiences and calculate target Q-value for the action taken
        for state, action, reward, next_state, done in minibatch:
            # if the episode is done then target is simply the observed reward
            target = reward
            if not done:
                next_state = np.reshape(next_state, [1, self.state_size])
                # if the episode is not done, the target Q-value is calculated using the Bellman equation
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            state = np.reshape(state, [1, self.state_size])
            # obtain the model prediction for the current state
            target_f = self.model.predict(state)
            # Q-value for the action taken is updated with the calculated target
            target_f[0][action] = target
            # the model is trained (updated) using this new target
            # this training step adjusts the model's weights to better predict the target Q-values in the future
            self.model.fit(state, target_f, epochs=1, verbose=2)
        # check if the exploration rate is greater than a minimum value
        if self.epsilon > self.epsilon_min:
            # decay the exploration rate
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        # save the current weights of the neural network model to a file
        self.model.save_weights(name)