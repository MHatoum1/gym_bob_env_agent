import numpy as np
import random

# Create the Q-learning agent
class QLearningAgent:
    def __init__(self, action_space, observation_space):
        # Initialize the action and observation spaces
        self.action_space = action_space
        self.observation_space = observation_space
        
        # Initialize the Q-table with all zeros
        #self.Q = np.zeros((observation_space.shape[0], action_space.n))
        self.Q = np.zeros((50, 50, 3))
        
        # Set the learning rate and discount factor
        self.alpha = 0.1
        self.gamma = 0.9

    def act(self, state, epsilon=0.1):
        ball1, ball2, diff = state

        # Select a random action with probability epsilon, otherwise select the action with the highest Q-value
        if random.uniform(0, 1) < epsilon:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.Q[ball1-1][ball2-1])
        return action
    
    def learn(self, state, action, reward, next_state, done):
        ball1, ball2, diff = state
        next_ball1, next_ball2, next_diff = next_state
        
        # Update the Q-value for the state-action pair using the Q-learning update rule
        if done:
            self.Q[ball1-1][ball2-1][action] = reward
        else:
            self.Q[ball1-1][ball2-1][action] = (1 - self.alpha) * self.Q[ball1-1][ball2-1][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_ball1-1][next_ball2-1]))
