# Q-Learning Agent

This repository contains a Q-learning agent implemented in Python for the [BOB environment](https://medium.datadriveninvestor.com/bowl-of-balls-bob-new-openai-gym-environment-fa4af856f58c). The agent is defined in the file `QLearningAgent.py` and is used to train a model for a simple environment in the file `run_agent.py`.

## Requirements

To run this code, you will need to have the following libraries installed:

- NumPy
- Pandas
- tqdm
- pickle
- json
- matplotlib

## File Structure

- `QLearningAgent.py`: Contains the implementation of the Q-learning agent.
- `run_agent.py`: Contains the code for training the agent and storing the results.

## Usage

To train the Q-learning agent, run the following command:

python run_agent.py


This will run the agent for three different numbers of episodes (100,000, 1,000,000, and 5,000,000) and store the results in the `results` directory. The results will be saved in a pickle file with the name `q_table_<num_episodes>.pkl`.

## Q-Learning Agent

The Q-learning agent is implemented as a class with the following methods:

### `__init__(self, action_space, observation_space)`

This method initializes the action and observation spaces for the agent and creates a Q-table with all zeros. It also sets the learning rate and discount factor for the agent.

### `act(self, state, epsilon=0.1)`

This method selects an action for the given state, with a probability of `epsilon` of selecting a random action to encourage exploration. Otherwise, it selects the action with the highest Q-value.

### `learn(self, state, action, reward, next_state, done)`

This method updates the Q-value for the given state-action pair based on the Q-learning update rule, which takes into account the reward and the maximum Q-value of the next state. If the episode is done, the Q-value is set to the reward. Otherwise, the Q-value is updated using the Q-learning update rule.

## Training

The Q-learning agent is trained using the `run_agent.py` script. The agent is run for a fixed number of episodes and at each step, the agent's action is passed to the environment, which returns the next state, reward, done flag, and additional information. The agent then learns from this experience and updates its Q-values. The results of the training process are stored in a pickle file in the `results` directory.
