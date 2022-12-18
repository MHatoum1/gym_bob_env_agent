
from tqdm import tqdm
import pickle

import pandas as pd
import json

import numpy as np

'''
In this example, the Q-learning agent is implemented as a class that has a Q-table to store the Q-values for each state-action pair. 
The agent's act method selects an action based on the Q-values of the current state, 
with some probability of selecting a random action to encourage exploration. 
The agent's learn method updates the Q-values based on the Q-learning update rule, 
which takes into account the reward and the maximum Q-value of the next state.

The agent is run for a fixed number of episodes, and at each step, the agent's action is passed


'''



env = BOBEnv()
# Initialize the agent
agent = QLearningAgent(env.action_space, env.observation_space)
EPISODES = [100000,1000000,5000000]

for num_episodes in EPISODES:

	FILE_NAME = f'results/q_table_{num_episodes}'


	# Run the episodes
	for episode in tqdm(range(num_episodes)):
		# Reset the environment
		state = env.reset()
		done = False

		# Run the episode
		while not done:
			# Render the environment
			#env.render()
			
			# Get the action from the agent
			action = agent.act(state)
			
			# Take a step in the environment
			next_state, reward, done, info = env.step(action)
			
			# Learn from the experience
			agent.learn(state, action, reward, next_state, done)
			
			# Update the state
			state = next_state

	# Close the environment
	env.close()

	# Convert agent_result to a numpy array
	agent_result = np.array(agent.Q)

	# Find the index of the best action (0, 1, or 2) for each state by finding the maximum value among the three actions
	best_actions = np.argmax(agent_result, axis=2)

	# Reshape the array to have shape (50, 50, 1)
	best_actions = best_actions.reshape((50, 50, 1))

	# Concatenate the best actions array with the original array to create a new array with shape (50, 50, 1)
	agent_result = np.concatenate((agent_result, best_actions), axis=2)

	# Save Q-table to a CSV file
	df = pd.DataFrame(agent_result.reshape(-1, 4), columns=['discard min ball', 'discard max ball', 'stand','best move'])
	df.to_csv(f"{FILE_NAME}.csv", index=False)

	# Save Q-table to a file
	with open(f"{FILE_NAME}.pkl", "wb") as f:
		pickle.dump(agent_result, f)

	# Save Q-table to a JSON file
	with open(f"{FILE_NAME}.json", "w") as f:
		json.dump(agent_result.tolist(), f)

	# Save Q-table to a file
	with open(f"{FILE_NAME}.txt", "w") as f:
		for i in range(50):
			for j in range(50):
				f.write(f"{agent_result[i][j][3]}\n")

