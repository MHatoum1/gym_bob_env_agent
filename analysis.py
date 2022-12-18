
import matplotlib.pyplot as plt
# Save Q-table to a file
import pickle
import numpy as np

EPISODES = [100000,1000000,5000000]
for num_episodes in EPISODES:
	FILE_NAME = f'results/q_table_{num_episodes}'

	# Open the file in binary mode
	with open(f'{FILE_NAME}.pkl', 'rb') as file:
		# Load the Q-table from the file
		agent_result = pickle.load(file)


	# Extract the fourth column of the array
	values = agent_result[:,:,3]

	# Find the unique values in the column and their counts
	unique_values, counts = np.unique(values, return_counts=True)

	# Print the results
	print(f'Results after {num_episodes} episodes')
	print(f'Unique values: {unique_values}')
	print(f'Counts: {counts}')

	# Extract the best action for each state from the array
	best_actions = agent_result[:,:,3]

	# Create the heatmap using the 'jet' colormap
	plt.imshow(best_actions, cmap='jet')

	# Add a colorbar
	plt.colorbar()

	# Add a title
	plt.title(f'Best Action for Each State {num_episodes} episodes')
	plt.xlabel('First Ball')
	plt.ylabel('Second Ball')

	# Show the plot
	plt.show()




