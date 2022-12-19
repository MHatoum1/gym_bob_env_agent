import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np

# Save Q-table to a file
FILE_NAME_100000 = f'results/q_table_100000'
FILE_NAME_1000000 = f'results/q_table_1000000'
FILE_NAME_5000000 = f'results/q_table_5000000'

# Open the file in binary mode and extract the data
with open(f'{FILE_NAME_100000}.pkl', 'rb') as file:
    q_table_1 = pickle.load(file)
    data_1 = q_table_1[:, :, 3]

# Open the file in binary mode and extract the data
with open(f'{FILE_NAME_1000000}.pkl', 'rb') as file:
    q_table_2 = pickle.load(file)
    data_2 = q_table_2[:, :, 3]

# Open the file in binary mode and extract the data
with open(f'{FILE_NAME_5000000}.pkl', 'rb') as file:
    q_table_3 = pickle.load(file)
    data_3 = q_table_3[:, :, 3]

# Create a figure with 3 subplots
fig, ax = plt.subplots(1, 3)

# Plot Q-values for each action
im1 = ax[0].imshow(data_1, cmap='jet')
im2 = ax[1].imshow(data_2, cmap='jet')
im3 = ax[2].imshow(data_3, cmap='jet')

# Add titles 
ax[0].set_title("Best Action After 100K Episode")
ax[1].set_title("Best Action After 1M Episode")
ax[2].set_title("Best Action After 5M Episode")

# Add axis labels
ax[0].set_xlabel('First Ball')
ax[0].set_ylabel('Second Ball')
ax[1].set_xlabel('First Ball')
ax[1].set_ylabel('Second Ball')
ax[2].set_xlabel('First Ball')
ax[2].set_ylabel('Second Ball')

# Create an axes divider for each subplot
#divider1 = make_axes_locatable(ax[0])
#divider2 = make_axes_locatable(ax[1])
divider3 = make_axes_locatable(ax[2])

# Create an axis for the colorbar on the right side of each subplot
#cax1 = divider1.append_axes("right", size="5%", pad=0.05)
#cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cax3 = divider3.append_axes("right", size="5%", pad=0.05)

# Add a colorbar to each subplot
#cb1 = fig.colorbar(im1, cax=cax1)
#cb2 = fig.colorbar(im2, cax=cax2)
cb3 = fig.colorbar(im3, cax=cax3)




# Extract the fourth column of the array
values_1 = data_1
# Find the unique values in the column and their counts
unique_values_1, counts_1 = np.unique(values_1, return_counts=True)


# Extract the fourth column of the array
values_2 = data_2
# Find the unique values in the column and their counts
unique_values_2, counts_2 = np.unique(values_2, return_counts=True)


# Extract the fourth column of the array
values_3 = data_3
# Find the unique values in the column and their counts
unique_values_3, counts_3 = np.unique(values_3, return_counts=True)

# Set the figure size
plt.figure(figsize=(10, 3))

# Set the bar width
bar_width = 0.25

# Set the positions of the bars
positions_1 = np.arange(len(unique_values_1))
positions_2 = positions_1 + bar_width
positions_3 = positions_2 + bar_width

# Create the bars
plt.bar(positions_1, counts_1, bar_width, label='After 100K Episode')
plt.bar(positions_2, counts_2, bar_width, label='After 1M Episode')
plt.bar(positions_3, counts_3, bar_width, label='After 5M Episode')

# Add the labels for the x-axis
plt.xticks(positions_1 + bar_width, unique_values_1)

# Add a legend
plt.legend()

# Show the plot
plt.show()
