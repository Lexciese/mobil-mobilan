import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import lanes as lanes

# Assuming lanes.py contains a variable 'left' that is a list of [x, y, z] coordinates
# from lanes import left

# If you don't have the file, here's how you'd plot such data:
# Example data (replace with your actual data)
left = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]

# Convert list of lists to numpy arrays for easier manipulation
left_array = np.array(lanes.left)
x = left_array[:, 0]
y = left_array[:, 1]
z = left_array[:, 2]

# Create 2D plot (x-y plane)
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the lane points in 2D (x-y view)
ax.scatter(x, y, c='blue', marker='o', label='Left Lane')
ax.plot(x, y, 'b-')  # Connect points with a line

# Add labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Lane Plot (2D View)')
ax.grid(True)
ax.legend()

plt.show()