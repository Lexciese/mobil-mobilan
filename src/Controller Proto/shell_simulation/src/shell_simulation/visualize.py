import carla
import sys
import os

# Import the lanes module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from shell_simulation.lanes import left  # Import the left array from lanes.py

# Connect to the CARLA server running on localhost
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)  # seconds
world = client.get_world()

start_coords = [280.363739,133.306351,0.001746] 
other_coords = [
    [334.949799,161.106171,0.001736],
    # ... (other coordinates)
]

# Visualize start and other coordinates on the CARLA map
start_location = carla.Location(x=start_coords[0], y=-start_coords[1], z=start_coords[2])
world.debug.draw_point(start_location, size=0.1, color=carla.Color(r=0, g=255, b=0), life_time=120.0)

for coord in other_coords:
    location = carla.Location(x=coord[0], y=-coord[1], z=coord[2])
    world.debug.draw_point(location, size=0.1, color=carla.Color(r=0, g=0, b=255), life_time=120.0)

# Visualize the left lane trajectory from lanes.py
for i in range(len(left) - 1):
    start_point = carla.Location(x=left[i][0], y=-left[i][1], z=0.001)
    end_point = carla.Location(x=left[i+1][0], y=-left[i+1][1], z=0.001)
    world.debug.draw_point(start_point, size=0.05, color=carla.Color(r=255, g=0, b=0), life_time=120.0)
    world.debug.draw_line(start_point, end_point, thickness=0.1, color=carla.Color(r=255, g=0, b=0), life_time=120.0)

print("Points and trajectory have been drawn in the CARLA world. They will remain visible for 120 seconds.")