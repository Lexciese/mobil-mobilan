#!/usr/bin/env python

import rospy
from shell_simulation.odom_data_processor import OdometryProcessor
from shell_simulation.laserscan_subscriber import Laser_Subs
from tf.transformations import euler_from_quaternion
import shell_simulation.lanes as lanes
import math
import numpy as np

RESOLUTION = 0.1

class WaypointSelection(object):

    def __init__(self, pointer):
        self.pointer = pointer
        self.pointer_previous = 1
        
        # Use only one lane (left lane)
        self.waypoints = lanes.left
        
        self.front_d = 100
        self.finished = False
        
        # Add state attribute for backward compatibility with controller
        self.state = 0  # 0 represents normal driving

        self.odom_data_processor = OdometryProcessor()
        self.laserscan_subscriber = Laser_Subs()

        self.s_min_f = 8.0

        # Adjust curvature detection parameters for earlier detection
        self.curvature_window = 50  # Look much further ahead (approx 10m with 0.2m spacing)
        self.planning_horizon = 100  # Even longer horizon for planning
        self.curvature_threshold = 0.04  # Slightly more sensitive
        self.last_curvature_log = 0
        self.curvature_log_interval = 5
        self.debug_curve = False
        
        # Track upcoming turns for speed planning
        self.upcoming_turn = None  # Will store (distance, curvature, direction)
        self.turn_detected = False
        self.turn_distance_threshold = 15.0  # Detect turns up to 15 meters ahead

    def detect_curvature(self, lookahead=None):
        """
        Detect curvature in the upcoming road segment using the offline trajectory.
        Args:
            lookahead: Optional parameter to specify how far to look ahead
        Returns the estimated curvature, turn direction, and distance to the turn
        """
        waypoint_data = self.waypoints
        
        # Use provided lookahead or default window
        window_size = lookahead if lookahead else self.curvature_window
        
        # Need at least 3 points to calculate curvature
        if self.pointer + window_size >= len(waypoint_data):
            return 0, "straight", float('inf')
        
        # Get a window of points ahead of current position
        points = []
        distances = []  # Track distances from current position to each point
        cumulative_distance = 0
        car_data = self.odom_data_processor.get_car_info()
        car_pos = np.array([car_data[0], car_data[1]])
        
        prev_point = None
        for i in range(window_size):
            idx = self.pointer + i
            if idx < len(waypoint_data):
                point = np.array([waypoint_data[idx][0], waypoint_data[idx][1]])
                points.append(point)
                
                # Calculate distance from car to this point
                if i == 0:
                    # First point distance is direct from car
                    segment_distance = np.linalg.norm(point - car_pos)
                else:
                    # Subsequent points are cumulative along path
                    segment_distance = np.linalg.norm(point - prev_point)
                
                cumulative_distance += segment_distance
                distances.append(cumulative_distance)
                prev_point = point
        
        if len(points) < 3:
            return 0, "straight", float('inf')
            
        # Calculate curvature using multiple points
        curvatures = []
        curvature_distances = []  # Distance from current position to each curvature point
        
        for i in range(len(points) - 2):
            p1, p2, p3 = points[i], points[i+1], points[i+2]
            
            # Method 1: Vector angle
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Get segment lengths
            d1 = np.linalg.norm(v1)
            d2 = np.linalg.norm(v2)
            
            # Skip if points are too close (might be noisy)
            if d1 < 0.01 or d2 < 0.01:
                continue
                
            # Normalize vectors for angle calculation
            v1_norm = v1 / d1
            v2_norm = v2 / d2
            
            # Calculate the angle between these vectors
            dot_product = np.dot(v1_norm, v2_norm)
            # Clamp to prevent numerical errors
            dot_product = max(-1.0, min(1.0, dot_product))
            angle = math.acos(dot_product)
            
            # Cross product for direction
            cross_z = np.cross(v1, v2)
            
            # Curvature = 2*sin(angle) / triangle side length
            if d1 > 0 and d2 > 0:
                segment_length = (d1 + d2) / 2
                curvature = 2 * math.sin(angle/2) / segment_length
                
                # Store the curvature and its distance
                curvatures.append(curvature)
                # Use distance to middle point (p2) as reference
                curvature_distances.append(distances[i+1])
                
                # Debug individual point curvature
                if self.debug_curve and abs(curvature) > self.curvature_threshold/2:
                    rospy.logwarn(f"Point {i} curvature: {curvature:.6f}, angle: {angle:.4f} rad, distance: {distances[i+1]:.2f}m")
        
        if not curvatures:
            return 0, "straight", float('inf')
        
        # Find point of maximum curvature and its distance
        max_curvature_idx = max(range(len(curvatures)), key=lambda i: abs(curvatures[i]))
        max_curvature = curvatures[max_curvature_idx]
        max_curvature_distance = curvature_distances[max_curvature_idx]
        
        # Calculate weighted average curvature around the maximum
        window_start = max(0, max_curvature_idx - 2)
        window_end = min(len(curvatures), max_curvature_idx + 3)
        
        curve_segment = curvatures[window_start:window_end]
        weighted_curvature = sum(curve_segment) / len(curve_segment)
        
        # Determine turn direction from maximum curvature point
        # Use nearby points for more stable direction calculation
        direction_idx = max_curvature_idx
        if direction_idx < len(points) - 2:
            v1 = points[direction_idx+1] - points[direction_idx]
            v2 = points[direction_idx+2] - points[direction_idx+1]
            cross_z = np.cross(v1, v2)
            direction = "right" if cross_z < 0 else "left" if cross_z > 0 else "straight"
        else:
            direction = "straight"
        
        return weighted_curvature, direction, max_curvature_distance

    def find_upcoming_turns(self):
        """
        Scan ahead on the trajectory to find upcoming turns and their distances.
        Updates self.upcoming_turn with the nearest significant turn.
        """
        # Scan with the longer planning horizon
        curvature, direction, distance = self.detect_curvature(self.planning_horizon)
        
        # If significant curvature detected and within threshold distance
        if abs(curvature) > self.curvature_threshold and distance < self.turn_distance_threshold:
            # Store information about the upcoming turn
            self.upcoming_turn = (distance, curvature, direction)
            self.turn_detected = True
            # Log only when we detect a turn or periodically
            if self.pointer % self.curvature_log_interval == 0:
                rospy.logwarn(f"UPCOMING TURN DETECTED: {direction.upper()} turn in {distance:.1f}m with curvature {curvature:.6f}")
            return True
        else:
            self.upcoming_turn = None
            self.turn_detected = False
            return False
            
    def waypoint_selection(self):
        s_front, s_left, s_right, s_front_left, s_front_right = self.laserscan_subscriber.return_s_min_laser()
        
        self.front_d = s_front
        
        # Simple waypoint selection without lane changing
        waypoint_data = self.waypoints
        car_data = self.odom_data_processor.get_car_info()

        # Look for upcoming turns for speed planning
        self.find_upcoming_turns()

        # Always detect and log road curvature every cycle for debugging
        curvature, direction, _ = self.detect_curvature()
        if abs(curvature) > self.curvature_threshold:
            rospy.logwarn(f"CURVE DETECTED: {direction.upper()} turn with curvature {curvature:.6f} at waypoint {self.pointer}")
        elif self.pointer % self.curvature_log_interval == 0:
            # Log straight segments only periodically to reduce noise
            rospy.logwarn(f"ROAD SEGMENT: Straight segment (curvature: {curvature:.6f}) at waypoint {self.pointer}")

        ############################ Indexing #################################################
        
        if self.pointer != self.pointer_previous:
            if self.pointer==0:
                self.x0_y0 = [car_data[0], car_data[1]]
                self.x1_y1 = [waypoint_data[self.pointer][0], waypoint_data[self.pointer][1]]
                self.pointer_previous = self.pointer
                # rospy.logwarn(f"Initial position: {self.x0_y0}, Next waypoint: {self.x1_y1}")
            else:
                self.x0_y0 = [waypoint_data[self.pointer-1][0], waypoint_data[self.pointer-1][1]]
                self.x1_y1 = [waypoint_data[self.pointer][0], waypoint_data[self.pointer][1]]
                # rospy.logwarn(f"Current waypoint: {self.x0_y0}, Next waypoint: {self.x1_y1}")

        try:
            m1 = (self.x1_y1[1] - self.x0_y0[1])/(self.x1_y0[0] - self.x0_y0[0])
            m2 = -1/m1

            c_1 = self.x1_y1[1] - m2*self.x1_y1[0]
            c_0 = self.x0_y0[1] - m2*self.x0_y0[0]
            c_car = car_data[1] - m2*car_data[0]

        except:
            c_1 = self.x1_y1[1]
            c_0 = self.x0_y0[1]
            c_car = car_data[1]

        if ((c_0 < c_1) and (c_car > c_1)) or ((c_0 > c_1) and (c_car < c_1)) or (c_0 == c_1):
            self.pointer = self.pointer+1
            self.x0_y0 = self.x1_y1
            # rospy.logwarn(f"Advancing to waypoint {self.pointer}")

            try:
                self.x1_y1 = [waypoint_data[self.pointer][0], waypoint_data[self.pointer][1]]
                # rospy.logwarn(f"Current waypoint: {self.x0_y0}, Next waypoint: {self.x1_y1}")
            except:
                self.x1_y1 = self.x0_y0
                self.finished = True
                # rospy.logwarn("Reached final waypoint")

    def waypoint_goal(self):
        self.waypoint_selection()
        return self.x0_y0, self.x1_y1, self.front_d

    def get_path_points(self, num_points=100):
        """
        Get a sequence of waypoints for path representation.
        Returns a list of (x,y) tuples representing points on the path.
        """
        points = []
        
        # Add current waypoint points from x0_y0 and x1_y1
        if hasattr(self, 'x0_y0') and hasattr(self, 'x1_y1'):
            points.append((self.x0_y0[0], self.x0_y0[1]))
            points.append((self.x1_y1[0], self.x1_y1[1]))
        
        # Use the single lane waypoints
        waypoint_data = self.waypoints
            
        # Add future waypoints based on current pointer
        max_points = min(num_points, len(waypoint_data) - self.pointer - 1)
        for i in range(1, max_points + 1):
            try:
                next_point_idx = self.pointer + i
                if next_point_idx < len(waypoint_data):
                    points.append((waypoint_data[next_point_idx][0], waypoint_data[next_point_idx][1]))
            except IndexError:
                # Stop adding points if we go out of range
                break
                
        # Ensure we have at least 2 points
        if len(points) < 2:
            # If not enough points, duplicate the last point with a small offset
            if len(points) == 1:
                # Create a point in the direction of vehicle heading
                car_data = self.odom_data_processor.get_car_info()
                heading = car_data[3]  # Assuming this is the yaw angle
                offset_x = math.cos(heading)
                offset_y = math.sin(heading)
                last_point = points[0]
                points.append((last_point[0] + offset_x, last_point[1] + offset_y))
                
        return points

    def get_future_waypoint(self):
        """
        Get a future waypoint beyond the current target for curvature calculation
        Returns (x,y) of a future waypoint or None if not available
        """
        waypoint_data = self.waypoints
            
        # Get a point 2 positions ahead of current pointer
        future_idx = self.pointer + 2
        if future_idx < len(waypoint_data):
            return (waypoint_data[future_idx][0], waypoint_data[future_idx][1])
        return None

    def get_upcoming_turn_info(self):
        """
        Returns information about the nearest upcoming turn.
        Returns: (distance, curvature, direction) or None if no turn detected
        """
        return self.upcoming_turn