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

        # Adjust curvature detection parameters for higher sensitivity
        self.curvature_window = 10  # Smaller window to detect more localized curves
        self.curvature_threshold = 0.05  # Lower threshold to detect subtle curves
        self.last_curvature_log = 0
        self.curvature_log_interval = 5  # More frequent logging
        self.debug_curve = False  # Enable detailed curve debugging

        # Adjust for much earlier curve detection
        self.early_detection_distance = 50.0  # Look up to 20m ahead
        self.early_detection_threshold = 0.03  # More sensitive for early detection
        self.always_detect_ahead = True  # Always look ahead, not just periodically

        # Track upcoming turns for cruise control
        self.upcoming_turn_info = None  # Will store (distance, curvature, direction)
        self.turn_speed_reduction_factor = 0.6  # How much to reduce speed for turns

    def detect_curvature(self):
        """
        Detect curvature in the upcoming road segment using the offline trajectory.
        Returns the estimated curvature and turn direction.
        """
        waypoint_data = self.waypoints
        
        # Need at least 3 points to calculate curvature
        if self.pointer + self.curvature_window >= len(waypoint_data):
            return 0, "straight"
        
        # Get a window of points ahead of current position
        points = []
        for i in range(self.curvature_window):
            idx = self.pointer + i
            if idx < len(waypoint_data):
                points.append(np.array([waypoint_data[idx][0], waypoint_data[idx][1]]))
        
        if len(points) < 3:
            return 0, "straight"
            
        # Calculate curvature using multiple points - print first few points for debugging
        if self.debug_curve:
            rospy.logwarn(f"Analyzing points near waypoint {self.pointer}: {points[0]}, {points[1]}, {points[2]}")
        
        # Alternative curvature calculation using circle fitting
        # This can be more stable than angle-based calculation
        curvatures = []
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
            
            # Method 2: Menger curvature (uses cross product)
            cross_z = np.cross(v1, v2)
            
            # Curvature = 2*sin(angle) / triangle side length
            # More pronounced for sharper turns
            if d1 > 0 and d2 > 0:
                segment_length = (d1 + d2) / 2
                curvature = 2 * math.sin(angle/2) / segment_length
                curvatures.append(curvature)
                
                # Debug individual point curvature
                if self.debug_curve and abs(curvature) > self.curvature_threshold/2:
                    rospy.logwarn(f"Point {i} curvature: {curvature:.6f}, angle: {angle:.4f} rad")
        
        if not curvatures:
            return 0, "straight"
            
        # Average curvature, weighted more heavily toward maximum values
        avg_curvature = sum(curvatures) / len(curvatures)
        max_curvature = max(curvatures, key=abs)
        weighted_curvature = (avg_curvature + max_curvature) / 2
        
        # For debug, log both average and maximum
        if self.debug_curve:
            rospy.logwarn(f"Avg curvature: {avg_curvature:.6f}, Max: {max_curvature:.6f}, Weighted: {weighted_curvature:.6f}")
            
        # Determine turn direction using cross product of first set of vectors
        # Take the direction from the point with highest curvature for accuracy
        max_idx = curvatures.index(max(curvatures, key=abs))
        v1 = points[max_idx+1] - points[max_idx]
        v2 = points[max_idx+2] - points[max_idx+1]
        cross_z = np.cross(v1, v2)
        direction = "right" if cross_z < 0 else "left" if cross_z > 0 else "straight"
        
        return weighted_curvature, direction

    def detect_future_turns(self, look_ahead_distance=20.0):
        """
        Detect turns much further ahead on the path (10-20 meters ahead)
        Returns True if a significant turn is detected
        """
        waypoint_data = self.waypoints
        car_data = self.odom_data_processor.get_car_info()
        car_pos = np.array([car_data[0], car_data[1]])
        
        # Calculate how many waypoints correspond to our look-ahead distance
        # Better estimate based on actual waypoint spacing
        points_near = min(10, len(waypoint_data) - self.pointer)
        if points_near < 2:
            return False
            
        # Calculate average waypoint spacing from actual data
        spacing_samples = []
        prev_point = np.array([waypoint_data[self.pointer][0], waypoint_data[self.pointer][1]])
        for i in range(1, points_near):
            point = np.array([waypoint_data[self.pointer + i][0], waypoint_data[self.pointer + i][1]])
            spacing_samples.append(np.linalg.norm(point - prev_point))
            prev_point = point
            
        avg_spacing = sum(spacing_samples) / len(spacing_samples) if spacing_samples else 0.2
        points_to_check = int(look_ahead_distance / avg_spacing) + 10  # Add buffer
        
        # Make sure we don't go beyond available waypoints
        if self.pointer + points_to_check >= len(waypoint_data):
            points_to_check = len(waypoint_data) - self.pointer - 1
            
        if points_to_check < 3:
            return False
        
        # Create distance-based ranges for analysis
        # We'll analyze points at different distances separately
        distance_ranges = [
            (5.0, 10.0),   # Near range
            (10.0, 15.0),  # Mid range
            (15.0, 20.0)   # Far range
        ]
        
        # Sample points along the path at different distances
        all_future_points = []
        cumulative_distance = 0.0
        distances = []
        
        # Start from current position and sample points up to max distance
        prev_point = car_pos
        for i in range(points_to_check):
            idx = self.pointer + i
            if idx >= len(waypoint_data):
                break
                
            point = np.array([waypoint_data[idx][0], waypoint_data[idx][1]])
            segment_distance = np.linalg.norm(point - prev_point)
            cumulative_distance += segment_distance
            
            # Store point with its distance
            all_future_points.append(point)
            distances.append(cumulative_distance)
            
            prev_point = point
            
            # Stop after reaching our maximum look-ahead
            if cumulative_distance > look_ahead_distance:
                break
                
        # If we don't have enough points to analyze
        if len(all_future_points) < 3:
            return False
            
        # For very early detection, analyze each distance range separately
        for min_dist, max_dist in distance_ranges:
            # Extract points for this range
            range_points = []
            
            for i, point in enumerate(all_future_points):
                if min_dist <= distances[i] <= max_dist:
                    range_points.append(point)
            
            # Skip if not enough points in this range
            if len(range_points) < 3:
                continue
                
            # Calculate curvatures within this range
            curvatures = []
            
            for i in range(len(range_points) - 2):
                p1, p2, p3 = range_points[i], range_points[i+1], range_points[i+2]
                
                # Calculate vectors between consecutive points
                v1 = p2 - p1
                v2 = p3 - p2
                
                # Get segment lengths
                d1 = np.linalg.norm(v1)
                d2 = np.linalg.norm(v2)
                
                if d1 < 0.01 or d2 < 0.01:
                    continue
                    
                # Calculate angle between vectors
                dot_product = np.dot(v1, v2) / (d1 * d2)
                dot_product = max(-1.0, min(1.0, dot_product))
                angle = math.acos(dot_product)
                
                # Get the cross product for direction
                cross_z = np.cross(v1, v2)
                direction = "right" if cross_z < 0 else "left"
                
                # Calculate curvature
                segment_length = (d1 + d2) / 2
                curvature = 2 * math.sin(angle/2) / segment_length
                
                # Use a lower threshold for early detection
                threshold = self.early_detection_threshold
                
                if abs(curvature) > threshold:
                    # Find actual distance to this curve by getting the middle point's distance
                    # Fix: Replace problematic array comparison with proper NumPy array comparison
                    distance_idx = -1
                    for idx, point in enumerate(all_future_points):
                        if np.array_equal(point, p2):
                            distance_idx = idx
                            break
                            
                    if distance_idx >= 0:
                        curve_distance = distances[distance_idx]
                    else:
                        curve_distance = (min_dist + max_dist) / 2
                        
                    # Store the upcoming turn information for speed planning
                    self.upcoming_turn_info = (curve_distance, curvature, direction)
                    
                    # Cruising info - how much we should reduce speed based on curvature
                    speed_factor = min(1.0, abs(curvature) / 0.1) * 100  # Percentage of speed reduction
                    # rospy.logwarn(f"CRUISE PLANNING: {direction.upper()} turn in {curve_distance:.1f}m requires {speed_factor:.0f}% speed reduction")
                    
                    rospy.logwarn(f"EARLY TURN DETECTION: {direction.upper()} turn detected {curve_distance:.1f}m ahead with curvature {curvature:.4f}")
                    return True
        
        # No turn detected
        self.upcoming_turn_info = None
        return False

    def waypoint_selection(self):
        s_front, s_left, s_right, s_front_left, s_front_right = self.laserscan_subscriber.return_s_min_laser()
        
        self.front_d = s_front
        
        # Simple waypoint selection without lane changing
        waypoint_data = self.waypoints
        car_data = self.odom_data_processor.get_car_info()

        # Always detect future turns, not just periodically
        if self.always_detect_ahead:
            self.detect_future_turns(self.early_detection_distance)

        # Detect immediate curve
        curvature, direction = self.detect_curvature()
        if abs(curvature) > self.curvature_threshold:
            rospy.logwarn(f"CURVE DETECTED: {direction.upper()} turn with curvature {curvature:.6f} at waypoint {self.pointer}")
        elif self.pointer % self.curvature_log_interval == 0:
            # Look for turns further ahead for advance warning - now redundant with always_detect_ahead
            # but kept for compatibility
            if not self.always_detect_ahead:
                self.detect_future_turns(look_ahead_distance=self.early_detection_distance)
            
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
            m1 = (self.x1_y1[1] - self.x0_y0[1])/(self.x1_y1[0] - self.x0_y0[0])
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
        Returns information about any upcoming turn for speed planning.
        Returns: (distance, curvature, direction) or None if no turn is detected
        """
        return self.upcoming_turn_info