#!/usr/bin/env python

import rospy
from shell_simulation.odom_data_processor import OdometryProcessor
from shell_simulation.waypoint_selection import WaypointSelection
import numpy as np
import math

IS_SUBMISSION = True

if IS_SUBMISSION:
    from std_msgs.msg import Float64, String #SUBMIT CARLA
else:
    from carla_msgs.msg import CarlaEgoVehicleControl #RUNING CARLA

class Controller_Carla(object):

    def __init__(self, pointer):
        #Subscribere
        self.odom_data_processor = OdometryProcessor()
        self.waypoint_selection = WaypointSelection(pointer)

        self.d_aman = 10
        # Desired velocity - convert from km/h to m/s
        KMH_TO_MS = 1/3.6  # Conversion factor from km/h to m/s
        self.v_desired = 17.625 * KMH_TO_MS  # 17.625 km/h ≈ 4.9 m/s
        self.init_time = None
        self.distance = 0.0
        self.last_pose = None

        # Longitudinal Control Parameters
        self.kp = 0.513
        self.ki = 0.0
        self.kd = 0.778

        # Pure Pursuit Parameters - Updated for tighter turns
        self.k_gain = 1.0  # Increased controller gain for more aggressive steering
        self.min_look_ahead = 0.8  # Reduced minimum look-ahead distance
        self.max_look_ahead = 2.5  # Reduced maximum look-ahead distance
        self.ld_gain = 0.25  # Adjusted look-ahead gain
        self.prev_steering = 0.0  # For steering filtering
        self.steering_filter = 0.3  # Reduced filtering for more responsive steering
        self.path_curvature = 0.0  # Track estimated path curvature

        #Publisher
        if IS_SUBMISSION:
            self.pub_throttle = rospy.Publisher("/throttle_command", Float64, queue_size=10)
            self.pub_steer = rospy.Publisher("/steering_command", Float64, queue_size=10)
            self.pub_brake = rospy.Publisher("/brake_command", Float64, queue_size=10)
            self.pub_gear = rospy.Publisher("/gear_command", String, queue_size=10)
        else:
            self.pub = rospy.Publisher("/carla/ego_vehicle/vehicle_control_cmd", CarlaEgoVehicleControl, queue_size=10)

    def publish_controller(self, throttle_value, steer_value, brake_value):
        if IS_SUBMISSION:
            self.pub_gear.publish(String("forward"))
            self.pub_throttle.publish(Float64(throttle_value))
            self.pub_steer.publish(Float64(steer_value))
            self.pub_brake.publish(Float64(brake_value))
        else:
            cmd = CarlaEgoVehicleControl()
            cmd.throttle = throttle_value
            cmd.steer = steer_value
            cmd.brake = brake_value
            self.pub.publish(cmd)

    def normalize_angle(self, angle):
        self.angle = angle
        while self.angle > np.pi:
            self.angle -= 2.0 * np.pi

        while self.angle < -np.pi:
            self.angle += 2.0 * np.pi

        return self.angle

    def calculate_pid(self, error, previous_error, integral, kp, ki, kd):
        """Calculate PID controller output"""
        # Update integral term
        integral += error
        integral = np.clip(integral, -50.0, 50.0)
        
        # Calculate derivative term
        derivative = error - previous_error
        
        # Calculate control output
        control_output = kp * error + ki * integral + kd * derivative
        
        return control_output, integral

    def apply_throttle_brake(self, control_signal, throttle_previous, brake_previous, finish_gliding=False):
        """Apply throttle or brake based on control signal"""
        if finish_gliding:
            return 0.0, 0.0
            
        if control_signal > 0:
            # Apply throttle
            throttle = np.tanh(control_signal)
            throttle = max(0.0, min(1.0, throttle))
            # Limit throttle increase rate
            if throttle - throttle_previous > 0.05:
                throttle = throttle_previous + 0.05
            brake = 0.0
        else:
            # Apply brake
            brake = np.tanh(abs(control_signal))
            brake = max(0.0, min(1.0, brake))
            throttle = 0.0 if throttle_previous < 0.2 else throttle_previous - 0.2
            
        return throttle, brake

    def estimate_path_curvature(self, path_points, num_points=5):
        """Estimate the curvature of the upcoming path"""
        if len(path_points) < 3:
            return 0.0
            
        # Use only a few points ahead for local curvature
        points_to_use = min(num_points, len(path_points))
        points = path_points[:points_to_use]
        
        # Calculate curvature using circumscribed circle method
        max_curvature = 0.0
        
        for i in range(len(points)-2):
            p1 = np.array(points[i])
            p2 = np.array(points[i+1])
            p3 = np.array(points[i+2])
            
            # Vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Skip if points are too close (avoid numerical issues)
            if np.linalg.norm(v1) < 0.01 or np.linalg.norm(v2) < 0.01:
                continue
                
            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
            angle = np.arccos(cos_angle)
            
            # Estimate curvature as 1/radius (radius ≈ segment_length / angle)
            # Larger angle = sharper curve = higher curvature
            avg_segment_length = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2
            if angle > 0.01:  # Avoid division by small angles
                curvature = angle / avg_segment_length
                max_curvature = max(max_curvature, curvature)
                
        return max_curvature

    def get_target_point(self, xc, yc, path_points, ld):
        """
        Find the target point for pure pursuit controller.
        
        Args:
            xc, yc: Current vehicle position
            path_points: List of path points [(x0,y0), (x1,y1), ...]
            ld: Look-ahead distance
            
        Returns:
            target_point: (x,y) coordinates of target point
        """
        # Initialize with the furthest path point as default target
        target_point = path_points[-1]
        min_dist_diff = float('inf')
        
        # Convert current position to numpy array
        current_pos = np.array([xc, yc])
        
        # First, find the closest point on the path to current position
        min_dist = float('inf')
        closest_idx = 0
        
        for i, point in enumerate(path_points):
            dist = np.linalg.norm(np.array(point) - current_pos)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Iterate from closest point forward to find intersection with look-ahead circle
        for i in range(closest_idx, len(path_points) - 1):
            # Get segment points
            p1 = np.array(path_points[i])
            p2 = np.array(path_points[i+1])
            
            # Vector from p1 to p2
            v = p2 - p1
            # Vector from p1 to current position
            w = current_pos - p1
            
            # Calculate segment length
            seg_len = np.linalg.norm(v)
            if seg_len < 0.001:  # Skip very small segments
                continue
                
            # Normalize v
            v_norm = v / seg_len
            
            # Projection of w onto v
            proj = np.dot(w, v_norm)
            
            # Closest point on segment to current position
            closest = p1 + max(0, min(seg_len, proj)) * v_norm
            
            # Distance from current position to closest point
            closest_dist = np.linalg.norm(current_pos - closest)
            
            # If we're further than ld from the path, skip to next segment
            if closest_dist > 1.5 * ld:
                continue
                
            # Solve for intersection(s) of segment with circle of radius ld
            # Based on quadratic formula for line-circle intersection
            a = np.dot(v, v)
            b = 2 * np.dot(w, v)
            c = np.dot(w, w) - ld * ld
            
            discriminant = b * b - 4 * a * c
            
            if discriminant < 0:  # No intersection
                continue
                
            # Find the two possible intersections
            t1 = (-b + np.sqrt(discriminant)) / (2 * a)
            t2 = (-b - np.sqrt(discriminant)) / (2 * a)
            
            # We want the intersection that's further along the path
            t = max(t1, t2)
            
            # If intersection is on this segment
            if 0 <= t <= 1:
                intersection = p1 + t * v
                
                # Calculate how close this is to our desired look-ahead
                dist_diff = abs(np.linalg.norm(intersection - current_pos) - ld)
                
                # If this is closer to desired look-ahead than previous best
                if dist_diff < min_dist_diff:
                    min_dist_diff = dist_diff
                    target_point = intersection
                    
        return target_point

    def check_car_info(self):
        r = rospy.Rate(60)

        #Initial value
        t_previous = rospy.get_time()
        t_initial = rospy.get_time()
        
        int_val_v = 0.0
        int_val_d = 0.0
        last_error_v = 0.0
        last_error_d = 0.0
        throttle_previous = 0.0
        brake_previous = 0.0

        while not rospy.is_shutdown():
            car_data = self.odom_data_processor.get_car_info()
            x0_y0, x1_y1, front_d = self.waypoint_selection.waypoint_goal()

            #Finish Gliding
            last_point = [-232.60,28.10]
            last_point_distance = np.linalg.norm(np.array(car_data[0:2]) - np.array(last_point))
            finish_gliding = (last_point_distance  <= 13.65)
            finish = (last_point_distance <= 3)


            isState2 = self.waypoint_selection.state == 2

            v_car = car_data[2]
            yaw_car = car_data[3]

            isMaintain = False

            if self.init_time is None:
                self.init_time = rospy.get_time()

            if self.last_pose is None:
                self.last_pose = np.array([car_data[0],car_data[1],car_data[-1]])
            else:
                position_now = np.array([car_data[0],car_data[1],car_data[-1]])
                self.distance += np.linalg.norm(position_now-self.last_pose)
                self.last_pose = position_now

            t_now = rospy.get_time()

            #########################   Get Time   #############################
            st =  t_now - t_previous
            t_previous = t_now

            
            #########################   Lateral  #############################

            #Previous Goal
            x0 = x0_y0[0]
            y0 = x0_y0[1]

            #Next Goal
            x1 = x1_y1[0]
            y1 = x1_y1[1]

            #Car's Position
            xc = car_data[0]
            yc = car_data[1]

            # Get additional waypoints for better path representation
            path_points = self.waypoint_selection.get_path_points()
            if not path_points or len(path_points) < 2:
                # Default to using the two waypoints we have
                path_points = [(x0, y0), (x1, y1)]
                
            # Calculate path direction for reference
            path_vector = np.array([x1 - x0, y1 - y0])
            path_length = np.linalg.norm(path_vector)
            
            if path_length > 0:
                # Normalize path vector
                path_vector = path_vector / path_length
                yaw_path = self.normalize_angle(np.arctan2(path_vector[1], path_vector[0]))
            else:
                yaw_path = self.normalize_angle(np.arctan2(y1 - y0, x1 - x0))
                
            # Calculate heading error (used for calculating steering direction)
            theta_e = self.normalize_angle(yaw_path - yaw_car)
            
            # Estimate path curvature for adjusting look-ahead distance
            self.path_curvature = self.estimate_path_curvature(path_points)
            
            # Adjust look-ahead distance based on velocity AND curvature
            # Reduce look-ahead on sharp curves for tighter turning
            curvature_factor = max(0.6, 1.0 - self.path_curvature * 2.0)
            look_ahead_dist = self.min_look_ahead + self.ld_gain * v_car
            look_ahead_dist = look_ahead_dist * curvature_factor  # Reduce on curves
            look_ahead_dist = min(self.max_look_ahead, max(self.min_look_ahead, look_ahead_dist))
            
            # Find target point using pure pursuit
            target_point = self.get_target_point(xc, yc, path_points, look_ahead_dist)
            
            # Convert to car's reference frame
            dx = target_point[0] - xc
            dy = target_point[1] - yc
            
            # Transform the target point from global coordinates to vehicle coordinates
            target_x_veh = dx * np.cos(yaw_car) + dy * np.sin(yaw_car)
            target_y_veh = -dx * np.sin(yaw_car) + dy * np.cos(yaw_car)
            
            # Pure pursuit formula - calculate curvature
            # This formula works when target is in vehicle's reference frame
            if abs(target_x_veh) < 0.01:  # Avoid division by zero
                curvature = 0.0
            else:
                curvature = 2 * target_y_veh / (target_x_veh * target_x_veh + target_y_veh * target_y_veh)
            
            # Calculate steering angle from curvature with dynamic gain
            # Higher gain for sharper turns
            dynamic_gain = self.k_gain * (1.0 + 0.5 * abs(curvature))
            steering_angle = dynamic_gain * curvature
            
            # Ensure we're steering in the same general direction as the path
            # by considering the sign of the heading error
            if abs(steering_angle) < 0.1 and abs(theta_e) > 0.1:
                steering_direction = 1 if theta_e > 0 else -1
                steering_angle = max(0.1, abs(steering_angle)) * steering_direction
            
            # Low-pass filter the steering to avoid jerky motions
            # Use less filtering for high curvature turns
            filter_coef = self.steering_filter * (1.0 - 0.3 * abs(curvature))
            self.prev_steering = filter_coef * steering_angle + (1 - filter_coef) * self.prev_steering
            
            # Clip to valid steering range
            if not finish_gliding:
                steer_output = np.clip(self.prev_steering, -1.0, 1.0)
            else:
                steer_output = 0.0

            ######################### Longitudinal ############################
            isWarning = front_d < 8.0

            # Calculate appropriate speed based on path curvature
            # Slow down for tight turns (prefer coasting over braking)
            curvature_speed_factor = 1.0 - min(0.7, self.path_curvature * 3.0)
            target_speed = self.v_desired * curvature_speed_factor
            
            # Determine if we need to slow down for a curve
            needs_slowdown = self.path_curvature > 0.1 and v_car > target_speed

            if (isState2 and v_car <= self.v_desired):
                isMaintain = True
            elif (isState2 and v_car > self.v_desired):
                isMaintain = False

            # Case 1: Maintain distance to front vehicle
            if isMaintain:
                # Calculate distance error
                if front_d == float('inf'):
                    d_error = 10 - self.d_aman
                else:
                    d_error = front_d - self.d_aman
                
                # Apply PID control for distance maintenance
                control_signal, int_val_d = self.calculate_pid(
                    d_error, last_error_d, int_val_d, self.kp, self.ki, self.kd
                )
                
                rospy.logwarn(f"rst = {control_signal}, front_d = {front_d}")
                
                # Apply throttle or brake
                throttle_output, brake_output = self.apply_throttle_brake(
                    control_signal, throttle_previous, brake_previous, finish_gliding
                )
                
                last_error_d = d_error
            
            # Case 2: Warning - vehicle too close
            elif isWarning:
                # Determine target speed based on front distance
                if front_d < 6.0:
                    print("Stop, car ahead")
                    v_desired = 0.0
                else:
                    v_desired = 4.0
                
                # Calculate velocity error
                delta_v = v_desired - v_car
                
                # Apply PID control for speed adjustment
                control_signal, int_val_v = self.calculate_pid(
                    delta_v, last_error_v, int_val_v, self.kp, self.ki, self.kd
                )
                
                # Apply throttle or brake
                throttle_output, brake_output = self.apply_throttle_brake(
                    control_signal, throttle_previous, brake_previous, finish_gliding
                )
                
                last_error_v = delta_v
            
            # Case a: Need to slow down for a curve - prioritize coasting
            elif needs_slowdown:
                # Calculate how much we need to slow down
                delta_v = target_speed - v_car
                
                # Gradual speed adjustment - prefer coasting (zero throttle)
                if delta_v < -0.5:  # We're going too fast for the curve
                    # Just coast by setting throttle to 0
                    throttle_output = 0.0
                    # Only apply gentle brake if speed is significantly over target
                    if delta_v < -2.0:
                        brake_output = min(0.3, abs(delta_v) * 0.1)  # Gentle braking
                    else:
                        brake_output = 0.0  # Let the car slow down naturally
                else:
                    # Normal speed control
                    control_signal, int_val_v = self.calculate_pid(
                        delta_v, last_error_v, int_val_v, self.kp, self.ki, self.kd
                    )
                    throttle_output, brake_output = self.apply_throttle_brake(
                        control_signal, throttle_previous, brake_previous, finish_gliding
                    )
                
                last_error_v = delta_v
            
            # Case 3: Normal speed control
            else:
                # Calculate velocity error
                delta_v = self.v_desired - v_car
                
                # Apply PID control for target speed
                control_signal, int_val_v = self.calculate_pid(
                    delta_v, last_error_v, int_val_v, self.kp, self.ki, self.kd
                )
                
                # Apply throttle or brake
                throttle_output, brake_output = self.apply_throttle_brake(
                    control_signal, throttle_previous, brake_previous, finish_gliding
                )
                
                last_error_v = delta_v
            
            throttle_previous = throttle_output
            brake_previous = brake_output

            self.publish_controller(throttle_output, steer_output, brake_output)

            if not finish :
                # Display: Time elapsed, Distance to last point, Current velocity, Total distance traveled
                rospy.logwarn(f'Time: {t_now-self.init_time:.1f}s | '
                             f'Distance to goal: {last_point_distance:.1f}m | '
                             f'Speed: {car_data[2]:.2f}m/s | '
                             f'Path K: {self.path_curvature:.3f} | '
                             f'Target Speed: {target_speed:.2f}m/s | '
                             f'Total traveled: {self.distance:.1f}m | '
                             f'Position: ({car_data[0]:.2f}, {car_data[1]:.2f}) | '
                             f'Throttle: {throttle_output:.2f}, Brake: {brake_output:.2f}')
            else :
                rospy.logwarn(f'LD: {last_point_distance:.1f}, V: {car_data[2]:.2f}')
            
        r.sleep()


def start_check():
    rospy.init_node('shell_simulation_node')
    pointer = 0
    check = Controller_Carla(pointer)
    check.check_car_info()

if __name__ == "__main__":
    start_check()

