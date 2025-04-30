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

        self.d_aman = 6.5 
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

        # Lateral Control Parameters
        self.k_e = 1  # Cross track error gain (reduced from 4)
        self.k_s = 3.0  # Speed coefficient (reduced from 5)
        self.k_t = 0.8  # Heading error gain (increased from 0.5)

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

            #ax + by + c = 0
            a = y1 - y0
            b = -(x1 - x0)
            c = y0*x1 - x0*y1

            #yaw error
            yaw_path = self.normalize_angle(np.arctan2(y1 - y0, x1 - x0))
            theta_e = self.normalize_angle(yaw_path - yaw_car)

            #crosstrack error
            try:
                e = (a*xc + b*yc + c)/math.sqrt(a**2 + b**2)
                # Determine sign based on which side of the path we're on
                if (a*np.cos(yaw_car) + b*np.sin(yaw_car)) > 0:
                    e = -e
            except ZeroDivisionError:
                e = 0.0  # Handle division by zero

            # Look-ahead distance proportional to velocity
            look_ahead = 1.0 + 0.1 * v_car
            # look_ahead = 0.5 + 0.1 * v_car * v_car

            # Adaptive gain based on velocity - reduces steering at higher speeds
            adaptive_k_e = self.k_e / (look_ahead)
            
            
            # Calculate path direction vector
            path_vector = np.array([x1 - x0, y1 - y0])
            path_length = np.linalg.norm(path_vector)
            
            if path_length > 0:
                # Normalize path vector
                path_vector = path_vector / path_length
                
                # Project look-ahead point along the path
                look_ahead_point = np.array([x0, y0]) + path_vector * min(look_ahead, path_length)
                
                # Use look-ahead point to calculate heading error for smoother anticipation of turns
                yaw_path = self.normalize_angle(np.arctan2(path_vector[1], path_vector[0]))
                theta_e = self.normalize_angle(yaw_path - yaw_car)
                
                # Weight cross-track error based on look-ahead distance
                adaptive_weight = 1.0 / (1.0 + 0.05 * look_ahead)
                e_weighted = e * adaptive_weight
                
                # Stanley controller with weighted components
                theta_d = np.arctan2(adaptive_k_e * e_weighted, self.k_s + v_car)
            else:
                # Fallback if path is too short
                theta_d = np.arctan2(adaptive_k_e * e, self.k_s + v_car)
            
            delta = self.k_t * theta_e + theta_d
            
            # Normalize and clip the steering angle
            delta = self.normalize_angle(delta)
            
            if not finish_gliding:
                steer_output = np.clip(delta, -1.0, 1.0)
            else:
                steer_output = 0.0

            ######################### Longitudinal ############################
            isWarning = front_d < 8.0

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
                if front_d < 4.0:
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
                             f'Cross-track error: {abs(e):.2f}m | '
                             f'Speed: {car_data[2]:.2f}m/s | '
                             f'Total traveled: {self.distance:.1f}m')
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

