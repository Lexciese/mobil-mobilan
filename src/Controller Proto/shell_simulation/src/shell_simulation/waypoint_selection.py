#!/usr/bin/env python

import rospy
from shell_simulation.odom_data_processor import OdometryProcessor
from shell_simulation.laserscan_subscriber import Laser_Subs
from tf.transformations import euler_from_quaternion
import shell_simulation.lanes as lanes
import math

RESOLUTION = 0.1

class WaypointSelection(object):

    def __init__(self, pointer):
        self.pointer = pointer
        self.pointer_previous = 1

        self.lane = 1   # 0 left 1 right
        self.state = 0  # 0 aman 1 pindah 2 maintain 3 gliding_finish
        
        self.left = lanes.left
        self.right = lanes.right

        self.front_d = 100
        self.finished = False
        self.finished_changing_lane = False
        self.pointer_change_lane = 0

        self.odom_data_processor = OdometryProcessor()
        self.laserscan_subscriber = Laser_Subs()

        self.s_min_f = 8.0
        self.s_min_fs = 4.0
        self.s_min_s = 4.0

    def waypoint_selection(self):
        s_front, s_left, s_right, s_front_left, s_front_right = self.laserscan_subscriber.return_s_min_laser()
        
        self.front_d = s_front
        
        #rospy.logerr(f'FINISH_CHANGE_LANE {self.finished_changing_lane}')
        #rospy.logerr(f'Pointer : {self.pointer}')
        # rospy.logerr(f'lane {self.lane} state {self.state}')
        # rospy.logerr(f'{s_front} {s_front_left}')
        # rospy.logerr(f'f_left{s_front_left} f {s_front} f_right {s_front_right}')

        if self.state == 0 :
            if s_front <= self.s_min_f :
                if (self.lane == 0 and (s_front_right <= self.s_min_fs or s_right <= self.s_min_s)) or (self.lane == 1 and (s_front_left <= self.s_min_fs or s_left <= self.s_min_s)):
                    self.state = 2
                    #rospy.logerr(f'state {self.state}')
                else:
                    self.state = 1
                    self.lane = 1 - self.lane
                        
                    self.pointer_prev = self.pointer
                    #rospy.logerr(f'change lane {self.lane}')

        if(self.lane == 0):
            waypoint_data = self.left
        else:
            waypoint_data = self.right

        car_data = self.odom_data_processor.get_car_info()

        ######################## Pindah Lane #################################################

        if self.state == 1 and self.finished_changing_lane:
            nearest_point_prev = math.inf
            POINTER_RANGE = int(100/(RESOLUTION))
            for i  in range(self.pointer-POINTER_RANGE, self.pointer+POINTER_RANGE):
                try:
                    nearest_point = math.hypot(waypoint_data[i][1]-car_data[1], waypoint_data[i][0]-car_data[0])
                except:
                    pass

                if nearest_point < nearest_point_prev:
                    nearest_pointer = i
                    self.finished_changing_lane = False
                    nearest_point_prev = nearest_point
                    self.pointer = nearest_pointer
                    self.pointer_change_lane = self.pointer
            
        if self.state == 1 and self.pointer > self.pointer_change_lane + 7.5/(RESOLUTION):
            self.finished_changing_lane = True
            self.state = 0

        if self.state == 2 and s_front > self.s_min_f:
            self.state = 0

        ############################ Indexing #################################################
        
        if self.pointer != self.pointer_previous:
            if self.pointer==0:
                self.x0_y0 = [car_data[0], car_data[1]]
                self.x1_y1 = [waypoint_data[self.pointer][0], waypoint_data[self.pointer][1]]
                self.pointer_previous = self.pointer

            else:
                self.x0_y0 = [waypoint_data[self.pointer-1][0], waypoint_data[self.pointer-1][1]]
                self.x1_y1 = [waypoint_data[self.pointer][0], waypoint_data[self.pointer][1]]

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

            try:
                self.x1_y1 = [waypoint_data[self.pointer][0], waypoint_data[self.pointer][1]]
            except:
                self.x1_y1 = self.x0_y0
                self.finished = True

            if (self.state == 2):
                if not ((self.lane == 0 and s_front_right <= self.s_min_fs) or (self.lane == 1 and s_front_left <= self.s_min_fs)):
                    self.state = 0
            

    def waypoint_goal(self):
        self.waypoint_selection()
        return self.x0_y0, self.x1_y1, self.front_d

    def get_path_points(self, num_points=5):
        """
        Get a sequence of waypoints for path representation.
        Returns a list of (x,y) tuples representing points on the path.
        """
        points = []
        
        # Add current waypoint points from x0_y0 and x1_y1
        if hasattr(self, 'x0_y0') and hasattr(self, 'x1_y1'):
            points.append((self.x0_y0[0], self.x0_y0[1]))
            points.append((self.x1_y1[0], self.x1_y1[1]))
        
        # Use the current lane's waypoints
        if self.lane == 0:
            waypoint_data = self.left
        else:
            waypoint_data = self.right
            
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
        # Use the current lane's waypoints
        if self.lane == 0:
            waypoint_data = self.left
        else:
            waypoint_data = self.right
            
        # Get a point 2 positions ahead of current pointer
        future_idx = self.pointer + 2
        if future_idx < len(waypoint_data):
            return (waypoint_data[future_idx][0], waypoint_data[future_idx][1])
        return None