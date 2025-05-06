#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import math

class OdometryProcessor(object):

    def __init__(self):
        self.odom_topic_name = '/carla/ego_vehicle/odometry'
        self._check_odom_data_ready()


    def _check_odom_data_ready(self):
        self.odom_data = None
        while self.odom_data is None and not rospy.is_shutdown():
            try:
                self.odom_data = rospy.wait_for_message(self.odom_topic_name, Odometry, timeout=1.0)
                rospy.logdebug("Current "+str(self.odom_topic_name)+" READY")

            except:
                rospy.logerr("Current "+str(self.odom_topic_name)+" not ready yet, retrying")

        self.process_odom_data(self.odom_data)

        self.sub = rospy.Subscriber (self.odom_topic_name, Odometry, self.get_odom_data)


    def extract_rpy_odom_data (self, odom_msg):
        orientation_quaternion = odom_msg.pose.pose.orientation
        orientation_list = [orientation_quaternion.x,
                            orientation_quaternion.y,
                            orientation_quaternion.z,
                            orientation_quaternion.w]

        roll, pitch, yaw = euler_from_quaternion (orientation_list)

        return roll, pitch, yaw


    def process_odom_data(self, msg):
        roll, pitch, yaw = self.extract_rpy_odom_data(msg)

        #get current position of the car
        car_x = msg.pose.pose.position.x
        car_y = msg.pose.pose.position.y
        car_z = msg.pose.pose.position.z

        #get current velocity of the car
        car_v = math.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)

        self.car_odom = [car_x,car_y,car_v,yaw,car_z]


    def get_odom_data(self, msg):
        self.process_odom_data(msg)

    
    def get_car_info(self):
        return self.car_odom


