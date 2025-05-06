#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
import math

class Laser_Subs(object):

    def __init__(self):
        self.front = list()
        self.back = list()
        self.left = list()
        self.right = list()

        self.laser_topic_name = "/carla/ego_vehicle/scan"
        self._check_data_ready()

    def _check_data_ready(self):
        self.laser_data = None
        while self.laser_data is None and not rospy.is_shutdown():
            try:
                self.laser_data = rospy.wait_for_message(self.laser_topic_name, LaserScan, timeout=1.0)
                rospy.logdebug("Current "+str(self.laser_topic_name)+" READY")

            except:
                rospy.logerr("Current "+str(self.laser_topic_name)+" not ready yet, retrying")

        self.process_laser_data(self.laser_data)

        self.sub = rospy.Subscriber (self.laser_topic_name, LaserScan, self.get_laser_data)

    def process_laser_data(self, msg):
        self.front = []
        self.front_left = []
        self.left = []
        self.right = []
        self.front_right = []

        for i in range(175,186):
            self.front.append(msg.ranges[i])
        self.min_front = min(self.front)

        for i in range(200,220):
            self.front_left.append(msg.ranges[i])
        self.min_front_left = min(self.front_left)

        for i in range(245, 295):
            self.left.append(msg.ranges[i])
        self.min_left = min(self.left)

        for i in range(65,115):
            self.right.append(msg.ranges[i])
        self.min_right = min(self.right)

        for i in range(140, 160):
            self.front_right.append(msg.ranges[i])
        self.min_front_right = min(self.front_right)


    def get_laser_data(self, msg):
        self.process_laser_data(msg)

    def return_s_min_laser(self):
        return self.min_front, self.min_left, self.min_right, self.min_front_left, self.min_front_right
