#!/usr/bin/env python3

"""
This is the robot circuit node for exercise 5 
based on the lane controller node from dt-core here: https://github.com/duckietown/dt-core/blob/daffy/packages/lane_control/src/lane_controller_node.py
the stop lane filter code is from here https://raw.githubusercontent.com/duckietown/dt-core/daffy/packages/stop_line_filter/src/stop_line_filter_node.py

"""

import numpy as np
import os
import math
import rospy
import time
import message_filters
import typing
from statistics import mode
from lane_controller import LaneController

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import (
    Pose2DStamped, 
    LanePose, 
    WheelEncoderStamped, 
    WheelsCmdStamped, 
    Twist2DStamped,
    BoolStamped,
    VehicleCorners,
    SegmentList,
    LEDPattern,
    )
from duckietown_msgs.srv import SetCustomLEDPattern
from std_msgs.msg import Header, Float32, String, Float64MultiArray, Float32MultiArray, Int32, Bool
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point32

import rosbag


# Change this before executing
VERBOSE = 0
SIM = False


class RobotCircuitNode(DTROS):
    """
    Robot Circuit Node is used to generate robot following commands based on the lane pose and make turns at certain intersections such that I looks at all the AprilTags.
    """
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(RobotCircuitNode, self).__init__(node_name=node_name, node_type=NodeType.DRIVER)
        if os.environ["VEHICLE_NAME"] is not None:
            self.veh_name = os.environ["VEHICLE_NAME"]
        else:
            self.veh_name = "csc22945"

        # Static parameters
        self.update_freq = 10
        self.rate = rospy.Rate(self.update_freq)
        ## for stop line detection
        self.stop_distance = 0.15 # distance from the stop line that we should stop
        self.min_segs = 20  # minimum number of red segments that we should detect to estimate a stop
        self.off_time = 2.0 # time to wait after we have passed the stop line
        self.max_y = 0.10   # If y value of detected red line is smaller than max_y we will not set at_stop_line true.
        self.stop_hist_len = 5
        self.stop_duration = 1
        self.stop_cooldown = 6 # The stop cooldown
        ## For duckiebot detection and avoidance
        self.safe_distance = 0.45
        self.english_drive_cooldown = 5

        # Initialize variables
        self.lane_pose = LanePose()
        ## lane following stuff
        self.lane_follow_omega = 0.0
        self.lane_follow_v = 0.0
        ## For stop line detection
        self.stop_line_distance = None
        self.stop_line_detected = False
        self.at_stop_line = False
        self.cmd_stop = False
        self.stop_hist = []
        self.stop_time = 0.0
        self.process_intersection = False
        self.tag_id = -1
        self.action_done = False
        ## For duckie detection
        self.duckie_detected = False
        ## For vehicle distance, detection, and avoidance
        self.vehicle_distance = 99.99
        self.vehicle_detected = False
        self.english_drive_time = 0.0
        self.english_driver = False
        
        # Publishers
        ## Publish commands to the motors
        self.pub_motor_commands = rospy.Publisher(f'/{self.veh_name}/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size=1)
        self.pub_car_cmd = rospy.Publisher(f'/{self.veh_name}/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=1)
        ## (re)set lane controller parameters
        self.pub_reset_lane_follow = rospy.Publisher(f'/{self.veh_name}/lane_follow_node/reset', Bool, queue_size=1)
        self.pub_english_lane_follow = rospy.Publisher(f'/{self.veh_name}/lane_follow_node/english', Bool, queue_size=1)

        # Subscribers
        ## Subscribe to the lane_pose node
        self.sub_lane_reading = rospy.Subscriber(f"/{self.veh_name}/lane_filter_node/lane_pose", LanePose, self.cb_lane_pose, queue_size = 1)
        self.sub_segment_list = rospy.Subscriber(f"/{self.veh_name}/line_detector_node/segment_list", SegmentList, self.cb_segments, queue_size=1)
        self.sub_tag_id = rospy.Subscriber(f"/{self.veh_name}/tag_id", Int32, self.cb_tag_id, queue_size=1)
        self.sub_shutdown_commands = rospy.Subscriber(f'/{self.veh_name}/number_detection_node/shutdown_cmd', String, self.shutdown, queue_size = 1)
        self.sub_lane_follow = rospy.Subscriber(f'/{self.veh_name}/lane_follow_node/car_cmd', Twist2DStamped, self.cb_lane_follow, queue_size = 1)
        self.sub_duckie_detected = rospy.Subscriber(f'/{self.veh_name}/duckie_detected', Bool, self.cb_duckie_detected, queue_size = 1)
        self.sub_detection = rospy.Subscriber(f"/{self.veh_name}/duckiebot_detection_node/detection", BoolStamped, self.cb_detection, queue_size=1)
        self.sub_distance_to_robot_ahead = rospy.Subscriber(f"/{self.veh_name}/duckiebot_distance_node/distance", Float32, self.cb_vehicle_distance, queue_size=1)

        #self.april_tag = -1

        rospy.on_shutdown(self.custom_shutdown)

        self.log("Initialized")

    # Start of callback functions
    def cb_segments(self, segment_list_msg):
        good_seg_count = 0
        stop_line_x_accumulator = 0.0
        stop_line_y_accumulator = 0.0
        for segment in segment_list_msg.segments:
            if segment.color != segment.RED:
                continue
            if segment.points[0].x < 0 or segment.points[1].x < 0: # The point is behind the robot
                continue
            p1_lane = self.to_lane_frame(segment.points[0])
            p2_lane = self.to_lane_frame(segment.points[1])
            avg_x = 0.5 * (p1_lane[0] + p2_lane[0])
            avg_y = 0.5 * (p1_lane[1] + p2_lane[1])
            stop_line_x_accumulator += avg_x
            stop_line_y_accumulator += avg_y
            good_seg_count += 1
        
        if good_seg_count < self.min_segs:
            self.stop_line_detected = False
            at_stop_line = False
            self.stop_line_distance = 99.9
        else:
            self.stop_line_detected = True
            stop_line_point_x = stop_line_x_accumulator / good_seg_count
            stop_line_point_y = stop_line_y_accumulator / good_seg_count
            self.stop_line_distance = np.sqrt(stop_line_point_x**2 + stop_line_point_y**2)
            # Only detect stop line if y is within max_y distance
            at_stop_line = (
                stop_line_point_x < self.stop_distance and np.abs(stop_line_point_y) < self.max_y
            )
        
        self.process_stop_line(at_stop_line)

    def cb_lane_pose(self, input_pose_msg):
        self.lane_pose = input_pose_msg

    def cb_lane_follow(self, lane_follow_msg):
        self.lane_follow_v = lane_follow_msg.v 
        self.lane_follow_omega = lane_follow_msg.omega
    
    def cb_duckie_detected(self, duckie_detected_msg):
        value = duckie_detected_msg.data
        if value == 1:
            self.duckie_detected = True
        else:
            self.duckie_detected = False
    
    def cb_tag_id(self, tag_msg):
        print("tag_msg",tag_msg)

        # Means no target detected.
        if tag_msg.data == -1:
            self.tag_id = -1

        else:
            # New action coming.
            if tag_msg.data != self.tag_id:
                self.tag_id = tag_msg.data
                self.action_done = False

    def cb_detection(self, detection_msg):
        self.vehicle_detected = detection_msg.data

    def cb_vehicle_distance(self, distance_msg):
        self.vehicle_distance = distance_msg.data

    ## end of callback functions

    def vehicle_ahead(self):
        if self.vehicle_detected and self.vehicle_distance < self.safe_distance:
            return True
        else:
            return False

    def process_stop_line(self, at_stop_line):
        """Storing the current distance to the next stop line, if one is detected.

        Args:
            msg (StopLineReading): The message containing the distance to the next stop line.
        """
        if len(self.stop_hist) > self.stop_hist_len:
            self.stop_hist.pop(0)

        if at_stop_line:
            self.stop_hist.append(True)
        else:
            self.stop_hist.append(False)

        if mode(self.stop_hist) == True:
            self.cmd_stop = True
        else:
            self.cmd_stop = False

    def get_control_action(self):
        """
        Callback function that receives a pose message and updates the related control command
        """
        curr_time = rospy.get_time()

        stop_time_diff = curr_time - self.stop_time
        english_driver_time_diff = curr_time - self.english_driver_time


        if (self.cmd_stop and stop_time_diff > self.stop_cooldown):
            self.stop_time = curr_time
            v = 0.0
            omega = 0.0

            self.process_intersection = True
            self.car_cmd(v, omega)
            rospy.sleep(self.stop_duration)
        elif self.process_intersection:
            if self.tag_id == 48:
                print("Gping right")
                self.turn_right()
            elif self.tag_id == 50:
                print("Going left")
                self.turn_left()
            else:
                print("Going straight")
                self.go_straight()

            self.process_intersection = False
        elif self.duckie_detected:
            v = 0.0
            omega = 0.0
            print("Duckie detected")
            self.car_cmd(v, omega)
            rospy.sleep(3.0)

        elif self.vehicle_ahead():
            v = 0.0
            omega = 0.0
            print("Vehicle detected")
            self.car_cmd(v, omega)
            rospy.sleep(3.0)

            self.english_driver_time = curr_time
            self.english_driver = True
            self.set_english_driver(True)
            
        elif self.english_driver and english_driver_time_diff > self.english_driver_duration:
            self.set_english_driver(False)
            self.english_driver = False
        else:
            self.car_cmd(self.lane_follow_v, self.lane_follow_omega)

    def car_cmd(self, v, omega):
        car_control_msg = Twist2DStamped()

        car_control_msg.v = v
        car_control_msg.omega = omega
        print(car_control_msg)
        self.pub_car_cmd.publish(car_control_msg)
    
    def turn_right(self):
        """Make a right turn at an intersection"""
        self.car_cmd(v=0.4, omega=0)
        rospy.sleep(1.0)
        self.car_cmd(v=0.45, omega=-5)
        rospy.sleep(2.0)
        self.stop_hist = []
        self.cmd_stop = False
        self.reset_lane_controller()

    def turn_left(self):
        """Make a left turn at an intersection"""
        self.car_cmd(v=0.5, omega = 2.0)
        rospy.sleep(3)
        self.stop_hist = []
        self.cmd_stop = False
        self.reset_lane_controller()

    def go_straight(self):
        """Go straight at an intersection"""
        self.car_cmd(v = 0.3, omega = 0.0)
        rospy.sleep(4)
        self.stop_hist = []
        self.cmd_stop = False
        self.reset_lane_controller()

    def to_lane_frame(self, point):
        p_homo = np.array([point.x, point.y, 1])
        phi = self.lane_pose.phi
        d = self.lane_pose.d
        T = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), d], [0, 0, 1]])
        p_new_homo = T.dot(p_homo)
        p_new = p_new_homo[0:2]
        return p_new

    def reset_lane_controller(self):
        reset_bool = Bool()
        reset_bool.data = True
        self.pub_reset_lane_follow.publish(reset_bool)

    
    def set_english_driver(self, set_english: bool):
        english_bool = Bool()
        english_bool.data = set_english
        self.pub_english_lane_follow.publish(english_bool)

    def custom_shutdown(self):
        """Cleanup function."""
        while not rospy.is_shutdown():
            motor_cmd = WheelsCmdStamped()
            motor_cmd.header.stamp = rospy.Time.now()
            motor_cmd.vel_left = 0.0
            motor_cmd.vel_right = 0.0
            self.pub_motor_commands.publish(motor_cmd)
            car_control_msg = Twist2DStamped()
            car_control_msg.header.stamp = rospy.Time.now()
            car_control_msg.v - 0.0
            car_control_msg.omega = 0.0
            self.pub_car_cmd.publish(car_control_msg)

    def shutdown(self, msg):
        if msg.data=="shutdown":
            motor_cmd = WheelsCmdStamped()
            motor_cmd.header.stamp = rospy.Time.now()
            motor_cmd.vel_left = 0.0
            motor_cmd.vel_right = 0.0
            self.pub_motor_commands.publish(motor_cmd)
            car_control_msg = Twist2DStamped()
            car_control_msg.header.stamp = rospy.Time.now()
            car_control_msg.v - 0.0
            car_control_msg.omega = 0.0
            self.pub_car_cmd.publish(car_control_msg)
            time.sleep(2)

            rospy.signal_shutdown("Robot circuit Node Shutdown command received")
            time.sleep(1)

if __name__ == '__main__':
    node = RobotCircuitNode(node_name='robot_circuit_node')
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        node.get_control_action()
        rate.sleep()