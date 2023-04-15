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
from PID import PID
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
from duckietown_msgs.msg import AprilTagDetection, AprilTagDetectionArray
from std_msgs.msg import Header, Float32, String, Float64MultiArray, Float32MultiArray, Int32, Bool
from sensor_msgs.msg import CompressedImage, Range
from geometry_msgs.msg import Point32
from nav_msgs.msg import Odometry
from turbojpeg import TurboJPEG 
import cv2
import rosbag
import tf2_ros

# Change this before executing
VERBOSE = 0
SIM = False
DEBUG = False



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
        self.update_freq = 20
        self.rate = rospy.Rate(self.update_freq)
        self.d_offset = -0.01
        self.lane_controller_parameters = {
            "Kp_d": 10.0,
            "Ki_d": 1.0,
            "Kd_d": 0.225,
            "Kp_theta": 6.0,
            "Ki_theta": 0.25,
            "Kd_theta": 0.025,
            "sample_time": 0.01,
            "d_bounds": (-2.0, 2.0),
            "theta_bounds": (-2.0,2.0),
        }

        ## for stop line detection
        self.stop_distance = 0.15 # distance from the stop line that we should stop
        self.min_segs = 15  # minimum number of red segments that we should detect to estimate a stop
        self.off_time = 2.0 # time to wait after we have passed the stop line
        self.max_y = 0.10   # If y value of detected red line is smaller than max_y we will not set at_stop_line true.
        self.stop_hist_len = 10
        self.stop_duration = 1
        self.stop_cooldown = 6 # The stop cooldown
        self.duckwalk_cooldown = 6
        ## For duckiebot detection and avoidance
        self.safe_distance = 0.65
        self.english_drive_cooldown = 5
        ## For parking
        self.parking_ids = [-1, 207, 226, 228, 75]
        ## What stall you are suppose to park at
        self.parking_stall = 2


        # Initialize variables
        self.lane_pose = LanePose()
        self.lane_pid_controller = LaneController(self.lane_controller_parameters)
        ## For stop line detection
        self.stop_line_distance = None
        self.stop_line_detected = False
        self.at_stop_line = False
        self.cmd_stop = False
        self.stop_hist = []
        self.stop_time = 0.0
        self.process_intersection = False
        self.tag_id = -1
        ## For duckie detection
        self.duckie_detected = False
        self.process_duckwalk = False
        self.duckwalk_time = 0.0
        ## For vehicle distance, detection, and avoidance
        self.vehicle_distance = 99.99
        self.vehicle_detected = False
        self.english_driver_time = 0.0
        self.english_driver = False

        ## For parking 
        self.all_tag_poses = AprilTagDetectionArray().detections 
        self.approaching_stage_3 = False

        # Publishers
        ## Publish commands to the motors
        self.pub_motor_commands = rospy.Publisher(f'/{self.veh_name}/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size=1)
        self.pub_car_cmd = rospy.Publisher(f'/{self.veh_name}/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=1)
        self.pub_duckiewalk = rospy.Publisher(f'{self.veh_name}/duckiewalk/image/compressed', CompressedImage, queue_size=1)
        self.pub_shutdown_cmd = rospy.Publisher(f'{self.veh_name}/shutdown_cmd', String, queue_size=1)
        # Subscribers
        ## Subscribe to the lane_pose node
        self.sub_lane_reading = rospy.Subscriber(f"/{self.veh_name}/lane_filter_node/lane_pose", LanePose, self.cb_lane_pose, queue_size = 1)
        self.sub_segment_list = rospy.Subscriber(f"/{self.veh_name}/line_detector_node/segment_list", SegmentList, self.cb_segments, queue_size=1)
        self.sub_tag_id = rospy.Subscriber(f"/{self.veh_name}/tag_id", Int32, self.cb_tag_id, queue_size=1)

        self.sub_duckie_detected = rospy.Subscriber(f'/{self.veh_name}/all_detection/duckie_detected', Bool, self.cb_duckie_detected, queue_size = 1)
        self.sub_duckiewalk_detected = rospy.Subscriber(f'/{self.veh_name}/all_detection/duckwalk_detected', Bool, self.cb_duckwalk_detected, queue_size = 1)
        self.sub_duckiebot_detected = rospy.Subscriber(f'/{self.veh_name}/all_detection/duckiebot_detected', Bool, self.cb_duckiebot_detected, queue_size = 1)

        self.sub_range_finder = rospy.Subscriber(f'/{self.veh_name}/front_center_tof_driver_node/range', Range, self.cb_range_finder, queue_size = 1)
        
        self.sub_all_tag_poses = rospy.Subscriber(f"/{self.veh_name}/all_tag_poses", AprilTagDetectionArray, self.cb_all_tag_poses, queue_size=1)

        self.april_tag = -1

        rospy.on_shutdown(self.custom_shutdown)
            
        self.log("Initialized")

        rospy.sleep(1.0)
        self.lane_pid_controller.reset_controller()

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
        self.get_control_action()

    def cb_duckie_detected(self, duckie_detected_msg):
        self.duckie_detected = duckie_detected_msg.data

    def cb_duckwalk_detected(self, bool_msg):
        self.process_duckwalk = bool_msg.data

    def cb_duckiebot_detected(self, duckiebot_detected_msg):
        self.vehicle_detected = duckiebot_detected_msg.data

    def cb_range_finder(self, range_msg):
        self.vehicle_distance = range_msg.range

    def cb_tag_id(self, tag_msg):
        # Only set tags that are different and have been found
        if tag_msg.data != -1 and tag_msg.data != self.tag_id:
            print("tag_msg",tag_msg)
            self.tag_id = tag_msg.data
            if self.tag_id == 38:
                self.approaching_stage_3 = True

    def cb_all_tag_poses(self, tag_msg):
        self.all_tag_poses = tag_msg.detections

    ## end of callback functions

    def vehicle_ahead(self):
        # print(f"Vehicle detected: {self.vehicle_detected}, distance: {self.vehicle_distance}")

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
        duckwalk_time_diff = curr_time - self.duckwalk_time
        english_driver_time_diff = curr_time - self.english_driver_time
        d_err = self.lane_pose.d - self.d_offset
        phi_err = self.lane_pose.phi

        if (self.cmd_stop and stop_time_diff > self.stop_cooldown):
            self.stop_time = curr_time
            v = 0.0
            omega = 0.0

            self.process_intersection = True
            self.car_cmd(v, omega)
            rospy.sleep(self.stop_duration)

        elif self.process_intersection:
            if self.tag_id in [48, 94]:
                print("Gping right")
                self.turn_right()
            elif self.tag_id in [50, 169]:
                print("Going left")
                self.turn_left()
            elif self.tag_id in [227, 38] or self.approaching_stage_3:
                self.do_parking()
            else:
                print("Going straight")
                self.go_straight()

            self.process_intersection = False
        elif self.process_duckwalk and self.tag_id in [21, 163] and duckwalk_time_diff > self.duckwalk_cooldown:
            v = 0.0
            omega = 0.0
            self.car_cmd(v, omega)
            print("Doing duckwalk")
            rospy.sleep(5.0)
            self.do_duckwalk()
            
        elif self.vehicle_ahead() and not self.english_driver:
            v = 0.0
            omega = 0.0
            print("Vehicle detected")
            self.car_cmd(v, omega)
            rospy.sleep(3.0)
            print("Going around")
            self.car_cmd(v=0.25, omega=2.0)
            rospy.sleep(1.0)
            self.car_cmd(v=0.25, omega=0.0)
            rospy.sleep(1.7)
            self.car_cmd(v=0.25, omega=-2.0)
            rospy.sleep(2.5)
            self.car_cmd(v=0.25, omega=0.0)
            rospy.sleep(1)
            self.car_cmd(v=0.25, omega=2.0)
            rospy.sleep(1)
            self.lane_pid_controller.reset_controller()
        else:
            _, omega = self.lane_pid_controller.compute_control_actions(d_err, phi_err, None)
            self.car_cmd(0.25, omega)

        self.rate.sleep()

    def car_cmd(self, v, omega):
        car_control_msg = Twist2DStamped()

        car_control_msg.v = v
        car_control_msg.omega = omega * 2.0
        self.pub_car_cmd.publish(car_control_msg)
    
    def turn_right(self):
        """Make a right turn at an intersection"""
        self.car_cmd(v=0.25, omega=0)
        rospy.sleep(1.4)
        self.car_cmd(v=0.35, omega=-3.25)
        rospy.sleep(0.66)
        self.stop_hist = []
        self.cmd_stop = False
        self.lane_pid_controller.reset_controller()

    def turn_left(self):
        """Make a left turn at an intersection"""
        self.car_cmd(v=0.30, omega = 1.00)
        rospy.sleep(2.5)
        self.stop_hist = []
        self.cmd_stop = False
        self.lane_pid_controller.reset_controller()

    def go_straight(self):
        """Go straight at an intersection"""
        self.car_cmd(v = 0.25, omega = 0.0)
        rospy.sleep(2.5)
        self.stop_hist = []
        self.cmd_stop = False
        self.lane_pid_controller.reset_controller()

    def do_duckwalk(self):
        """Processing duckwalk"""
        while self.duckie_detected:
            print("duckie detected")
            rospy.sleep(2.0)
            self.car_cmd(v=0.0, omega=0.0)
        
        self.car_cmd(v=0.25, omega=0.0)
        rospy.sleep(3)
        self.process_duckwalk = False
        self.lane_pid_controller.reset_controller()
        self.duckwalk_time = rospy.get_time()
        print("Done duckwalk")

    def do_parking(self):
        """Do parking"""
        print("Doing parking")
        self.lane_pid_controller.reset_controller()
        if self.parking_stall == 1:
            self.car_cmd(v=0.25, omega=0.0)
            rospy.sleep(1.5)
            self.car_cmd(v=0.25, omega=2.0)
            rospy.sleep(2.95)
            self.car_cmd(v=0.0,omega=0.0)
            rospy.sleep(1)
            self.drive_to_tag(self.parking_ids[self.parking_stall], stop_dist=0.15)
        elif self.parking_stall == 2:
            self.car_cmd(v=0.25, omega=2.00)
            rospy.sleep(2.1)
            self.car_cmd(v=0.0,omega=0.0)
            rospy.sleep(1)
            self.drive_to_tag(self.parking_ids[self.parking_stall], stop_dist=0.15)
        elif self.parking_stall == 3:
            self.car_cmd(v=0.30, omega=0.0)
            rospy.sleep(1.0)
            self.car_cmd(v=0.25, omega=-2.25)
            rospy.sleep(3.0)
            self.car_cmd(v=0.0,omega=0.0)
            rospy.sleep(1)
            self.drive_to_tag(self.parking_ids[self.parking_stall], stop_dist=0.15)
        elif self.parking_stall == 4:
            self.car_cmd(v=0.30, omega=0.0)
            rospy.sleep(1.25)
            self.car_cmd(v=0.25, omega=-2.45)
            rospy.sleep(1.1)
            self.car_cmd(v=0.0,omega=0.0)
            rospy.sleep(1)
            self.drive_to_tag(self.parking_ids[self.parking_stall], stop_dist=0.15)
        else:
            print(f"Parking Stall: {self.parking_stall} is not valid")
        print("Done parking")
        rospy.sleep(5)
        self.pub_shutdown_cmd.publish('shutdown')        
        self.shutdown('shutdown', True)

    def drive_to_tag(self, tag_id, stop_dist = 0.25):
        self.lane_pid_controller.reset_controller()
        lane_controller_parameters = {
            "Kp_d": 5.0,
            "Ki_d": 0.5,
            "Kd_d": 0.25,
            "Kp_theta": 3.0,
            "Ki_theta": 0.25,
            "Kd_theta": 0.025,
            "sample_time": 0.05,
            "d_bounds": (-1.0, 1.0),
            "theta_bounds": (-1.0,1.0),
        }
        parking_pid_controller = LaneController(lane_controller_parameters)

        while True:
            target_tag = None
            all_tags = self.all_tag_poses
            if self.vehicle_distance <= stop_dist:
                self.car_cmd(v=0.0, omega=0.0)
                return 
            for tag in all_tags:
                if tag.tag_id == tag_id:
                    target_tag = tag
                    print(target_tag)
                    if target_tag.transform.translation.x <= stop_dist:
                        self.car_cmd(v=0.0, omega=0.0)
                        return
                    d_err = tag.transform.translation.y * 1.5
                    omega_err = (0.50 - tag.transform.rotation.z) * 1.5
                    _, omega = parking_pid_controller.compute_control_actions(-d_err, omega_err, None)
                    self.car_cmd(v=0.25, omega=omega)
                    break
            self.rate.sleep()


    def to_lane_frame(self, point):
        p_homo = np.array([point.x, point.y, 1])
        phi = self.lane_pose.phi
        d = self.lane_pose.d
        T = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), d], [0, 0, 1]])
        p_new_homo = T.dot(p_homo)
        p_new = p_new_homo[0:2]
        return p_new

    
    def set_english_driver(self, set_english: bool):
        if set_english:
            self.d_offset = 0.20
        else:
            self.d_offset = -0.0

    def _clamp(self, value, lower=-1, upper=1):
        return max(lower, min(value, upper))

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

    def shutdown(self, msg, override = False):
        if override:
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
    rospy.spin()