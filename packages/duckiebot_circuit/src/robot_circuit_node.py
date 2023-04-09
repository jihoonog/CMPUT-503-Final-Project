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
from std_msgs.msg import Header, Float32, String, Float64MultiArray, Float32MultiArray, Int32, Bool
from sensor_msgs.msg import CompressedImage, Range
from geometry_msgs.msg import Point32
from nav_msgs.msg import Odometry
from turbojpeg import TurboJPEG 
import cv2
import rosbag


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
        self.update_freq = 15
        self.rate = rospy.Rate(self.update_freq)
        self.velocity_controller = PID(
            Kp=0.5,
            Ki=0.0,
            Kd=0.1,
            setpoint=0.25,
            sample_time= 0.01,
            output_limits=(0.0, 0.5)
        )
        self.d_offset = 0.0
        self.lane_controller_parameters = {
            "Kp_d": 7.0,
            "Ki_d": 0.75,
            "Kd_d": 0.125,
            "Kp_theta": 5.0,
            "Ki_theta": 0.25,
            "Kd_theta": 0.025,
            "sample_time": 0.01,
            "d_bounds": (-2.0, 2.0),
            "theta_bounds": (-2.0,2.0),
        }

        ## for stop line detection
        self.stop_distance = 0.15 # distance from the stop line that we should stop
        self.min_segs = 20  # minimum number of red segments that we should detect to estimate a stop
        self.off_time = 2.0 # time to wait after we have passed the stop line
        self.max_y = 0.10   # If y value of detected red line is smaller than max_y we will not set at_stop_line true.
        self.stop_hist_len = 10
        self.stop_duration = 1
        self.stop_cooldown = 6 # The stop cooldown
        ## For duckiebot detection and avoidance
        self.safe_distance = 0.45
        self.english_drive_cooldown = 5
        ## For duckwalk stopage
        self.duckwalk_stop_distance = 75


        # Initialize variables
        self.lane_pose = LanePose()
        self.lane_pid_controller = LaneController(self.lane_controller_parameters)
        self.first_start = True
        self.jpeg = TurboJPEG()
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
        self.duckwalk_cy = 0.0
        self.process_duckwalk = False
        ## For vehicle distance, detection, and avoidance
        self.vehicle_distance = 99.99
        self.vehicle_detected = False
        self.english_driver_time = 0.0
        self.english_driver = False

        
        # Publishers
        ## Publish commands to the motors
        self.pub_motor_commands = rospy.Publisher(f'/{self.veh_name}/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size=1)
        self.pub_car_cmd = rospy.Publisher(f'/{self.veh_name}/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=1)
        self.pub_duckiewalk = rospy.Publisher(f'{self.veh_name}/duckiewalk/image/compressed', CompressedImage, queue_size=1)
        ## (re)set lane controller parameters

        # Subscribers
        ## Subscribe to the lane_pose node
        self.sub_lane_reading = rospy.Subscriber(f"/{self.veh_name}/lane_filter_node/lane_pose", LanePose, self.cb_lane_pose, queue_size = 1)
        self.sub_segment_list = rospy.Subscriber(f"/{self.veh_name}/line_detector_node/segment_list", SegmentList, self.cb_segments, queue_size=1)
        self.sub_tag_id = rospy.Subscriber(f"/{self.veh_name}/tag_id", Int32, self.cb_tag_id, queue_size=1)
        self.sub_shutdown_commands = rospy.Subscriber(f'/{self.veh_name}/number_detection_node/shutdown_cmd', String, self.shutdown, queue_size = 1)
        self.sub_duckie_detected = rospy.Subscriber(f'/{self.veh_name}/duckie_detected', Int32, self.cb_duckie_detected, queue_size = 1)
        self.sub_range_finder = rospy.Subscriber(f'/{self.veh_name}/front_center_tof_driver_node/range', Range, self.cb_range_finder, queue_size = 1)
        self.sub_camera = rospy.Subscriber(f'/{self.veh_name}/camera_node/image/compressed', CompressedImage, self.cb_duckiewalk_detector, queue_size=1, buff_size="20MB")
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

    def cb_duckiewalk_detector(self, image_msg):
        img = self.jpeg.decode(image_msg.data)
        crop = img[200:-1, :, :]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (90,87,100), (142,255,255))
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area
        
        if max_idx != -1:
            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if cy > self.duckwalk_stop_distance:
                    self.process_duckwalk = True
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass
        if DEBUG:
            rect_img_msg = CompressedImage(format='jpeg', data=self.jpeg.encode(crop))
            self.pub_duckiewalk.publish(rect_img_msg)

        self.rate.sleep()


    def cb_lane_pose(self, input_pose_msg):
        self.lane_pose = input_pose_msg
        self.get_control_action()

    def cb_duckie_detected(self, duckie_detected_msg):
        self.duckie_detected = duckie_detected_msg.data

    def cb_range_finder(self, range_msg):
        self.vehicle_distance = range_msg.range

    def cb_tag_id(self, tag_msg):

        # New action coming.
        if tag_msg.data != -1 and tag_msg.data != self.tag_id:
            print("tag_msg",tag_msg)
            self.tag_id = tag_msg.data
            self.action_done = False


    ## end of callback functions

    def vehicle_ahead(self):
        if self.tag_id == 163 and self.vehicle_distance < self.safe_distance:
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
            else:
                print("Going straight")
                self.go_straight()

            self.process_intersection = False
        elif self.process_duckwalk and self.tag_id in [21, 163]:
            v = 0.0
            omega = 0.0
            self.car_cmd(v, omega)
            rospy.sleep(3.0)
            print("Doing duckwalk")
            if self.duckie_detected:
                print("duckie detected")
                rospy.sleep(2.0)
                self.car_cmd(v=0.0, omega=0.0)
            else:
                self.car_cmd(v=0.25, omega=0.0)
                rospy.sleep(2)
                self.process_duckwalk = False
                self.lane_pid_controller.reset_controller()

        # elif self.vehicle_ahead() and not self.english_driver:
        #     v = 0.0
        #     omega = 0.0
        #     print("Vehicle detected")
        #     self.car_cmd(v, omega)
        #     rospy.sleep(3.0)

        #     self.english_driver_time = curr_time
        #     self.english_driver = True
        #     self.set_english_driver(True)
            
        elif self.english_driver and english_driver_time_diff > self.english_drive_cooldown:
            self.set_english_driver(False)
            self.english_driver = False
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
        print("Doing duckwalk")
        while self.duckie_detected:
            print("duckie detected")
            rospy.sleep(2.0)
            self.car_cmd(v=0.0, omega=0.0)
        
        self.car_cmd(v=0.25, omega=0.0)
        rospy.sleep(2)
        self.process_duckwalk = False
        self.lane_pid_controller.reset_controller()

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
            self.d_offset = 0.0

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
    rospy.spin()