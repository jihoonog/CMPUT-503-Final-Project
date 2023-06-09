#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32
from turbojpeg import TurboJPEG
import cv2
import numpy as np
import os
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
from std_msgs.msg import Bool

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
DEBUG = False
ENGLISH = False

class LaneFollowNode(DTROS):

    def __init__(self, node_name):
        super(LaneFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        if os.environ["VEHICLE_NAME"] is not None:
            self.veh = os.environ["VEHICLE_NAME"]
        else:
            self.veh = "csc22945"

        # Publishers & Subscribers
        self.pub = rospy.Publisher("/" + self.veh + "/output/image/mask/compressed",
                                   CompressedImage,
                                   queue_size=1)
        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.callback,
                                    queue_size=1,
                                    buff_size="20MB")
        self.sub_reset = rospy.Subscriber("/" + self.veh + "/lane_follow_node/reset",
                                            Bool,
                                            self.reset_callback,
                                            queue_size=1)
        
        self.sub_english = rospy.Subscriber("/" + self.veh + "/lane_follow_node/english",
                                            Bool,
                                            self.english_callback,
                                            queue_size=1)
        
        self.pub_vel_cmd = rospy.Publisher(f'/{self.veh}/lane_follow_node/car_cmd', Twist2DStamped, queue_size=1)

        self.jpeg = TurboJPEG()

        self.loginfo("Initialized")

        # PID Variables
        self.proportional = None
        if ENGLISH:
            self.offset = -230
        else:
            self.offset = 230
        self.velocity = 0.3
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        self.P = 0.0466
        self.D = -0.005
        self.I = 0.009
        self.last_error = 0
        self.last_time = rospy.get_time()

        # Wait a little while before sending motor commands
        rospy.Rate(0.20).sleep()


    def reset_callback(self, msg):
        boolean = msg.data
        if boolean:
            self.proportional = None
            self.last_error = 0
            self.last_time = rospy.get_time()

    def english_callback(self, msg):
        boolean = msg.data
        if boolean:
            self.offset = -240
        else:
            self.offset = 240

    def callback(self, msg):
        img = self.jpeg.decode(msg.data)
        crop = img[300:-1, :, :]
        crop_width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, hierarchy = cv2.findContours(mask,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)

        # Search for lane in front
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
                self.proportional = cx - int(crop_width / 2) + self.offset
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass
        else:
            self.proportional = None

        if DEBUG:
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
            self.pub.publish(rect_img_msg)

    def drive(self):
        if self.proportional is None:
            self.twist.omega = 0
            self.last_error = 0
        else:
            # P Term
            P = -self.proportional * self.P

            # D Term
            d_time = (rospy.get_time() - self.last_time)
            d_error = (self.proportional - self.last_error) / d_time
            self.last_error = self.proportional
            self.last_time = rospy.get_time()
            D = d_error * self.D

            # I Term
            I = -self.proportional * self.I * d_time

            self.twist.v = self.velocity
            self.twist.omega = P + I + D
            if DEBUG:
                self.loginfo(f"{self.proportional}, {P}, {D}, {self.twist.omega}, {self.twist.v}")

        self.pub_vel_cmd.publish(self.twist)


if __name__ == "__main__":
    node = LaneFollowNode("lanefollow_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        node.drive()
        rate.sleep()

