#!/usr/bin/env python3
import rospy
import numpy as np
from duckietown_msgs.msg import Twist2DStamped, LanePose, SegmentList, Segment
from geometry_msgs.msg import Point
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from std_msgs.msg import String,Int32
from sensor_msgs.msg import CompressedImage, Image
from duckietown_utils.jpg import bgr_from_jpg
import cv2
import cv2 as cv
import time
import rospkg 
from cv_bridge import CvBridge, CvBridgeError
import yaml
import duckietown_utils as dtu
from duckietown_utils import (logger)
from duckietown_utils.yaml_wrap import (yaml_write_to_file)

from turbojpeg import TurboJPEG 

import os


DEBUG = True


class ObstacleDetectionNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ObstacleDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        if os.environ["VEHICLE_NAME"] is not None:
            self.veh = os.environ["VEHICLE_NAME"]
        else:
            self.veh = "csc22935"
        self.pub_image = rospy.Publisher(f"/{self.veh}/obj_detection/image/compressed", CompressedImage, queue_size=1)

        self.bridge = CvBridge()
        self.jpeg = TurboJPEG()
        self.rate = rospy.Rate(15)

        self.pub_duckie_detected = rospy.Publisher(f'/{self.veh}/duckie_detected', Int32, queue_size=1)
        self.sub_image = rospy.Subscriber( f'/{self.veh}/camera_node/image/compressed',CompressedImage,self.processImage, queue_size=1)
        #rospy.Subscriber("~corrected_image/compressed", CompressedImage, self.processImage, queue_size=1)
        self.upper_bound = -1


    def processImage(self, image_msg):
        image_size = [480,640]
        # top_cutoff = 40

        start_time = time.time()
        try:
            image_cv = self.jpeg.decode(image_msg.data)
        except ValueError as e:
            print("image decode error", e)
            return
        # Crop for both the duckiewalk and duckie
        duckiewalk_crop = image_cv[200:-1, 213:427, :]
        duckie_crop = image_cv[200:-1, 213:427, :]
        duckiewalk_hsv = cv2.cvtColor(duckiewalk_crop, cv2.COLOR_BGR2HSV)
        duckie_hsv = cv2.cvtColor(duckiewalk_crop, cv2.COLOR_BGR2HSV)
        # There are specific for both the duckiewalk and duckie
        duckiewalk_mask = cv2.inRange(duckiewalk_hsv, (90,87,100), (142,255,255))
        duckie_mask = cv2.inRange(duckie_hsv, (10,55,100), (45,255,255))
        duckiewalk_crop = cv2.bitwise_and(duckiewalk_crop, duckiewalk_crop, mask=duckiewalk_mask)
        duckie_crop = cv2.bitwise_and(duckie_crop, duckie_crop, mask=duckie_mask)
        line_contour, _ = cv2.findContours(duckiewalk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        duckie_contour, _ = cv2.findContours(duckie_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        duckie_max_area = 1500
        line_max_area = 50

        line_idx = -1
        duckie_idx = -1
        for i in range(len(line_contour)):
            area = cv2.contourArea(line_contour[i])
            if area > line_max_area:
                line_max_area = area
                line_idx = i

        for i in range(len(duckie_contour)):
            area = cv2.contourArea(duckie_contour[i])
            if area > duckie_max_area:
                duckie_max_area = area
                duckie_idx = i

        if line_idx != -1:
            M = cv2.moments(line_contour[line_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                print(cy)
                if DEBUG:
                    cv2.drawContours(duckiewalk_crop, line_contour, max_idx, (0, 255, 0), 3)
                    cv2.circle(duckiewalk_crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass

        if duckie_idx != -1:
            print("Duckie exist")

        if DEBUG:
            rect_img_msg = CompressedImage(format='jpeg', data=self.jpeg.encode(duckiewalk_crop)) 
            self.pub_image.publish(rect_img_msg)           

        self.rate.sleep()

if __name__ == "__main__":
    #defining node, publisher, subscriber
    #rospy.init_node("purepursuit_controller_node", anonymous=True)
    obs_detection_node = ObstacleDetectionNode(node_name="obstacle_detections_node")

    rospy.spin()

    
