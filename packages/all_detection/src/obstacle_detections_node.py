#!/usr/bin/env python3
import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage, Image
import cv2
from cv_bridge import CvBridge

from turbojpeg import TurboJPEG 

import os


DEBUG = False


class ObstacleDetectionNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ObstacleDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        if os.environ["VEHICLE_NAME"] is not None:
            self.veh = os.environ["VEHICLE_NAME"]
        else:
            self.veh = "csc22935"

        self.bridge = CvBridge()
        self.jpeg = TurboJPEG()
        self.rate = rospy.Rate(20)

        # Set the threshold of how far the blue line has to be from the top of the image to trigger a stop
        self.duckwalk_threshold = 75
        ## parameters for the blob detector 
        self.blobdetector_min_area = 10
        self.blobdetector_min_dist_between_blobs = 2

        self.set_parameters()


        ## Setup subscribers
        self.sub_image = rospy.Subscriber( f'/{self.veh}/camera_node/image/compressed',CompressedImage,self.processImage, queue_size=1)

        ## Setup publishers 
        self.pub_duckie_detected = rospy.Publisher(f'/{self.veh}/all_detection/duckie_detected', Bool, queue_size=1)
        self.pub_duckwalk_detected = rospy.Publisher(f'/{self.veh}/all_detection/duckwalk_detected', Bool, queue_size=1)
        self.pub_duckiebot_detected = rospy.Publisher(f'/{self.veh}/all_detection/duckiebot_detected', Bool, queue_size=1)
        
        self.pub_debug_duckwalk_image = rospy.Publisher(f"/{self.veh}/obj_detection/duckwalk_detection/image/compressed", CompressedImage, queue_size=1)
        self.pub_debug_duckie_image = rospy.Publisher(f"/{self.veh}/obj_detection/duckie_detection/image/compressed", CompressedImage, queue_size=1)
        self.pub_debug_duckiebot_image = rospy.Publisher(f"/{self.veh}/obj_detection/duckiebot_detection/image/compressed", CompressedImage, queue_size=1)
    
    def set_parameters(self):
        params = cv2.SimpleBlobDetector_Params()
        params.minArea = self.blobdetector_min_area
        params.minDistBetweenBlobs = self.blobdetector_min_dist_between_blobs
        self.simple_blob_detector = cv2.SimpleBlobDetector_create(params)


    def detect_duckiebot_tag(self, image):
        duckiebot_detected = False

        # Grid circle detector
        (detection, centers) = cv2.findCirclesGrid(
            image,
            patternSize=tuple([7,3]),
            flags=cv2.CALIB_CB_SYMMETRIC_GRID,
            blobDetector=self.simple_blob_detector,
        )

        if detection > 0:
            duckiebot_detected = True

            if True:
                cv2.drawChessboardCorners(image, tuple([7,3]), centers, detection)
                image_msg = CompressedImage(format='jpeg', data=self.jpeg.encode(image)) 
                self.pub_debug_duckiebot_image.publish(image_msg)

        return duckiebot_detected            

    def detect_duckwalk(self, image):
        duckwalk_detected = False

        duckiewalk_crop = image[200:-1, 213:427, :]
        duckiewalk_hsv = cv2.cvtColor(duckiewalk_crop, cv2.COLOR_BGR2HSV)
        duckiewalk_mask = cv2.inRange(duckiewalk_hsv, (90,87,100), (142,255,255))
        duckiewalk_crop = cv2.bitwise_and(duckiewalk_crop, duckiewalk_crop, mask=duckiewalk_mask)
        line_contour, _ = cv2.findContours(duckiewalk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        line_max_area = 100
        line_max_idx = -1
        for i in range(len(line_contour)):
            area = cv2.contourArea(line_contour[i])
            if area > line_max_area:
                line_max_area = area
                line_max_idx = i

        if line_max_idx != -1:
            M = cv2.moments(line_contour[line_max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                if cy > self.duckwalk_threshold:
                    duckwalk_detected = True
                
                if DEBUG:
                    # print(cy)
                    cv2.drawContours(duckiewalk_crop, line_contour, line_max_idx, (0, 255, 0), 3)
                    cv2.circle(duckiewalk_crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass

        if DEBUG:
            rect_img_msg = CompressedImage(format='jpeg', data=self.jpeg.encode(duckiewalk_crop)) 
            self.pub_debug_duckwalk_image.publish(rect_img_msg)         

        return duckwalk_detected
    
    def detect_duckie(self, image):
        duckie_detected = False

        # Crop for both the duckiewalk and duckie
        duckie_crop = image[150:-1, 200:450, :]
        # Conver them to HSV
        duckie_hsv = cv2.cvtColor(duckie_crop, cv2.COLOR_BGR2HSV)
        # There are specific for both the duckiewalk and duckie
        duckie_mask = cv2.inRange(duckie_hsv, (10,55,100), (45,255,255))
        # Apply the mask
        duckie_crop = cv2.bitwise_and(duckie_crop, duckie_crop, mask=duckie_mask)
        # Get the contours
        duckie_contour, _ = cv2.findContours(duckie_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        duckie_max_area = 1500
        duckie_max_idx = -1

        for i in range(len(duckie_contour)):
            area = cv2.contourArea(duckie_contour[i])
            if area > duckie_max_area:
                duckie_max_area = area
                duckie_max_idx = i

        if duckie_max_idx != -1:
            duckie_detected = True
            if DEBUG:
                cv2.drawContours(duckie_crop, duckie_contour, duckie_max_idx, (0, 255, 0), 3)
                rect_img_msg = CompressedImage(format='jpeg', data=self.jpeg.encode(duckie_crop))
                self.pub_debug_duckie_image.publish(rect_img_msg)

        return duckie_detected
    
    def processImage(self, image_msg):

        try:
            image_cv = self.jpeg.decode(image_msg.data)
        except ValueError as e:
            print("image decode error", e)
            return
        
        duckie_detected = self.detect_duckie(image_cv)
        # print("Duckie detected" if duckie_detected else "No duckie")
        self.pub_duckie_detected.publish(duckie_detected)

        duckwalk_detected = self.detect_duckwalk(image_cv)
        # print("Duckwalk detected" if duckwalk_detected else "No duckwalk")
        self.pub_duckwalk_detected.publish(duckwalk_detected)

        duckiebot_detected = self.detect_duckiebot_tag(image_cv)
        # print("Duckiebot detected" if duckiebot_detected else "No duckiebot")
        self.pub_duckiebot_detected.publish(duckiebot_detected)

        self.rate.sleep()

if __name__ == "__main__":
    obs_detection_node = ObstacleDetectionNode(node_name="obstacle_detections_node")
    rospy.spin()

    
