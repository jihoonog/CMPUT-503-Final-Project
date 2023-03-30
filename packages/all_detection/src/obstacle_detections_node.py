#!/usr/bin/env python3
import rospy
import numpy as np
from duckietown_msgs.msg import Twist2DStamped, LanePose, SegmentList, Segment
from geometry_msgs.msg import Point
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from std_msgs.msg import String
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

import os



class ObstacleDetectionNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ObstacleDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        if os.environ["VEHICLE_NAME"] is not None:
            self.veh = os.environ["VEHICLE_NAME"]
        else:
            self.veh = "csc22935"
        self.rospack = rospkg.RosPack()
        self.pub_car_cmd = rospy.Publisher(f"/{self.veh}/car_cmd", Twist2DStamped, queue_size=10)
        self.pub_image = rospy.Publisher(f"/{self.veh}/image_with_object", Image, queue_size=1)

        self.bridge = CvBridge()

        self.stop = False
        self.sub_image = rospy.Subscriber( f'/{self.veh}/camera_node/image/compressed',CompressedImage,self.processImage, queue_size=1)
        #rospy.Subscriber("~corrected_image/compressed", CompressedImage, self.processImage, queue_size=1)
        self.sub_filtered_seglist = rospy.Subscriber(f"/{self.veh}/lane_filter_node/seglist_filtered", SegmentList, self.filtered_seglist_cb)
        self.upper_bound = -1

    def filtered_seglist_cb(self, seglist_msg):
        #print("In",filtered_seglist_cb)
        
            
        #initialize variables
        yellow_offset, white_offset, omega_gain = -0.12, 0.15, 2
        white_seg_count, yellow_seg_count = 0, 0
        white_x_accumulator, white_y_accumulator, yellow_x_accumulator, yellow_y_accumulator = 0.0, 0.0, 0.0, 0.0
        white_centroid_x, white_centroid_y, yellow_centroid_x, yellow_centroid_y = 0.0, 0.0, 0.0, 0.0

        for segment in seglist_msg.segments:
            #the point is behind us
            if segment.points[0].x < 0 or segment.points[1].x < 0: 
                continue

            #calculate white segments sum, count values
            if segment.color == segment.WHITE:
                white_x_accumulator += (segment.points[0].x + segment.points[1].x) / 2
                white_y_accumulator += (segment.points[0].y + segment.points[1].y) / 2 
                white_seg_count += 1.0
            #calculate yellow segments sum, count values
            elif segment.color == segment.YELLOW:
                yellow_x_accumulator += (segment.points[0].x + segment.points[1].x) / 2
                yellow_y_accumulator += (segment.points[0].y + segment.points[1].y) / 2 
                yellow_seg_count += 1.0
            #skip red segments
            else:
                continue

        #calculate centroid for white segments
        if white_seg_count > 0:
            white_centroid_x, white_centroid_y = white_x_accumulator/white_seg_count, white_y_accumulator/white_seg_count

        #calculate centroid for yellow segments
        if yellow_seg_count > 0:
            yellow_centroid_x, yellow_centroid_y = yellow_x_accumulator/yellow_seg_count, yellow_y_accumulator/yellow_seg_count

        #if white seg count is greater, trust white line segments
        if  white_seg_count >  yellow_seg_count:   
            follow_point_x = white_centroid_x
            follow_point_y = white_centroid_y + white_offset

        #if yellow seg count is greater, trust yellow line segments
        elif  yellow_seg_count > white_seg_count:  
            follow_point_x = yellow_centroid_x
            follow_point_y = yellow_centroid_y + yellow_offset
        
        #if both are equal, take average
        else:
            follow_point_x = 0.5 * (white_centroid_x + yellow_centroid_x)
            follow_point_y = 0.5 * (white_centroid_y + yellow_centroid_y)
            #check if they are zero, because they might become zero if no white/yellow segments are encountered
            if follow_point_x == 0 and follow_point_y == 0:
                follow_point_x, follow_point_y = 0.1, 0

        #tan_alpha = y/x => alpha = tan-1(y/x)
        alpha = np.arctan2(follow_point_y, follow_point_x)
        lookahead_dist = np.sqrt(follow_point_x * follow_point_x + follow_point_y * follow_point_y)
        #calculating v, omega

        
        if np.abs(follow_point_y) > 0.2:
            v, omega_gain = 0.25, 3
        else:
            if np.abs(follow_point_x) >= 0.55:
                v, omega_gain = 0.7, 1.5
            elif np.abs(follow_point_x) > 0.48 and np.abs(follow_point_x) < 0.55:
                v, omega_gain = 0.4, 1.5
            else:
                v, omega_gain = 0.25, 2

        omega  =  2 * v * np.sin(alpha) / lookahead_dist        

        #publishing to car_cmd topic
        car_control_msg = Twist2DStamped()
        if self.stop is True:
            car_control_msg.v = 0
            car_control_msg.omega = 0
            print("Sending V= 0, omega = 0")
        else:
            car_control_msg.v = v
            car_control_msg.omega = omega * omega_gain
            print('sending: v, omega', v, omega)
        #pub_car_cmd.publish(car_control_msg)

        #cali_file_folder = self.rospack.get_path('all_detection')+'/config/calibrations/camera_extrinsic/'
#self.cali_file = self.rospack.get_path('all_detection') + f"/config/calibrations/camera_intrinsic/{self.veh}.yaml"
    def readYamlFile(self):
        fname = self.rospack.get_path('all_detection') + f"/config/calibrations/camera_extrinsic/{self.veh}.yaml"
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)["homography"]
                return np.array(yaml_dict).reshape((3,3))
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return

    def load_homography(self):
        '''Load homography (extrinsic parameters)'''
        

        rospy.loginfo("Using extrinsic calibration of " + self.veh)
        data = yaml_load_file(filename)
        #print("data",data)
        logger.info("Loaded homography for {}".format(os.path.basename(filename)))
        return np.array(data['homography']).reshape((3,3))

    def point2ground(self, x_arr, y_arr, norm_x, norm_y):
            new_x_arr, new_y_arr = [], []
            H = self.readYamlFile()
            for i in range(len(x_arr)):
                u = x_arr[i] * 480/norm_x
                v = y_arr[i] * 640/norm_y
                uv_raw = np.array([u, v])
                uv_raw = np.append(uv_raw, np.array([1]))
                ground_point = np.dot(H, uv_raw)
                point = Point()
                x = ground_point[0]
                y = ground_point[1]
                z = ground_point[2]
                point.x = x/z
                point.y = y/z
                point.z = 0.0
                new_x_arr.append(point.x)
                new_y_arr.append(point.y)
            return new_x_arr, new_y_arr



    def processImage(self, image_msg):
        #print("in processimage")

        image_size = [480,640]
        # top_cutoff = 40

        start_time = time.time()
        try:
            image_cv = bgr_from_jpg(image_msg.data)
        except ValueError as e:
            print("image decode error", e)
            return
        
        # Resize and crop image
        hei_original, wid_original = image_cv.shape[0:2]

        if image_size[0] != hei_original or image_size[1] != wid_original:
            image_cv = cv2.resize(image_cv, (image_size[1], image_size[0]),
                                    interpolation=cv2.INTER_NEAREST)

        hsv = cv.cvtColor(image_cv, cv.COLOR_BGR2HSV)
        # hsv_obs_red1 = np.array([0,140, 100])
        # hsv_obs_red2 = np.array([15,255,255])
        hsv_obs_red3 = np.array([10,116, 180]) 
        hsv_obs_red4 = np.array([25,136,260])

        
        #hsv_obs_red3 =  np.array([110,50,50])#np.array([165,140, 100]) 
        #hsv_obs_red4 = np.array([130,255,255])#np.array([180,255,255])
        #lower_blue = np.array([110,50,50])
        #upper_blue = np.array([130,255,255])

        bw1 = cv.inRange(hsv, hsv_obs_red3, hsv_obs_red4)
        #bw2 = cv.inRange(hsv, hsv_obs_red3, hsv_obs_red4)
        bw = bw1#cv.bitwise_or(bw1, bw2)
        cnts = cv2.findContours(bw.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(cnts)>1:
            #print('object detected')
            red_area = max(cnts, key=cv2.contourArea)
            (xg,yg,wg,hg) = cv2.boundingRect(red_area)
            if yg<120 or yg > 400:

                return

            box_img = cv2.rectangle(image_cv,(xg,yg),(xg+wg, yg+hg),(0,255,0),2)
            #print(xg,yg,wg,hg)
            #print('BEFORE X', [xg, xg+wg], " BEFORE Y", [yg+hg, yg+hg])
            x_arr, y_arr = self.point2ground([xg, xg+wg], [yg + hg, yg + hg], image_size[0], image_size[1])
            #print("BOTTOM OF ROBOT : X ", x_arr, ' Y :', y_arr)
            if x_arr[0] < 0.35:
                print("STOP THE Bot")
                self.stop = True
            else:
                self.stop = False
            image_msg_out = self.bridge.cv2_to_imgmsg(box_img, "bgr8")
            image_msg_out.header.stamp = image_msg.header.stamp
            self.pub_image.publish(image_msg_out)
        else:
            self.stop = False
            image_msg_out = self.bridge.cv2_to_imgmsg(image_cv, "bgr8")
            image_msg_out.header.stamp = image_msg.header.stamp
            self.pub_image.publish(image_msg_out)
        
        #print('Time to process', time.time() - start_time)

if __name__ == "__main__":
    #defining node, publisher, subscriber
    #rospy.init_node("purepursuit_controller_node", anonymous=True)
    obs_detection_node = ObstacleDetectionNode(node_name="obstacle_detections_node")

    rospy.spin()

    
