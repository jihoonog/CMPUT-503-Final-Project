#!/usr/bin/env python3
import os
import rospy
import cv2
import yaml
import numpy as np
from nav_msgs.msg import Odometry
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from sensor_msgs.msg import CompressedImage, CameraInfo
from augmented_reality_basics import Augmenter
from cv_bridge import CvBridge
from duckietown_utils import load_homography, load_map,get_duckiefleet_root
import rospkg 
from dt_apriltags import Detector, Detection
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, WheelsCmdStamped, LEDPattern, AprilTagDetection, AprilTagDetectionArray
from duckietown_msgs.srv import SetCustomLEDPattern, ChangePattern
# Code from https://github.com/Coral79/exA-3/blob/44adf94bad728507608086b91fbf5645fc22555f/packages/augmented_reality_basics/include/augmented_reality_basics/augmented_reality_basics.py
# https://docs.photonvision.org/en/latest/docs/getting-started/pipeline-tuning/apriltag-tuning.html
import math
import geometry_msgs.msg
import tf
from std_msgs.msg import Header, Float32, String, Float64MultiArray, Float32MultiArray, Int32
import tf2_ros


def argmin(lst):
    tmp = lst.index(min(lst))

    return tmp



class AprilTagNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(AprilTagNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        if os.environ["VEHICLE_NAME"] is not None:
            self.veh = os.environ["VEHICLE_NAME"]
        else:
            self.veh = "csc22945"
        self.rospack = rospkg.RosPack()
        self.read_params_from_calibration_file()
        # Get parameters from config
        self.camera_info_dict = self.load_intrinsics()
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.tmp_broadcaster = tf2_ros.StaticTransformBroadcaster()

        self.augmenter = Augmenter(self.homography,self.camera_info_msg)    
        # Subscribing 
        self.sub_image = rospy.Subscriber( f'/{self.veh}/camera_node/image/compressed',CompressedImage,self.project, queue_size=1)
        
        # Publisher
        # Keep this state so you don't need to reset the same color over and over again.
        self.pub_tag_id = rospy.Publisher(f'/{self.veh}/tag_id', Int32, queue_size=1)
        self.pub_all_tag_poses = rospy.Publisher(f'/{self.veh}/all_tag_poses', AprilTagDetectionArray, queue_size=1)
        self.current_led_pattern = 4

        self.frequency_control = 0
        
        # extract parameters from camera_info_dict for apriltag detection
        f_x = self.camera_info_dict['camera_matrix']['data'][0]
        f_y = self.camera_info_dict['camera_matrix']['data'][4]
        c_x = self.camera_info_dict['camera_matrix']['data'][2]
        c_y = self.camera_info_dict['camera_matrix']['data'][5]
        self.camera_params = [f_x, f_y, c_x, c_y]
        K_list = self.camera_info_dict['camera_matrix']['data']
        self.K = np.array(K_list).reshape((3, 3))

        # initialise the apriltag detector
        self.at_detector = Detector(searchpath=['apriltags'],
                           families='tag36h11',
                           nthreads=3,
                           quad_decimate=2,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)

        self.r = rospy.Rate(20) # 30hz

        #rospy.init_node('static_tf2_broadcaster_tag')
        self.buffer = tf2_ros.Buffer()

        self.buffer_listener = tf2_ros.TransformListener(self.buffer)
        self.sub_shutdown_commands = rospy.Subscriber(f'/{self.veh}/all_detection_node/shutdown_cmd', String, self.shutdown, queue_size = 1)
        
        self.get_pose = False

        self.fusion_x = 0
        self.fusion_y = 0
        self.fusion_z = 0
        self.br = CvBridge()

        self.fusion_rotation_z = 0



    def transform_camera_view(self,pose_t,pose_R):
        # print("pose_t",pose_t)
        # print("pose_R",pose_R)
        static_transformStamped = geometry_msgs.msg.TransformStamped()

        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = f"{self.veh}/camera_optical_frame"
        static_transformStamped.child_frame_id = f"{self.veh}/new_location"

        static_transformStamped.transform.translation.x = float(pose_t[0][0])
        static_transformStamped.transform.translation.y = float(pose_t[1][0])
        static_transformStamped.transform.translation.z = float(pose_t[2][0])

        
        yaw = math.atan2(pose_R[1][0], pose_R[0][0])
        pitch = math.atan2(-pose_R[2][0], math.sqrt(pose_R[2][1]**2+pose_R[2][2]**2))
        roll = math.atan2(pose_R[2][1], pose_R[2][2])

        quat = tf.transformations.quaternion_from_euler(roll,pitch,yaw)

        
        static_transformStamped.transform.rotation.x = quat[0]
        static_transformStamped.transform.rotation.y = quat[1]
        static_transformStamped.transform.rotation.z = quat[2]
        static_transformStamped.transform.rotation.w = quat[3]
        #print("static_transformStamped",static_transformStamped)
        self.broadcaster.sendTransform(static_transformStamped)



    def shutdown(self, msg):
        if msg.data=="shutdown":
            rospy.signal_shutdown("Apriltag detection Node Shutdown command received")
            exit()


    def project(self, msg):        
        # Convert image to cv2 image.
        self.raw_image = self.br.compressed_imgmsg_to_cv2(msg)
        # Convert to grey image and distort it.
        dis = self.augmenter.process_image(self.raw_image)
        new_img = dis

        if self.frequency_control % 5 == 0:
            tags = self.at_detector.detect(dis, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=0.065) # returns list of detection objects

            # print("pre stage 3:", tags)
            detection_threshold = 10 # The target margin need to be larger than this to get labelled.
            # print("here")
            print("tags:",tags)
            if len(tags) == 0:
                # Means there's no tags present. Set the led to white.
                self.pub_tag_id.publish(-1)    
                #pass
            else:
                margin_list = []
                tag_list = []
                max_margin = 10
                best_tag_id = 0
                the_tag = None
                min_distance = 1000
                for tag in tags:
                    # print("tags:", tag)
                    if tag.decision_margin > max_margin:
                        if tag.pose_t[2][0] < min_distance:
                            min_distance = tag.pose_t[2][0]
                            the_tag = tag
                            best_tag_id = tag.tag_id

                print(best_tag_id)
                self.pub_tag_id.publish(best_tag_id)
                
                
                if best_tag_id in [227, 38]:
                    self.get_pose  = True

            self.frequency_control += 1

            # print("Stage 3 tags:", tags)
            # tags_msg = AprilTagDetectionArray()
            # tags_msg.header.stamp = msg.header.stamp
            # tags_msg.header.frame_id = msg.header.frame_id

            # for tag in tags:

            #     if tag.tag_id not in [38, 75, 207, 226, 227, 228]:
            #         continue
            #     self.transform_camera_view(tag.pose_t,tag.pose_R)
            #     trans = self.buffer.lookup_transform(f"{self.veh}/footprint",f"{self.veh}/new_location",time=rospy.Time.now(),timeout=rospy.Duration(1.0))
            #     # print(trans)
            #     detection = AprilTagDetection(
            #         transform=trans.transform,
            #         tag_id=tag.tag_id,
            #         hamming=tag.hamming,
            #         decision_margin=tag.decision_margin,
            #         homography=tag.homography.flatten().astype(np.float32).tolist(),
            #         center=tag.center.tolist(),
            #         corners=tag.corners.flatten().tolist(),
            #         pose_error=tag.pose_err,
            #     )
            #     tags_msg.detections.append(detection)
            # # print("tagsmsg:",tags_msg)
            # self.pub_all_tag_poses.publish(tags_msg)

        #self.r.sleep()

    def read_params_from_calibration_file(self):
        # Get static parameters
        file_name_ex = self.get_extrinsic_filepath(self.veh)
        self.homography = self.readYamlFile(file_name_ex)
        self.camera_info_msg = rospy.wait_for_message(f'/{self.veh}/camera_node/camera_info', CameraInfo)


    def get_extrinsic_filepath(self,name):
        #TODO: retrieve the calibration info from the right path.
        cali_file_folder = self.rospack.get_path('all_detection')+'/config/calibrations/camera_extrinsic/'

        cali_file = cali_file_folder + name + ".yaml"
        return cali_file

    def readYamlFile(self,fname):
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)["homography"]
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return

    def readYamlFile2(self,fname):
        """
            Reads the 'fname' yaml file and returns a dictionary with its input.

            You will find the calibration files you need in:
            `/data/config/calibrations/`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file, Loader=yaml.Loader)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown('No calibration file found.')
                return

    def load_intrinsics(self):
        # Find the intrinsic calibration parameters
        # cali_file_folder = '/data/config/calibrations/camera_intrinsic/'
        # self.frame_id = self.veh + '/camera_optical_frame'
        # self.cali_file = cali_file_folder + self.veh + ".yaml"

        self.cali_file = self.rospack.get_path('all_detection') + f"/config/calibrations/camera_intrinsic/{self.veh}.yaml"

        # Locate calibration yaml file or use the default otherwise
        rospy.loginfo(f'Looking for calibration {self.cali_file}')
        if not os.path.isfile(self.cali_file):
            self.logwarn("Calibration not found: %s.\n Using default instead." % self.cali_file)
            self.cali_file = (cali_file_folder + "default.yaml")

        # Shutdown if no calibration file not found
        if not os.path.isfile(self.cali_file):
            rospy.signal_shutdown("Found no calibration file ... aborting")

        # Load the calibration file
        calib_data = self.readYamlFile2(self.cali_file)
        self.log("Using calibration file: %s" % self.cali_file)

        return calib_data

if __name__ == '__main__':
    augmented_reality_basics_node = AprilTagNode(node_name='apriltag_node')
    # Keep it spinning to keep the node alive
    rospy.spin()