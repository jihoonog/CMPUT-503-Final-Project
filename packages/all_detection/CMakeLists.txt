cmake_minimum_required(VERSION 2.8.3)
project(all_detection)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  duckietown_msgs
  roscpp
  rospy
)


catkin_package()

