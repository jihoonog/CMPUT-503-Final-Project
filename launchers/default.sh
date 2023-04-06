#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# launching app
dt-exec roslaunch all_detection all_detection.launch 
dt-exec roslaunch purepursuit purepursuit_controller.launch 
dt-exec roslaunch duckiebot_detection duckiebot_detection_node.launch
dt-exec roslaunch lane_follow lane_follow_node.launch
sleep 2
dt-exec roslaunch duckiebot_circuit duckiebot_circuit_node.launch

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
