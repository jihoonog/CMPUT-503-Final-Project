#!/bin/bash

echo "Running exercise 5"
BOT=csc22945

dts duckiebot demo --demo_name lane_following --duckiebot_name $BOT --package_name duckietown_demos --image duckietown/dt-core:daffy-arm64v8
# dts duckiebot demo --demo_name lane_filter --duckiebot_name $BOT --package_name duckietown_demos --image duckietown/dt-core:daffy-arm64v8

