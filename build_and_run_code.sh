#!/bin/bash
set -e
set -x

echo "Building and running final protect"


# cross compile the code on the laptop
dts devel build --loop --arch arm64v8


# Run the code on the bot
dts devel run -H $BOT

