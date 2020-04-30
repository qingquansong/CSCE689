#!/bin/bash
#Replace the variables with your github repo url, repo name, test
video name, json named by your UIN
GIT_REPO_URL="https://github.tamu.edu/song-3134/CSCE689.git"
# REPO="openpose"
VIDEO_FILE_PATH="video"
UIN_JSON="625007598.json"
UIN_JPG="625007598.jpg"
git clone $GIT_REPO_URL

# cd $REPO
#Replace this line with commands for running your test python file.

# echo $VIDEO_FILE_PATH
# python test.py --video_name $VIDEO

#If your test file is ipython file, uncomment the following lines and
# replace IPYTHON_NAME with your test ipython file.
IPYTHON_NAME="final_test.ipynb"
echo $IPYTHON_NAME
jupyter notebook
#rename the generated timeLabel.json and figure with your UIN.
cp timeLable.json $UIN_JSON
cp timeLable.jpg $UIN_JPG

