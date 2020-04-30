#!/bin/bash
#Replace the variables with your github repo url, repo name, test
video name, json named by your UIN
GIT_REPO_URL="https://github.tamu.edu/song-3134/CSCE689.git"
REPO="CSCE689"

# The absolute path of the directory that contains all the testing videos (.mp4 format)
VIDEO_DIRECTORY_PATH="./video"

# The absolute path of the pretrained model that contains two pretrained neural networks
# make sure you download them from Google Drive
PRETRAIN_DIRECTORY_PATH="/data/qq/CSCE689/pretrain"

# Model to be tested (select from: hw4, hw5, hw6, hw8, final)
MODEL_NAME="final"


# You may need to install the required packages by yourself, I do not include the requirement.txt here.
git clone $GIT_REPO_URL
cd $REPO
python final_test.py --video-directory-path $VIDEO_DIRECTORY_PATH --model $MODEL_NAME --pretrain-directory-path $PRETRAIN_DIRECTORY_PATH
cp ./final_test_results/* .

