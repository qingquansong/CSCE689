#!/bin/bash
#Replace the variables with your github repo url, repo name, test
video name, json named by your UIN
GIT_REPO_URL="https://github.tamu.edu/song-3134/CSCE689.git"
REPO="CSCE689"
git clone $GIT_REPO_URL

cd $REPO
python final_test.py --video-file-path /data/qq/CSCE689/video/jiang --model hw6 --pretrain-file-path /data/qq/CSCE689/pretrain
cp ./final_test_results/* .

