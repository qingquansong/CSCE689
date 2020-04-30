#!/bin/bash
#Replace the variables with your github repo url, repo name, test
video name, json named by your UIN
GIT_REPO_URL="https://github.tamu.edu/song-3134/CSCE689.git"
REPO="CSCE689"
git clone $GIT_REPO_URL

cd $REPO
python final_test.py --video-file-path "" --model hw6 --pretrain-file-path ""
cp ./final_test_results/* .

