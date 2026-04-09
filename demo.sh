#!/bin/bash

cd /root/autodl-tmp/liuyongjie/Uni-Sign || exit 1
export PYTHONPATH=$(pwd):$PYTHONPATH

ckpt_path=./demo/pt/csl_daily_rgb_pose_slt.pth
video_path=./data/exam/example_1.mp4

python ./demo/online_inference.py \
  --online_video "$video_path" \
  --finetune "$ckpt_path" \
  --rgb_support