## 🛠️ Installation
We need to install some package to launch the files in here.
```bash
# activate environment
conda activate Uni-Sign
# install other relevant dependencies
pip install onnxruntime-gpu cuda-toolkit
cd ./demo/rtmlib-main
pip install -e .
```

## 🦴 Pose Extraction
```bash
# cd to root workspace
cd Uni-Sign
# pose extraction
# Note: Please specify the paths for {video_dir} and {pose_dir} before running command
# The {video_dir} directory contains multiple .mp4 files.
python ./demo/pose_extraction.py \
    --src_dir {video_dir} \
    --tgt_dir {pose_dir}
```

## ✈️ Online Inference
```bash
# cd to root workspace
cd Uni-Sign
# online inference, we provide two mode here
# Note: Please specify the video path for {video_path} before running command
   
# Mode1: Pose-only setting
ckpt_path=pt/csl_daily_pose_only_slt.pth

python ./demo/online_inference.py \
   --online_video {video_path} \
   --finetune {ckpt_path}
   
# Mode2: RGB-pose setting
ckpt_path=./demo/pt/csl_daily_rgb_pose_slt.pth
video_path = ./data/exam/
python ./demo/online_inference.py \
   --online_video {video_path} \
   --finetune {ckpt_path} \
   --rgb_support
```
