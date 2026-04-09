#!/bin/bash

# YuJian FastAPI 服务启动脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 模型配置
export CHECKPOINT_PATH="./demo/pt/csl_daily_pose_only_slt.pth"
export DEVICE="cpu"  # 或 "cuda"
export RGB_SUPPORT="false"  # 是否启用 RGB 辅助分支
export MAX_LENGTH="256"
export DATASET="CSL_Daily"  # 或 "CSL_News", "How2Sign", "OpenASL"

# 视频帧率配置
# 设置为 null 或空值表示不采样，使用所有帧
# 设置为数字表示目标 FPS，如 "5.0" 表示每秒采 5 帧
export TARGET_FPS=""

# 语音合成配置
export ENABLE_TTS="true"  # 是否启用语音合成功能 (true/false)

# EmoLLM 心理咨询模型配置
export ENABLE_EMOLLM="true"  # 是否启用 EmoLLM 功能 (true/false)
export EMOLLM_MODEL_PATH="/home/xtk/models/haiyangpengai/careyou_7b_16bit_v3_2_qwen14_4bit"
export EMOLLM_TOKENIZER_PATH="/home/xtk/qwen_tokenizer/qwen/Qwen2-7B-Instruct"

# 启动服务
echo "启动 YuJian 手语翻译服务..."
echo "检查点路径: $CHECKPOINT_PATH"
echo "设备: $DEVICE"
echo "数据集: $DATASET"
echo "启用TTS: $ENABLE_TTS"

python -m uvicorn api:app --host 0.0.0.0 --port 6006 --workers 1
