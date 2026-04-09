#!/bin/bash

# Uni-Sign API 启动脚本
# 支持：手语翻译 + ChatTTS语音合成 + EmoLLM心理咨询

# 配置
export CHECKPOINT_PATH="./demo/pt/csl_daily_pose_only_slt.pth"
export DEVICE="cuda"
export ENABLE_TTS="true"
export ENABLE_EMOLLM="true"
export EMOLLM_MODEL_PATH="/home/xtk/models/haiyangpengai/careyou_7b_16bit_v3_2_qwen14_4bit"
export EMOLLM_TOKENIZER_PATH="/home/xtk/qwen_tokenizer/qwen/Qwen2-7B-Instruct"
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=4

# 端口
PORT=6006

echo "========================================"
echo "  YuJian API 启动脚本"
echo "========================================"
echo ""

# 检查是否已在运行
PID=$(ps aux | grep "uvicorn.*api:app.*$PORT" | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "⚠️  服务已在运行 (PID: $PID)"
    echo "   如需重启，请先执行: ./stop.sh"
    echo ""
    echo "测试服务:"
    curl -s http://localhost:$PORT/health | python3 -m json.tool 2>/dev/null || echo "服务响应异常"
    exit 0
fi

echo "📋 配置信息:"
echo "   端口: $PORT"
echo "   设备: $DEVICE"
echo "   TTS: $ENABLE_TTS"
echo "   EmoLLM: $ENABLE_EMOLLM"
echo ""

echo "🚀 启动服务..."
echo "   日志文件: /tmp/unisign_api.log"
echo ""

# 启动服务
# 设置 CUDA 库路径
export LD_LIBRARY_PATH=/home/xtk/.cache/uv/archive-v0/L-1MGlRHZM-fJSA590Vc1/nvidia/cusparselt/lib:/home/xtk/.cache/uv/archive-v0/9t4U8cw1AOSJiHEKGUGUR/nvidia/cu13/lib:/home/xtk/.local/lib/python3.12/site-packages/nvidia/nvtx/lib:/home/xtk/.local/lib/python3.12/site-packages/nvidia/nccl/lib:/home/xtk/.local/lib/python3.12/site-packages/nvidia/cusparse/lib:/home/xtk/.local/lib/python3.12/site-packages/nvidia/curand/lib:/home/xtk/.local/lib/python3.12/site-packages/nvidia/cufft/lib:/home/xtk/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/xtk/.local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/home/xtk/.local/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:/home/xtk/.local/lib/python3.12/site-packages/nvidia/cublas/lib:/home/xtk/.local/lib/python3.12/site-packages/nvidia/cusolver/lib:/home/xtk/.local/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

nohup python -m uvicorn api:app --host 0.0.0.0 --port $PORT > /tmp/unisign_api.log 2>&1 &

# 等待服务启动
echo "⏳ 等待服务启动..."
for i in {1..30}; do
    sleep 1
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo ""
        echo "✅ 服务启动成功!"
        echo ""
        echo "📍 访问地址:"
        echo "   本地: http://localhost:$PORT"
        echo "   外部: https://u895901-9072-0273df24.westc.seetacloud.com:8443"
        echo ""
        echo "🔍 健康检查:"
        curl -s http://localhost:$PORT/health | python3 -m json.tool 2>/dev/null
        echo ""
        echo "📖 查看日志: tail -f /tmp/unisign_api.log"
        exit 0
    fi
    echo -n "."
done

echo ""
echo "❌ 服务启动超时，请检查日志:"
echo "   tail -50 /tmp/unisign_api.log"
