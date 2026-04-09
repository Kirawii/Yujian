#!/bin/bash

# 启动 YuJian API + TTS 服务

echo "========================================"
echo "  YuJian API + TTS 服务启动脚本"
echo "========================================"
echo ""

# 检查 TTS 服务是否已运行
TTS_PID=$(pgrep -f "python.*tts_server.py" | head -1)
if [ -n "$TTS_PID" ]; then
    echo "⚠️  TTS 服务已在运行 (PID: $TTS_PID)"
else
    echo "🚀 启动 TTS 服务 (端口 6007)..."
    cd /root/autodl-tmp/liuyongjie/Uni-Sign
    source tts_env/bin/activate
    nohup python tts_server.py > /tmp/tts_server.log 2>&1 &
    echo "   TTS 服务 PID: $!"
    deactivate
    echo "   日志: /tmp/tts_server.log"
    sleep 5
fi

echo ""
./start.sh
