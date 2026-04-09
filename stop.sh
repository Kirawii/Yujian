#!/bin/bash

# Uni-Sign API 停止脚本

echo "========================================"
echo "  语见 API 停止脚本"
echo "========================================"
echo ""

PORT=6006
PID=$(ps aux | grep "uvicorn.*api:app.*$PORT" | grep -v grep | awk '{print $2}')

if [ -n "$PID" ]; then
    echo "🛑 停止服务 (PID: $PID)..."
    kill $PID
    sleep 2

    # 检查是否还在运行
    if ps -p $PID > /dev/null 2>&1; then
        echo "⚠️  服务未停止，强制终止..."
        kill -9 $PID
    fi

    echo "✅ 服务已停止"
else
    echo "ℹ️  服务未运行"
fi
