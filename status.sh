#!/bin/bash

# Uni-Sign API 状态检查脚本

echo "========================================"
echo "  语见 API 状态检查"
echo "========================================"
echo ""

PORT=6006
PID=$(ps aux | grep "uvicorn.*api:app.*$PORT" | grep -v grep | awk '{print $2}')

if [ -n "$PID" ]; then
    echo "✅ 服务运行中"
    echo "   PID: $PID"
    echo ""
    echo "🔍 健康检查:"
    curl -s http://localhost:$PORT/health | python3 -m json.tool 2>/dev/null
    echo ""
    echo "📊 资源使用:"
    ps -p $PID -o pid,ppid,cmd,pcpu,pmem --no-headers
else
    echo "❌ 服务未运行"
fi

echo ""
echo "📖 最近日志 (最后10行):"
tail -10 /tmp/unisign_api.log 2>/dev/null || echo "暂无日志"
