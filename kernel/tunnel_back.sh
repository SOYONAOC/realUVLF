#!/bin/bash
# 用法: ./tunnel.sh [端口] [主机] [cmd: start/stop/check]
PORT=${1:-$JUPYTER_PORT}; HOST=${2:-amd1}; CMD=${3:-start}
SOCK="/tmp/ssh_tunnel_${PORT}_${HOST}.sock" # 控制文件路径

case $CMD in
    "stop")  ssh -S $SOCK -O exit $HOST 2>/dev/null && echo "🛑 隧道已关闭" ;;
    "check") ssh -S $SOCK -O check $HOST 2>/dev/null && echo "✅ 隧道运行中" || echo "❌ 隧道未运行" ;;
    *)       ssh -M -S $SOCK -fnNT -L $PORT:$HOST:$PORT \
             -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes $HOST \
             && echo "🚀 隧道已启动: localhost:$PORT -> $HOST" ;;
esac