#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PORT=${1:-1695}
export JUPYTER_PORT=$PORT
export AMD_TOKEN=${AMD_TOKEN:-$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")}
export PROJECT_ROOT
export KERNEL_DIR="$SCRIPT_DIR"

chmod +x "$SCRIPT_DIR/amd.sh"
chmod +x "$SCRIPT_DIR/tunnel_back.sh"

JOB_ID=$(sbatch --export=ALL,JUPYTER_PORT="$JUPYTER_PORT",AMD_TOKEN="$AMD_TOKEN",PROJECT_ROOT="$PROJECT_ROOT",KERNEL_DIR="$KERNEL_DIR" "$SCRIPT_DIR/amd.sh" "$JUPYTER_PORT" | awk '{print $4}')
echo "作业 $JOB_ID 已提交，正在等待分配节点..."

NODE=""
while [ -z "$NODE" ]; do
    NODE=$(squeue -j "$JOB_ID" -h -o %N)
    [ -z "$NODE" ] && sleep 1
done
echo "作业运行在节点: $NODE"

nohup "$SCRIPT_DIR/tunnel_back.sh" "$JUPYTER_PORT" "$NODE" start > "$SCRIPT_DIR/tunnel_start.log" 2>&1 &

echo "SSH 隧道后台启动: 登录节点:$PORT -> $NODE:$PORT"
echo "Open in browser: http://localhost:$PORT/?token=$AMD_TOKEN"
echo "连接信息也会写入: $SCRIPT_DIR/amd_jupyter.txt"
