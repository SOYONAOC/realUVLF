#!/bin/bash
cmd=$(jq -r '.tool_input.command // ""')
if echo "$cmd" | grep -q sbatch; then
  printf '{"continue":false,"stopReason":"Use slurm-auto-node-select skill instead of sbatch"}\n'
fi
