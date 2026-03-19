#!/bin/bash
#SBATCH --job-name=chiaki
#SBATCH --output=mksny.txt
#SBATCH --error=jupyter.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amd
#SBATCH --cpus-per-task=1

set -euo pipefail

if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT=$(cd "$PROJECT_ROOT" && pwd)
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}/.venv" ]; then
    PROJECT_ROOT=$(cd "$SLURM_SUBMIT_DIR" && pwd)
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}/../.venv" ]; then
    PROJECT_ROOT=$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)
else
    echo "Unable to determine project root. Export PROJECT_ROOT before submitting." >&2
    exit 1
fi

SCRIPT_DIR=${KERNEL_DIR:-"$PROJECT_ROOT/kernel"}
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
VENV_JUPYTER="$PROJECT_ROOT/.venv/bin/jupyter"
PORT=${1:-${JUPYTER_PORT:-1695}}

if [ ! -x "$VENV_PYTHON" ] || [ ! -x "$VENV_JUPYTER" ]; then
    echo "Missing project virtualenv at $PROJECT_ROOT/.venv" >&2
    exit 1
fi

export AMD_TOKEN=${AMD_TOKEN:-$("$VENV_PYTHON" -c "import secrets; print(secrets.token_urlsafe(24))")}

{
    echo "Launching Jupyter Notebook on $(hostname) at port $PORT"
    echo "Project root: $PROJECT_ROOT"
    echo "Python: $VENV_PYTHON"
    echo "Jupyter: $VENV_JUPYTER"
    echo "To connect from your local machine, run:"
    echo "  ssh -L $PORT:$(hostname):$PORT $(hostname)"
    echo "Then open in browser: http://localhost:$PORT/?token=$AMD_TOKEN"
    echo "Token: $AMD_TOKEN"
} | tee "$SCRIPT_DIR/amd_jupyter.txt"

exec "$VENV_JUPYTER" lab \
    --no-browser \
    --port="$PORT" \
    --ip=0.0.0.0 \
    --IdentityProvider.token="$AMD_TOKEN" \
    --ServerApp.allow_origin='*' \
    --ServerApp.disable_check_xsrf=True \
    --allow-root \
    --notebook-dir="$PROJECT_ROOT"
