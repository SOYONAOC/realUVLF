#!/bin/bash
#SBATCH --job-name=uvlf_full
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm_full_uvlf_%j.out

source /home/zhuhourui/AstroCode/realUVLF/.venv/bin/activate
cd /home/zhuhourui/AstroCode/realUVLF

echo "Job $SLURM_JOB_ID started on $(hostname) with $SLURM_CPUS_PER_TASK CPUs"
python run_full_uvlf.py
echo "Job $SLURM_JOB_ID finished at $(date)"
