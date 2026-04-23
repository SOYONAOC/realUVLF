#!/bin/bash
#SBATCH --job-name=uvlf_4z
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm_uvlf_4z_%j.out

source /home/zhuhourui/AstroCode/realUVLF/.venv/bin/activate
cd /home/zhuhourui/AstroCode/realUVLF

echo "Job $SLURM_JOB_ID started on $(hostname) with $SLURM_CPUS_PER_TASK CPUs"
python run_uvlf_4z_separate.py
echo "Job $SLURM_JOB_ID finished at $(date)"
