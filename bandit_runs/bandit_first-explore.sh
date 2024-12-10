#!/bin/bash
#SBATCH --open-mode=append
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=br
#SBATCH --output=o%j.out
#SBATCH --error=o%j.err
#SBATCH --array=0-9
#SBATCH --qos=m3

source ~/hf/bin/activate
export LD_LIBRARY_PATH=/scratch/ssd001/pkgs/cuda-11.3/lib64:/scratch/ssd001/pkgs/cudnn-11.4-v8.2.4.15/lib64:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/scratch/ssd001/pkgs/cuda-11.3
export PATH=/scratch/ssd001/pkgs/cuda-11.3/bin:$PATH
python bandit_first-explore.py --SBATCHID $SLURM_ARRAY_TASK_ID --run_ID bandit_first-explore --group bandit_first-explore --SLURM_JOB_ID $SLURM_JOB_ID
