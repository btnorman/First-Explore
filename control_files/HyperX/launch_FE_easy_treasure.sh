#!/bin/bash
#SBATCH --open-mode=append
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=hx6etr
#SBATCH --output=o%j.out
#SBATCH --error=o%j.err
#SBATCH --array=0-4

source ~/vbad_env/bin/activate
python -m main --env fe_treasure_easy --wandb_id easy6roll_$SLURM_ARRAY_TASK_ID --seed $SLURM_ARRAY_TASK_ID
