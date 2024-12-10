#!/bin/bash
#SBATCH --open-mode=append
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=vb6etr
#SBATCH --output=o%j.out
#SBATCH --error=o%j.err
#SBATCH --array=3-9

source ~/vbad_env/bin/activate


# run hyper on the Dark Treasure-Room, with \rho=0
#python -m main --env fe_treasure_easy --wandb_id easy_treasure_$SLURM_ARRAY_TASK_ID --seed $SLURM_ARRAY_TASK_ID
# run rl2 on the Meta-RL Decieving Bandit with \mu_1=0.5
#python -m main --env mean_bandit_rl2 --wandb_id mean_bandit_rl2_$index --seed $index