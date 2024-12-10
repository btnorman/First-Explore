#!/bin/bash
#SBATCH --open-mode=append
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=rl_bandit
#SBATCH --output=o%j.out
#SBATCH --error=o%j.err
#SBATCH --array=0-4

source ~/vbad_env/bin/activate
python -m main --env mean_bandit_rl2 --wandb_id bandit_mmp_rl2_3_$index --seed $index --ppo_num_minibatch 4 --ppo_clip_param 0.05 --policy_num_steps 100 --vae_batch_num_trajs 25 --num_vae_updates 3 --kl_weight 1 --num_frames 100000000