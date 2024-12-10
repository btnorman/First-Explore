#!/bin/bash
#SBATCH --open-mode=append
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=fe1
#SBATCH --output=o%j.out
#SBATCH --error=o%j.err
#SBATCH --qos m2
#SBATCH --array=1-5
#SBATCH --signal=B:USR1@60

handler()
{
echo "function handler called at $(date)"
sbatch ${BASH_SOURCE[0]}
}
# register signal handler
trap handler SIGUSR1

## note the array call may not work with the handler automatic requeing.


source ~/FE/bin/activate
export LD_LIBRARY_PATH=/scratch/ssd001/pkgs/cuda-11.3/lib64:/scratch/ssd001/pkgs/cudnn-11.4-v8.2.4.15/lib64:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/scratch/ssd001/pkgs/cuda-11.3
export PATH=/scratch/ssd001/pkgs/cuda-11.3/bin:$PATH

python -m tiny_world_first_explore --SBATCHID $SLURM_ARRAY_TASK_ID --run_ID tiny_world_1 --group tiny_world --SLURM_JOB_ID $SLURM_JOB_ID &
wait