#!/bin/bash -x
#SBATCH --job-name=intention_speaker_transformer
#SBATCH --output=slurm/%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

eval "$(/nas/ucb/tutrinh/anaconda3/bin/conda shell.bash hook)"
conda activate chai

export CUDA_LAUNCH_BLOCKING=1

cd /nas/ucb/tutrinh/selfexplainingai

srun python3 baselines/transformer_baseline.py -m traj -t
