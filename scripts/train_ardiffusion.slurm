#!/bin/bash

#SBATCH -J primal_ardiffusion
#SBATCH -p seas_gpu
#SBATCH -c 8
#SBATCH -t 96:00:00
#SBATCH --mem=64G
#SBATCH -o /n/home01/ppuma/primal-imu/logs/ardiffusion_%j.out
#SBATCH -e /n/home01/ppuma/primal-imu/logs/ardiffusion_%j.err
#SBATCH --gres=gpu:1

# Load the required Python module.
module load python/3.12.8-fasrc01

# Navigate to project directory
cd /n/home01/ppuma/primal-imu

# Activate conda environment and run training
srun -c 8 bash -c "conda init && source ~/.bashrc && conda activate primal-imu && python scripts/train.py"

