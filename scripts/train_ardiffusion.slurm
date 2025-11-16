#!/bin/bash

#SBATCH -J primal_ardiffusion
#SBATCH -p test_gpu
#SBATCH -c 8
#SBATCH -t 1:00:00
#SBATCH --mem=64G
#SBATCH -o /n/home01/ppuma/primal-imu/logs/ardiffusion_%j.out
#SBATCH --gres=gpu:1

export AMASS_DATA_PATH=/n/netscratch/walsh_lab_seas/Everyone/ppuma/datasets/AMASS/amass_data
export MODEL_REGISTRY_PATH=/n/netscratch/walsh_lab_seas/Everyone/ppuma/model-registry
# Load the required Python module.
module load python/3.12.8-fasrc01

# Navigate to project directory
cd /n/home01/ppuma/primal-imu

# Activate conda environment and run training
srun -c 8 bash -c "conda init && source ~/.bashrc && conda activate primal-imu && python scripts/train.py"

