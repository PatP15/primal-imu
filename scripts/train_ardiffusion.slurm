#!/bin/bash

#SBATCH -J primal_ardiffusion
#SBATCH -p gpu_test
#SBATCH -c 8
#SBATCH -t 1:00:00
#SBATCH --mem=128G
#SBATCH -o /n/home01/ppuma/Github/primal-imu/scripts/ardiff_logs/ardiffusion_%j.out
#SBATCH --gres=gpu:1

#where the data and model at
export AMASS_DATA_PATH=/n/netscratch/walsh_lab_seas/Everyone/ppuma/datasets/AMASS/amass_data
export MODEL_REGISTRY_PATH=/n/netscratch/walsh_lab_seas/Everyone/ppuma/model-registry
# Load the required Python module.
module load python/3.12.8-fasrc01

# add to python path
export PYTHONPATH=/n/home01/ppuma/Github/primal-imu:$PYTHONPATH

# Navigate to project directory
cd /n/home01/ppuma/Github/primal-imu

# Activate conda environment and run training
mamba activate primal-imu && python scripts/train.py

