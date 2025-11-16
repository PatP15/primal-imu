#!/bin/bash

#SBATCH -J convert_amass
#SBATCH -p gpu_test
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH --mem=64G
#SBATCH -o /n/home01/ppuma/Github/primal-imu/logs/convert_amass_%j.log
#SBATCH --gres=gpu:1

export AMASS_DATA_PATH=/n/netscratch/walsh_lab_seas/Everyone/ppuma/datasets/AMASS/amass_data
export MODEL_REGISTRY_PATH=/n/netscratch/walsh_lab_seas/Everyone/ppuma/model-registry
export PYTHONPATH=/n/home01/ppuma/Github/primal-imu:$PYTHONPATH
export PYTHONUNBUFFERED=1

module load python/3.12.8-fasrc01
cd /n/home01/ppuma/Github/primal-imu

# Activate conda environment and run conversion
# Note: This will process all AMASS files and add jts_body to each NPZ file
mamba activate primal-imu && python -u primal/utils/cvt_amass_to_yup.py 2>&1

