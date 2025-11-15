# SLURM Batch Job Scripts for Training

This directory contains SLURM batch job scripts for training the PRIMAL models on a cluster.

## Files

- `train_ardiffusion.slurm` - Training script for original ARDiffusion (velocity-based)
- `train_ardiffusion_acceleration.slurm` - Training script for ARDiffusionAcceleration (acceleration-based)

## Usage

### Before Running

1. **Create conda environment**:
   ```bash
   # On the cluster, create the environment from environment.yml
   conda env create -f environment.yml
   
   # Activate the environment
   conda activate primal-imu
   ```
   
   **Note**: See `CONDA_SETUP.md` in the project root for detailed setup instructions and troubleshooting.

2. **Update paths** in the SLURM scripts:
   - Replace `/n/home01/ppuma/primal-imu` with your actual project path
   - Update log directory paths if needed

3. **Create log directory**:
   ```bash
   mkdir -p /n/home01/ppuma/primal-imu/logs
   ```

4. **Verify environment**:
   - Ensure your conda environment has all required dependencies
   - Check that `AMASS_DATA_PATH` and `MODEL_REGISTRY_PATH` are set correctly
   - Note: Some packages (gloss-rs, smpl-rs) may require Rust toolchain

### Submitting Jobs

```bash
# Submit original ARDiffusion training
sbatch scripts/train_ardiffusion.slurm

# Submit ARDiffusionAcceleration training
sbatch scripts/train_ardiffusion_acceleration.slurm
```

### Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View output logs
tail -f /n/home01/ppuma/primal-imu/logs/ardiffusion_<JOBID>.out

# View error logs
tail -f /n/home01/ppuma/primal-imu/logs/ardiffusion_<JOBID>.err
```

## Resource Configuration

### Current Settings:
- **Partition**: `seas_gpu`
- **CPUs**: 8 cores
- **Memory**: 64GB
- **Time Limit**: 96 hours (4 days) - based on paper training time
- **GPUs**: 1 GPU

### Adjusting Resources

If you need to modify resources, edit the `#SBATCH` directives:

```bash
# For longer training (e.g., 5 days)
#SBATCH -t 120:00:00

# For more memory (if needed)
#SBATCH --mem=128G

# For multiple GPUs (if available)
#SBATCH --gres=gpu:2
# And update devices in config: trainer.devices=2
```

### Multi-GPU Training

To use multiple GPUs, modify the SLURM script:

```bash
#SBATCH --gres=gpu:2  # Request 2 GPUs

# Then in the command, override devices:
python scripts/train.py trainer.devices=2
```

## Notes

- **Time Limit**: Training takes approximately 4 days according to the paper (96 hours set as default)
- **Memory**: 64GB should be sufficient for batch size 256, but monitor usage
- **Checkpoints**: Models are saved automatically by PyTorch Lightning
- **Resume Training**: If job times out, you can resume using the checkpoint:
  ```bash
  python scripts/train.py resume_from_exp=/path/to/experiment
  ```

## Troubleshooting

### Job Fails Immediately
- Check error log: `cat /n/home01/ppuma/primal-imu/logs/ardiffusion_accel_<JOBID>.err`
- Verify paths are correct
- Ensure conda environment is activated correctly

### Out of Memory
- Reduce batch size: Add `data.batch_size=128` to command
- Request more memory: `#SBATCH --mem=128G`

### Time Limit Exceeded
- Request more time: `#SBATCH -t 72:00:00`
- Or resume from checkpoint (see above)

### GPU Not Found
- Verify partition has GPUs: `sinfo -p seas_gpu`
- Check GPU availability: `squeue -p seas_gpu`

