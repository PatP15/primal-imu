# Conda Environment Setup for PRIMAL-IMU

This guide explains how to set up the conda environment for training on a SLURM cluster.

## Creating the Environment

### On the Cluster

1. **Navigate to project directory**:
   ```bash
   cd /n/home01/ppuma/primal-imu
   ```

2. **Create conda environment from environment.yml**:
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment**:
   ```bash
   conda activate primal-imu
   ```

### Troubleshooting Package Installation

#### Rust Packages (gloss-rs, smpl-rs)

These packages require Rust to compile. If installation fails:

1. **Install Rust** (if not already available):
   ```bash
   # Check if rust is available
   which rustc
   
   # If not, you may need to install it or load a module
   # Some clusters have rust available via modules
   ```

2. **Install packages manually**:
   ```bash
   conda activate primal-imu
   pip install gloss-rs>=0.7.0
   pip install smpl-rs>=0.7.0
   ```

#### PyTorch CUDA Version

The environment.yml uses `pytorch-cuda=11.8`. Adjust based on your cluster's CUDA version:

- **CUDA 11.8**: Use `pytorch-cuda=11.8` (current)
- **CUDA 12.1**: Change to `pytorch-cuda=12.1`

To check CUDA version on cluster:
```bash
nvidia-smi  # Shows CUDA version
```

#### Human Body Prior (Git Dependency)

The `human-body-prior` package is installed from GitHub. If you have network restrictions:

1. **Clone repository first**:
   ```bash
   git clone https://github.com/nghorbani/human_body_prior.git
   cd human_body_prior
   pip install .
   cd ..
   ```

2. **Or install directly** (should work in environment.yml):
   ```bash
   pip install git+https://github.com/nghorbani/human_body_prior.git
   ```

## Verifying Installation

After creating the environment, verify key packages:

```bash
conda activate primal-imu
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import lightning; print(f'Lightning: {lightning.__version__}')"
python -c "import hydra; print(f'Hydra: {hydra.__version__}')"
```

## Updating the Environment

If you need to update packages:

```bash
conda activate primal-imu
conda env update -f environment.yml --prune
```

## Exporting Current Environment

If you want to export your working environment:

```bash
conda activate primal-imu
conda env export > environment_exported.yml
```

## Notes

- The environment name is `primal-imu` (matches the SLURM scripts)
- Some packages may take time to install (especially Rust packages)
- Ensure you have sufficient disk space for the environment (~5-10GB)
- If using shared conda environments, consider creating in a user-specific location

