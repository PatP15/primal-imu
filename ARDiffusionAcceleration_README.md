# ARDiffusionAcceleration - Acceleration-Based Motion Diffusion

## Overview

`ARDiffusionAcceleration` is a new class that extends `ARDiffusion` to condition on accelerations instead of velocities. This variant maintains all the functionality of the base class but uses acceleration (second derivative of joint positions) as the conditioning signal.

## Files Created

1. **Class**: `primal/models/motion_diffuser.py`
   - New class: `ARDiffusionAcceleration` (lines ~2657-3081)
   - Inherits from `ARDiffusion`
   - No modifications to existing code

2. **Config**: `primal/configs/model/motion_diffuser_ar_acceleration.yaml`
   - Configuration file for the acceleration-based model
   - Uses `weight_acc` instead of `weight_vel` for loss weighting

## Key Differences from Base ARDiffusion

### 1. Motion Representation
- **Base**: Uses velocities `vel = kpts[:,1:] - kpts[:,:-1]`
- **Acceleration**: Uses accelerations `acc = vel[:,1:] - vel[:,:-1]` or `acc = kpts[:,2:] - 2*kpts[:,1:-1] + kpts[:,:-2]`

### 2. Sequence Length
- **Base**: Loses 1 frame → `nt_tw = seq_len - 1`
- **Acceleration**: Loses 2 frames → `nt_tw = seq_len - 2`

### 3. Metric Scaling
- **Base**: `vel * fps` (m/s)
- **Acceleration**: `acc * (fps ** 2)` (m/s²)

### 4. Loss Function
- **Base**: `loss_vel` with `weight_vel`
- **Acceleration**: `loss_acc` with `weight_acc`

### 5. Classifier Guidance
- **Base**: Guides on average velocity direction
- **Acceleration**: Guides on average acceleration direction

## Usage

### Training

**Command to run training:**
```bash
cd scripts
python train_acceleration.py
```

**GPU Requirements:**
- **Default**: 1 GPU (configured in `primal/configs/trainer/default_gpu.yaml`)
- **Minimum**: 1 CUDA-capable GPU
- **Multi-GPU**: Override devices parameter:
  ```bash
  # Use 2 GPUs
  python train_acceleration.py trainer.devices=2
  
  # Use 4 GPUs
  python train_acceleration.py trainer.devices=4
  
  # Use all available GPUs
  python train_acceleration.py trainer.devices=-1
  ```

**Note**: With DDP strategy, effective batch size = `batch_size × num_gpus`. Default batch size is 256.

**If you run out of memory**, reduce batch size:
```bash
python train_acceleration.py data.batch_size=128
```

**Or in Python:**
```python
from primal.models.motion_diffuser import ARDiffusionAcceleration

model = ARDiffusionAcceleration(cfg)
```

### Configuration

Key config parameters:
- `weight_acc`: Weight for acceleration consistency loss (replaces `weight_vel`)
- `use_l1_norm_vel`: Controls whether to use L1 or MSE for acceleration loss
- `motion_repr`: Should be `smplx_jts_locs_velocity_rotcont` (same as base)

### Generation

Same interface as base class:
```python
betas, xb_gen, kpts_vis, logs = model.generate_perpetual_navigation(
    batch,
    n_inference_steps=10,
    nt_max=1200,
    # ... other parameters
)
```

## Important Notes

1. **Sequence Length**: The effective window size is 2 frames shorter than the base class. Make sure your `seq_len` in data config accounts for this.

2. **Seed Computation**: In autoregressive generation, the class needs to handle the fact that acceleration requires 3 consecutive frames. The implementation includes fallback logic to handle edge cases.

3. **Visualization**: For visualization, velocities are recomputed from generated keypoints since the model outputs accelerations.

4. **Inertialization**: The inertialization function still works with the acceleration-based representation since it operates on the full motion representation tensor.

## Testing

To test the new class:
1. Use the config file: `primal/configs/model/motion_diffuser_ar_acceleration.yaml`
2. Ensure your data config has appropriate `seq_len` (should be at least 3 to compute accelerations)
3. Train and compare with base `ARDiffusion` model

## Future Extensions

You can create similar variants by:
- Inheriting from `ARDiffusionAcceleration`
- Overriding specific methods as needed
- Creating new config files

Example: `ARDiffusionAccelerationAction` for action-conditioned acceleration-based generation.

