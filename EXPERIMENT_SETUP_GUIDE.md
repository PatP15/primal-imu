# Experiment Setup Guide: Velocity vs Acceleration Comparison

## Overview

This guide explains how to run comparison experiments between the velocity-based (`ARDiffusion`) and acceleration-based (`ARDiffusionAcceleration`) models.

## Files Created

### 1. Training Scripts
- **Original**: `scripts/train.py` (uses `ARDiffusion`)
- **Acceleration**: `scripts/train_acceleration.py` (uses `ARDiffusionAcceleration`)

### 2. Config Files
- **Original**: 
  - `primal/configs/train_diffusion.yaml`
  - `primal/configs/model/motion_diffuser_ar.yaml`
- **Acceleration**:
  - `primal/configs/train_diffusion_acceleration.yaml`
  - `primal/configs/model/motion_diffuser_ar_acceleration.yaml`

### 3. Model Classes
- **Original**: `ARDiffusion` (in `primal/models/motion_diffuser.py`)
- **Acceleration**: `ARDiffusionAcceleration` (in `primal/models/motion_diffuser.py`)

## Running Experiments

### Original (Velocity-Based) Model
```bash
cd scripts
python train.py
```

This will:
- Load config from `train_diffusion.yaml`
- Use `ARDiffusion` model
- Train with velocity conditioning
- Save to `logs/motion_diffuser_ar/`

### Acceleration-Based Model
```bash
cd scripts
python train_acceleration.py
```

This will:
- Load config from `train_diffusion_acceleration.yaml`
- Use `ARDiffusionAcceleration` model
- Train with acceleration conditioning
- Save to `logs/motion_diffuser_ar_acceleration/`

## Key Differences

### Model Differences (Internal)
1. **Motion Representation**:
   - Velocity: `vel = kpts[:,1:] - kpts[:,:-1]` (loses 1 frame)
   - Acceleration: `acc = vel[:,1:] - vel[:,:-1]` (loses 2 frames)

2. **Loss Function**:
   - Velocity: `loss_vel` weighted by `weight_vel`
   - Acceleration: `loss_acc` weighted by `weight_acc`

3. **Sequence Length**:
   - Velocity: Effective window = `seq_len - 1` = 15 frames
   - Acceleration: Effective window = `seq_len - 2` = 14 frames

### Training Script Differences
**NONE** - The training scripts are identical except:
- Config file name (line 17)
- Docstring (for documentation)

### Config Differences
1. **Model Target**:
   - Velocity: `primal.models.motion_diffuser.ARDiffusion`
   - Acceleration: `primal.models.motion_diffuser.ARDiffusionAcceleration`

2. **Loss Weight**:
   - Velocity: `weight_vel: 1`
   - Acceleration: `weight_acc: 1`

3. **Task Name**:
   - Velocity: `task_name: motion_diffuser_ar`
   - Acceleration: `task_name: motion_diffuser_ar_acceleration`

## Experiment Comparison

### What's the Same
✅ Same data (AMASS dataset)
✅ Same data preprocessing (canonicalization, etc.)
✅ Same network architecture (TransformerInContext)
✅ Same hyperparameters (h_dim, n_layer, n_head, dropout, etc.)
✅ Same training infrastructure (PyTorch Lightning, EMA, etc.)
✅ Same optimizer settings (AdamW, lr=1e-4)
✅ Same scheduler settings (DDPM, 50 timesteps)

### What's Different
🔀 Motion representation (velocity vs acceleration)
🔀 Loss function name (`loss_vel` vs `loss_acc`)
🔀 Loss weight parameter (`weight_vel` vs `weight_acc`)
🔀 Effective sequence length (15 vs 14 frames per batch)
🔀 Model class (`ARDiffusion` vs `ARDiffusionAcceleration`)

## Verification

Both models should:
- ✅ Load data identically
- ✅ Use same batch size (256)
- ✅ Use same learning rate (1e-4)
- ✅ Use same number of epochs
- ✅ Use same validation strategy
- ✅ Use same checkpointing strategy
- ✅ Use same logging (TensorBoard)

## Expected Results

### Training Metrics to Compare
1. **Loss Components**:
   - `loss_simple` (direct prediction loss)
   - `loss_fk` (forward kinematics loss)
   - `loss_vel` vs `loss_acc` (velocity vs acceleration consistency)

2. **Training Speed**:
   - Should be similar (same architecture)
   - Acceleration model processes slightly shorter sequences (14 vs 15 frames)

3. **Model Quality**:
   - Compare validation losses
   - Compare generated motion quality
   - Compare generation speed

## Troubleshooting

### Issue: "KeyError: 'weight_acc'"
**Solution**: Make sure you're using `train_diffusion_acceleration.yaml` which loads `motion_diffuser_ar_acceleration.yaml` with `weight_acc: 1`

### Issue: "Sequence too short for acceleration"
**Solution**: Ensure `data.seq_len >= 3`. Current setting is 16, which is sufficient.

### Issue: Cannot resume from velocity checkpoint
**Solution**: This is expected. Acceleration model has different architecture. Start fresh training or fine-tune with `strict=False` (not recommended).

## Next Steps

1. **Run both training scripts** with identical hyperparameters
2. **Monitor training** via TensorBoard:
   ```bash
   tensorboard --logdir logs/
   ```
3. **Compare metrics**:
   - Training loss curves
   - Validation loss curves
   - Component losses (simple, FK, vel/acc)
4. **Generate samples** from both models and compare qualitatively
5. **Evaluate** using your evaluation scripts

## Notes

- Both models use the same data loader - no changes needed there
- Both models use the same trainer configuration
- The only difference is in how the model processes the motion representation internally
- Results should be directly comparable since everything else is identical

