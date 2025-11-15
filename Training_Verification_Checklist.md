# Training Script Verification Checklist

This document verifies that `train_acceleration.py` works EXACTLY the same as `train.py`, except for the necessary differences related to using accelerations instead of velocities.

## Step-by-Step Verification

### ✅ Step 1: Imports and Setup
**Status**: IDENTICAL
- Both scripts import the same modules
- Both set `torch.set_float32_matmul_precision('high')`
- Both use the same logger setup

### ✅ Step 2: Hydra Configuration
**Status**: DIFFERENT (as intended)
- `train.py`: Uses `config_name="train_diffusion"`
- `train_acceleration.py`: Uses `config_name="train_diffusion_acceleration"`

**Verification**:
- `train_diffusion.yaml` → loads `model: motion_diffuser_ar` → `ARDiffusion`
- `train_diffusion_acceleration.yaml` → loads `model: motion_diffuser_ar_acceleration` → `ARDiffusionAcceleration`

### ✅ Step 3: Resume Training Logic
**Status**: IDENTICAL
- Both handle `resume_from_exp` the same way
- Both load config from checkpoint's `.hydra/config.yaml`
- Both preserve device configuration
- Both copy hydra directory structure

### ✅ Step 4: Data Module Creation
**Status**: IDENTICAL
- Both create `MixedMotionDataModule(cfg.data)`
- Uses the same data config (`amass_smplx`)
- **Note**: Data loading is identical - the difference is only in how the model processes the data

### ✅ Step 5: Model Instantiation
**Status**: DIFFERENT (as intended)
- `train.py`: `hydra.utils.instantiate(cfg.model)` → `ARDiffusion`
- `train_acceleration.py`: `hydra.utils.instantiate(cfg.model)` → `ARDiffusionAcceleration`

**Verification**:
- Both use Hydra's instantiate mechanism
- Both models inherit from `MotionDiffuserBase` and have the same interface
- Both support the same training modes (fresh, resume, finetune)

### ✅ Step 6: Logger Creation
**Status**: IDENTICAL
- Both use `hydra.utils.instantiate(cfg.logger)`
- Both use the same logger config (`tensorboard`)

### ✅ Step 7: Trainer Setup
**Status**: IDENTICAL
- Both create `ModelCheckpoint` with same parameters
- Both instantiate trainer with same configuration
- Both use the same trainer config (`default_gpu`)

### ✅ Step 8: Training Modes
**Status**: IDENTICAL

#### 8a. Resume Training
- Both find checkpoint using same glob pattern
- Both load checkpoint the same way
- Both call `trainer.fit()` with checkpoint path

#### 8b. Fine-tuning
- Both find checkpoint using same glob pattern
- Both load state dict with `strict=False`
- Both handle EMA weights the same way
- Both call `trainer.fit()` without checkpoint path

#### 8c. Fresh Training
- Both call `trainer.fit()` directly

## Necessary Differences (Expected)

### 1. Model Class
- **Original**: `ARDiffusion` (velocity-based)
- **Acceleration**: `ARDiffusionAcceleration` (acceleration-based)
- **Impact**: Only affects internal computation, not training script

### 2. Config File
- **Original**: `train_diffusion.yaml` → `motion_diffuser_ar.yaml`
- **Acceleration**: `train_diffusion_acceleration.yaml` → `motion_diffuser_ar_acceleration.yaml`
- **Impact**: Loads different model class and uses `weight_acc` instead of `weight_vel`

### 3. Loss Weight Parameter
- **Original**: Uses `weight_vel` in config
- **Acceleration**: Uses `weight_acc` in config
- **Impact**: Model accesses `self.hparams.weight_acc` instead of `self.hparams.weight_vel`
- **Verification**: ✅ Model code correctly uses `weight_acc` (line 2761 in motion_diffuser.py)

## Data Processing Differences (Internal to Model)

These differences are handled inside the model class, not the training script:

### 1. Motion Representation Computation
- **Original**: Computes velocities `vel = kpts[:,1:] - kpts[:,:-1]`
- **Acceleration**: Computes accelerations `acc = vel[:,1:] - vel[:,:-1]`
- **Impact**: Model loses 2 frames instead of 1, but data loading is identical

### 2. Sequence Length
- **Original**: Effective window size = `seq_len - 1`
- **Acceleration**: Effective window size = `seq_len - 2`
- **Impact**: Slightly shorter sequences per batch, but handled internally by model

### 3. Loss Computation
- **Original**: `loss_vel` with `weight_vel`
- **Acceleration**: `loss_acc` with `weight_acc`
- **Impact**: Different loss name, but same training loop structure

## Verification Tests

### Test 1: Config Loading
```python
# Should work without errors
from omegaconf import DictConfig
cfg = OmegaConf.load("primal/configs/train_diffusion_acceleration.yaml")
assert cfg.model._target_ == "primal.models.motion_diffuser.ARDiffusionAcceleration"
```

### Test 2: Model Instantiation
```python
# Should instantiate correctly
from hydra import initialize, compose
with initialize(config_path="../primal/configs", version_base=None):
    cfg = compose(config_name="train_diffusion_acceleration")
    model = hydra.utils.instantiate(cfg.model)
    assert isinstance(model, ARDiffusionAcceleration)
```

### Test 3: Training Script Execution
```bash
# Should run without errors (dry run)
cd scripts
python train_acceleration.py --help
```

## Potential Issues and Solutions (Acceleration-Specific Only)

These errors are specific to the acceleration-based changes and would not occur in the original velocity-based training.

### Issue 1: Missing `weight_acc` in Config
**Status**: ✅ **FIXED** - Verified in code
**Symptom**: `KeyError: 'weight_acc'` or `AttributeError: 'DictConfig' object has no attribute 'weight_acc'`
**Location**: `motion_diffuser.py` line 2761 (`self.hparams.weight_acc`)
**Verification**: Config file `motion_diffuser_ar_acceleration.yaml` line 35 has `weight_acc: 1` ✅

### Issue 2: Data Sequence Length Too Short for Acceleration
**Status**: ✅ **OK** - Current config sufficient
**Symptom**: `IndexError: index out of range` or `RuntimeError: Sizes of tensors must match` when computing acceleration
**Location**: `motion_diffuser.py` lines 2697-2698 (acceleration computation: `vel[:,1:] - vel[:,:-1]`)
**Verification**: `amass_smplx.yaml` has `seq_len: 16` ✅ (need ≥3, have 16)

### Issue 3: Dimension Mismatch in Loss Computation
**Status**: ⚠️ **POTENTIAL** - Needs runtime verification
**Symptom**: `RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)` in loss_acc
**Location**: `motion_diffuser.py` line 2756 (`losses['loss_acc'] = fn_dist_acc(acc, acc_pred_fk)`)
**Code Check**: 
- Line 2698: `acc = vel[:,1:] - vel[:,:-1]` → loses 2 frames ✅
- Line 2704: `xb = xb[:,:-2]`, `kpts = kpts[:,:-2]` → loses 2 frames ✅
- Line 2753: `acc_pred_fk = kpts_pred_fk[:,2:] - 2*kpts_pred_fk[:,1:-1] + kpts_pred_fk[:,:-2]` → loses 2 frames ✅
- Both `acc` and `acc_pred_fk` should have shape `[batch, seq_len-2, n_kpts, 3]` ✅
**Note**: Should work, but verify at runtime

### Issue 4: Window Size Mismatch in Generation
**Status**: ✅ **FIXED** - Verified in code
**Location**: `motion_diffuser.py` line 2798 (`nt_tw = self.hparams.data.seq_len - 2`)
**Verification**: Uses `seq_len - 2` correctly ✅

### Issue 5: Seed Computation Error in Autoregressive Generation (tt > 0)
**Status**: ✅ **HAS FALLBACK** - Should handle edge cases
**Location**: `motion_diffuser.py` lines 2873-2894 (seed computation for subsequent windows)
**Code Check**: 
- Line 2875: Checks `if kpts_seed_c.shape[1] >= 2:` ✅
- Line 2877: Checks `if vel_seed_c.shape[1] >= 1:` ✅
- Lines 2880-2886, 2888-2894: Fallback to `acc_gen_w_[:,-1:]` ✅
**Note**: Fallback logic exists, should handle edge cases

### Issue 6: Metric Velocity Scaling Error (fps² instead of fps)
**Status**: ✅ **FIXED** - All locations verified
**Location**: `motion_diffuser.py` lines 2701, 2755, 2850, 3053-3054
**Verification**: All use `(self.fps ** 2)`:
- Line 2701: `acc = acc * (self.fps ** 2)` ✅
- Line 2755: `acc_pred_fk = acc_pred_fk * (self.fps ** 2)` ✅
- Line 2850: `acc_seed_c = acc_seed_c * (self.fps ** 2)` ✅
- Lines 3053-3054: `acc_gen *= (self.fps ** 2)` ✅

### Issue 7: Inertialization Shape Mismatch
**Status**: ✅ **OK** - Function designed for this
**Location**: `motion_diffuser.py` line 3010 (`inertialize(xs_seed, input, ...)`)
**Code Check**: 
- `inertialize` function (in `mop_repr.py` line 78) expects:
  - `seq1`: `[b, 1, d]` (single frame) ✅ - `xs_seed` is `[nb, 1, x_dim]`
  - `seq2`: `[b, t, d]` (multi-frame, any `t`) ✅ - `input` is `[nb, nt_tw, x_dim]`
- Function handles different time dimensions ✅
**Note**: This is correct - `inertialize` is designed to blend from 1 frame to N frames

### Issue 8: Frame Indexing Error in Output Storage
**Status**: ✅ **OK** - Should work with current seq_len
**Location**: `motion_diffuser.py` lines 3032-3035 (storing generated sequences)
**Code Check**: 
- Uses `[:,:-2]` slice ✅
- `nt_tw = seq_len - 2 = 16 - 2 = 14` ✅
- `xb_gen_w_.shape[1] = nt_tw = 14` (before slice) ✅
- After `[:,:-2]`: shape becomes `[nb, 12]` ✅
- Requires `nt_tw >= 2`, which means `seq_len >= 4` ✅ (have 16)
**Note**: Should work, but verify `seq_len >= 4` if using custom data config

### Issue 9: Step Size Error in Autoregressive Loop
**Status**: ✅ **FIXED** - Verified in code
**Location**: `motion_diffuser.py` line 3041 (`tt += nt_tw - 2`)
**Verification**: 
- Uses `nt_tw - 2` ✅
- Matches `xb_gen.append(xb_gen_w_[:,:-2])` which stores `nt_tw - 2` frames ✅

### Issue 10: Classifier Guidance on Accelerations (Wrong Extraction)
**Status**: ✅ **FIXED** - Verified in code
**Location**: `motion_diffuser.py` lines 2957-2962 (acceleration guidance computation)
**Code Check**: 
- Line 2957: `accs_reshaped = modeloutput[...,-3*self.n_kpts:]` ✅
- Model output structure: `[xb | kpts | acc]` where acc is last `3*n_kpts` dims ✅
- Line 2957: Reshapes to `[nb, -1, n_kpts, 3]` ✅

### Issue 11: Visualization Velocity Computation from Keypoints
**Status**: ✅ **OK** - Should work with concatenated sequences
**Location**: `motion_diffuser.py` lines 3057-3059 (visualization velocity computation)
**Code Check**: 
- Line 3049: `kpts_gen = torch.cat(kpts_gen, dim=1)` concatenates all sequences ✅
- Line 3057: `vel_gen = kpts_gen[:,1:] - kpts_gen[:,:-1]` requires `kpts_gen.shape[1] >= 2` ✅
- After concatenation, should have many frames ✅
**Note**: Should work, but verify at runtime if generation produces very short sequences

### Issue 12: Config Model Target Mismatch
**Status**: ✅ **FIXED** - Verified in config
**Location**: Config file `_target_` field
**Verification**: `motion_diffuser_ar_acceleration.yaml` line 1 has `_target_: primal.models.motion_diffuser.ARDiffusionAcceleration` ✅

## Error Prevention Checklist (Acceleration-Specific)

Before running acceleration training, verify these acceleration-specific items:

### Config Verification
- [ ] Config file exists: `primal/configs/train_diffusion_acceleration.yaml`
- [ ] Model config exists: `primal/configs/model/motion_diffuser_ar_acceleration.yaml`
- [ ] Model config has `_target_: primal.models.motion_diffuser.ARDiffusionAcceleration` (not `ARDiffusion`)
- [ ] Model config has `weight_acc: 1` (NOT `weight_vel: 1`)
- [ ] Model class exists: `ARDiffusionAcceleration` in `motion_diffuser.py` (line ~2657)

### Data Configuration
- [ ] Data config has `seq_len >= 3` (currently 16 ✅ - sufficient)
- [ ] If using custom data, verify sequences are long enough for acceleration computation

### Model Code Verification
- [ ] Acceleration computation uses `fps ** 2` (not just `fps`) - check lines 2701, 2755, 2850, 3053
- [ ] Loss uses `weight_acc` (not `weight_vel`) - check line 2761
- [ ] Window size calculation uses `seq_len - 2` (not `seq_len - 1`) - check line 2798
- [ ] Step size in generation loop uses `nt_tw - 2` (not `nt_tw - 1`) - check line 3041
- [ ] Frame slicing uses `[:-2]` (not `[:-1]`) - check lines 3032-3035

### Shape Consistency Checks
- [ ] Acceleration loses exactly 2 frames: `acc.shape[0] == kpts.shape[0] - 2`
- [ ] Window size matches: `nt_tw == seq_len - 2`
- [ ] Seed shape compatible: `xs_seed.shape == [nb, 1, x_dim]` where `x_dim` includes acceleration

## Runtime Error Monitoring (Acceleration-Specific)

During training, watch for these acceleration-specific issues:

### Shape Mismatches
- [ ] No `RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)` in loss_acc
- [ ] No `IndexError` when accessing `[:,:-2]` slices
- [ ] No shape errors in `generate_perpetual_navigation` window size

### Numerical Issues
- [ ] Acceleration values are reasonable (not orders of magnitude different from velocities)
- [ ] Loss values are finite (not NaN or Inf) - acceleration computation can be more sensitive
- [ ] No division by zero in acceleration guidance gradients

### Generation Issues
- [ ] No errors in seed computation for subsequent windows (tt > 0)
- [ ] No errors in inertialization with acceleration-based seeds
- [ ] Visualization works correctly (velocities recomputed from keypoints)

## Execution Comparison

### Original Training
```bash
cd scripts
python train.py
```

### Acceleration Training
```bash
cd scripts
python train_acceleration.py
```

### GPU Configuration

**Original Authors' Setup**:
- **Default**: 1 GPU (configured in `primal/configs/trainer/default_gpu.yaml`)
  - `devices: 1` - Original authors trained with single GPU
  - `strategy: ddp` - DDP strategy is set (unusual for single GPU, but works)
  - `accelerator: cuda` - Requires CUDA-capable GPU

**Multi-GPU Support**:
- ✅ **Code supports multi-GPU**: Infrastructure is in place
  - DDP strategy configured
  - Multi-GPU-aware logging (`rank_zero_only` decorator)
  - Device config preserved during resume training
  - Commented line suggests authors considered `DDPStrategy(find_unused_parameters=True)`
- ⚠️ **Potential Issue**: Using `strategy: ddp` with `devices: 1` works but is unusual
  - PyTorch Lightning should handle this (falls back to single process)
  - Typically you'd use `strategy: "auto"` or `strategy: null` for single GPU
  - The current setup works but may have slight overhead

**To use multiple GPUs**, override the devices parameter:
```bash
# Use 2 GPUs
python train_acceleration.py trainer.devices=2

# Use 4 GPUs
python train_acceleration.py trainer.devices=4

# Use all available GPUs
python train_acceleration.py trainer.devices=-1
```

**Note**: With DDP strategy, each GPU processes a portion of the batch. Total effective batch size = `batch_size × num_gpus`.

**Minimum Requirements**:
- 1 GPU with CUDA support
- Sufficient GPU memory for batch size 256 (default)
- If memory issues occur, reduce batch size: `python train_acceleration.py data.batch_size=128`

**Recommendation**: 
- For single GPU: The current setup works, but you could change `strategy: "auto"` for cleaner single-GPU training
- For multi-GPU: Should work out of the box, but monitor for any DDP-related issues (unused parameters, etc.)

Both commands:
- Use the same data
- Use the same trainer settings
- Use the same logger
- Use the same checkpointing strategy
- Only differ in the model class used

## Summary

✅ **Training script is identical** except for:
1. Config file name (line 17)
2. Docstring update (lines 19-37)

✅ **All training logic is identical**:
- Resume training
- Fine-tuning
- Fresh training
- Data loading
- Trainer setup
- Checkpointing

✅ **Model differences are handled internally**:
- Acceleration computation
- Loss calculation
- Sequence length handling

✅ **Ready for comparison experiments**:
- Same data
- Same hyperparameters (except weight_acc vs weight_vel)
- Same training infrastructure
- Only difference is velocity vs acceleration conditioning

## Final Checklist

- [x] Training script created (`train_acceleration.py`)
- [x] Config file created (`train_diffusion_acceleration.yaml`)
- [x] Model config created (`motion_diffuser_ar_acceleration.yaml`)
- [x] All imports identical
- [x] All training logic identical
- [x] Model uses correct weight parameter (`weight_acc`)
- [x] Data config compatible (seq_len >= 3)
- [x] Ready for experiments

