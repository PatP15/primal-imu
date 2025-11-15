# Plan: Modifying ARDiffusion to Use Accelerations Instead of Velocities

## Overview

This document outlines all the changes needed to modify ARDiffusion to condition on accelerations instead of velocities, while keeping everything else the same.

## Key Mathematical Changes

### Current (Velocities):
- **Computation**: `vel = kpts[:,1:] - kpts[:,:-1]` (first derivative)
- **Sequence loss**: 1 frame (from `seq_len` to `seq_len-1`)
- **Dimension**: Same as joint positions (n_kpts × 3)

### New (Accelerations):
- **Computation**: `acc = kpts[:,2:] - 2*kpts[:,1:-1] + kpts[:,:-2]` (second derivative)
  - OR: `vel = kpts[:,1:] - kpts[:,:-1]` then `acc = vel[:,1:] - vel[:,:-1]`
- **Sequence loss**: 2 frames (from `seq_len` to `seq_len-2`)
- **Dimension**: Same as joint positions (n_kpts × 3)

---

## 1. Training Forward Pass (`forward_one()` method)

### Location: `primal/models/motion_diffuser.py` lines ~585-651

#### Changes Needed:

**A. Acceleration Computation (instead of velocity)**
```python
# OLD:
vel = kpts[:,1:] - kpts[:,:-1]
if self.use_metric_velocity:
    vel = vel * self.fps
xb = xb[:,:-1]
kpts = kpts[:,:-1]

# NEW:
# Option 1: Direct second derivative
acc = kpts[:,2:] - 2*kpts[:,1:-1] + kpts[:,:-2]
if self.use_metric_velocity:
    acc = acc * (self.fps ** 2)  # Note: acceleration scales with fps^2
xb = xb[:,:-2]  # Lose 2 frames instead of 1
kpts = kpts[:,:-2]

# Option 2: Via velocities (two-step)
vel = kpts[:,1:] - kpts[:,:-1]
acc = vel[:,1:] - vel[:,:-1]
if self.use_metric_velocity:
    acc = acc * (self.fps ** 2)
xb = xb[:,:-2]
kpts = kpts[:,:-2]
```

**B. Concatenation (replace `vel` with `acc`)**
```python
# OLD:
xs = torch.cat([
    xb, 
    kpts.reshape(nb, nt, -1),
    vel.reshape(nb, nt, -1)
], dim=-1)

# NEW:
xs = torch.cat([
    xb, 
    kpts.reshape(nb, nt, -1),
    acc.reshape(nb, nt, -1)  # Changed from vel to acc
], dim=-1)
```

**C. Loss Computation (rename and update)**
```python
# OLD:
vel_pred_fk = kpts_pred_fk[:,1:] - kpts_pred_fk[:,:-1]
if self.use_metric_velocity:
    vel_pred_fk = vel_pred_fk * self.fps
losses['loss_vel'] = fn_dist_vel(vel[:,:-1], vel_pred_fk)

# NEW:
acc_pred_fk = kpts_pred_fk[:,2:] - 2*kpts_pred_fk[:,1:-1] + kpts_pred_fk[:,:-2]
if self.use_metric_velocity:
    acc_pred_fk = acc_pred_fk * (self.fps ** 2)
losses['loss_acc'] = fn_dist_vel(acc, acc_pred_fk)  # Note: still use fn_dist_vel (just rename variable)
```

**D. Total Loss (update weight name)**
```python
# OLD:
losses['loss'] = self.hparams.weight_simple*losses['loss_simple'] \
    + self.hparams.weight_fk*losses['loss_fk'] \
    + self.hparams.weight_vel*losses['loss_vel']

# NEW:
losses['loss'] = self.hparams.weight_simple*losses['loss_simple'] \
    + self.hparams.weight_fk*losses['loss_fk'] \
    + self.hparams.weight_acc*losses['loss_acc']  # Update config to use weight_acc
```

---

## 2. Inference/Generation (`generate_perpetual_navigation()` method)

### Location: `primal/models/motion_diffuser.py` lines ~672-1086

#### Changes Needed:

**A. Initial Seed Computation (first iteration, tt==0)**
```python
# OLD (line ~824):
vel_seed_c = kpts[:,1:] - kpts[:,:-1]
if self.use_metric_velocity:
    vel_seed_c = vel_seed_c * self.fps
xb_seed_c = xb[:,:-1]
kpts_seed_c = kpts[:,:-1]

# NEW:
acc_seed_c = kpts[:,2:] - 2*kpts[:,1:-1] + kpts[:,:-2]
if self.use_metric_velocity:
    acc_seed_c = acc_seed_c * (self.fps ** 2)
xb_seed_c = xb[:,:-2]
kpts_seed_c = kpts[:,:-2]
```

**B. Seed from Previous Window (tt > 0)**
```python
# OLD (line ~849):
vel_seed_w = vel_gen_w_[:,-1:].detach().clone()
vel_seed_c = torch.einsum(
    'bij,btpj->btpi', 
    rotmat_c.permute(0,2,1),
    vel_seed_w
)

# NEW:
# Need to compute acceleration from last 2 frames of previous window
# Option 1: Store last 2 frames of velocities
acc_seed_w = acc_gen_w_[:,-1:].detach().clone()  # Need to track acc_gen_w_
acc_seed_c = torch.einsum(
    'bij,btpj->btpi', 
    rotmat_c.permute(0,2,1),
    acc_seed_w
)

# Option 2: Recompute from keypoints (more accurate)
kpts_seed_w_last2 = kpts_gen_w_[:,-2:].detach().clone()  # Last 2 frames
kpts_seed_c_last2 = torch.einsum(
    'bij,btpj->btpi', 
    rotmat_c.permute(0,2,1), 
    kpts_seed_w_last2-transl_c.unsqueeze(-2)
)
acc_seed_c = kpts_seed_c_last2[:,1:] - 2*kpts_seed_c_last2[:,:-1] + ...  # Need previous frame too
# Actually, we need 3 frames total for acceleration, so we'd need to store more history
```

**C. Velocity Perturbations (lines ~856-876)**
```python
# OLD:
if use_vel_perburbation and tt==0:
    # ... applies perturbations to vel_seed_c

# NEW:
# Rename to use_acc_perburbation, but conceptually same
# Note: Acceleration perturbations might need different magnitudes
if use_acc_perburbation and tt==0:  # Rename variable
    # Apply perturbations to acc_seed_c instead
    # May need to adjust perturbation magnitudes (acceleration is second derivative)
```

**D. Seed Concatenation (line ~889-892)**
```python
# OLD:
xs_seed = torch.cat([xb_seed_c, 
                    kpts_seed_c.reshape(nb, 1, -1),
                    vel_seed_c.reshape(nb, 1, -1)],
                    dim=-1)

# NEW:
xs_seed = torch.cat([xb_seed_c, 
                    kpts_seed_c.reshape(nb, 1, -1),
                    acc_seed_c.reshape(nb, 1, -1)],  # Changed from vel to acc
                    dim=-1)
```

**E. Classifier Guidance (lines ~917-960)**
```python
# OLD (line ~926):
vels_reshaped = modeloutput[...,-3*self.n_kpts:].reshape(nb, -1, self.n_kpts, 3)
grad_mv_vels = 2*(vels_reshaped.mean(dim=[1,2],keepdim=True) - ori_c.unsqueeze(-2))
grad_mv_vels = grad_mv_vels.repeat(1, nt_tw, self.n_kpts,1).reshape(nb, nt_tw, -1)
grad_mv_pad = torch.zeros_like(input[...,:-3*self.n_kpts])
grad_mv = torch.cat([grad_mv_pad, grad_mv_vels], dim=-1)

# NEW:
# Guidance on accelerations instead of velocities
# Note: Guidance on acceleration means controlling change in velocity
accs_reshaped = modeloutput[...,-3*self.n_kpts:].reshape(nb, -1, self.n_kpts, 3)
# For movement direction guidance, we might want to integrate acceleration to get velocity
# Or directly guide average acceleration direction
grad_mv_accs = 2*(accs_reshaped.mean(dim=[1,2],keepdim=True) - ori_c.unsqueeze(-2))
grad_mv_accs = grad_mv_accs.repeat(1, nt_tw, self.n_kpts,1).reshape(nb, nt_tw, -1)
grad_mv_pad = torch.zeros_like(input[...,:-3*self.n_kpts])
grad_mv = torch.cat([grad_mv_pad, grad_mv_accs], dim=-1)
```

**F. Output Decomposition (line ~1004-1006)**
```python
# OLD:
xb_gen_c = output[...,:-self.n_kpts*6]
kpts_gen_c = output[...,-self.n_kpts*6:-self.n_kpts*3].reshape(nb,-1,self.n_kpts,3)
vel_gen_c = output[...,-self.n_kpts*3:].reshape(nb,-1,self.n_kpts,3)

# NEW:
xb_gen_c = output[...,:-self.n_kpts*6]
kpts_gen_c = output[...,-self.n_kpts*6:-self.n_kpts*3].reshape(nb,-1,self.n_kpts,3)
acc_gen_c = output[...,-self.n_kpts*3:].reshape(nb,-1,self.n_kpts,3)  # Changed from vel to acc
```

**G. Coordinate Transform (line ~1011-1012)**
```python
# OLD:
kpts_gen_w_ = torch.einsum('bij,btpj->btpi', rotmat_c, kpts_gen_c) + transl_c.unsqueeze(-2)
vel_gen_w_ = torch.einsum('bij,btpj->btpi', rotmat_c, vel_gen_c)

# NEW:
kpts_gen_w_ = torch.einsum('bij,btpj->btpi', rotmat_c, kpts_gen_c) + transl_c.unsqueeze(-2)
acc_gen_w_ = torch.einsum('bij,btpj->btpi', rotmat_c, acc_gen_c)  # Changed from vel to acc
```

**H. Storage for Next Iteration (line ~1014-1017)**
```python
# OLD:
xb_gen.append(xb_gen_w_[:,:-1].detach().clone())
kpts_gen.append(kpts_gen_w_[:,:-1].detach().clone())
vel_gen.append(vel_gen_w_[:,:-1].detach().clone())
vel_gen_avg.append(vel_gen_w_[:,:-1].mean(dim=[1,2],keepdim=True).repeat(1,nt_tw-1,1,1).detach().clone())

# NEW:
# Note: We lose 2 frames with acceleration, so adjust indices
xb_gen.append(xb_gen_w_[:,:-2].detach().clone())  # Changed from :-1 to :-2
kpts_gen.append(kpts_gen_w_[:,:-2].detach().clone())
acc_gen.append(acc_gen_w_[:,:-2].detach().clone())  # Changed from vel to acc
acc_gen_avg.append(acc_gen_w_[:,:-2].mean(dim=[1,2],keepdim=True).repeat(1,nt_tw-2,1,1).detach().clone())
```

**I. Window Size Adjustment (line ~776, ~1023)**
```python
# OLD:
nt_tw = self.hparams.data.seq_len-1
tt += nt_tw-1

# NEW:
nt_tw = self.hparams.data.seq_len-2  # Lose 2 frames instead of 1
tt += nt_tw-2  # Adjust step size accordingly
```

**J. Visualization (lines ~1039-1075)**
```python
# OLD: Uses vel_gen for visualization rays
vel_dir = vel_gen / vel_gen.norm(dim=-1, keepdim=True).clamp(min=1e-8)
# ... creates rays from velocities

# NEW:
# Option 1: Still visualize velocities (compute from accelerations)
# Need to integrate accelerations to get velocities, or compute from keypoints
vel_gen = kpts_gen[:,1:] - kpts_gen[:,:-1]  # Recompute velocities for visualization
vel_dir = vel_gen / vel_gen.norm(dim=-1, keepdim=True).clamp(min=1e-8)

# Option 2: Visualize accelerations directly (different visualization)
acc_dir = acc_gen / acc_gen.norm(dim=-1, keepdim=True).clamp(min=1e-8)
# ... create rays from accelerations
```

---

## 3. Subclass Methods (ARDiffusionAction, ARDiffusionSpatial)

### Location: `primal/models/motion_diffuser.py` lines ~1091-2648

**All subclasses need the same changes as base ARDiffusion:**
- `ARDiffusionAction.forward()` (lines ~1176-1271)
- `ARDiffusionAction.generate_perpetual_navigation()` (lines ~1312-1700)
- `ARDiffusionSpatial.forward()` (lines ~1821-1914)
- `ARDiffusionSpatial.generate_perpetual_navigation()` (lines ~1959-2340)
- `ARDiffusionSpatial.generate_perpetual_navigation_ue()` (lines ~2346-2648)

**Apply all the same changes as outlined in sections 1 and 2.**

---

## 4. Configuration Updates

### Location: Config files (e.g., `primal/configs/model/motion_diffuser_ar.yaml`)

**Changes needed:**
```yaml
# OLD:
weight_vel: 1

# NEW:
weight_acc: 1  # Rename weight_vel to weight_acc
```

**Also consider:**
- Rename `use_l1_norm_vel` → `use_l1_norm_acc`
- Update motion representation name if desired (e.g., `smplx_jts_locs_acceleration_rotcont`)

---

## 5. Variable Naming Updates

### Throughout the codebase, rename:
- `vel` → `acc`
- `vel_seed` → `acc_seed`
- `vel_gen` → `acc_gen`
- `vel_pred_fk` → `acc_pred_fk`
- `loss_vel` → `loss_acc`
- `fn_dist_vel` → `fn_dist_acc` (or keep same function, just rename variable)
- `use_metric_velocity` → `use_metric_acceleration` (or keep name, just update scaling)
- `use_vel_perburbation` → `use_acc_perburbation`
- `guidance_weight_mv` → Keep same (it's for movement velocity, but now computed from acc)

---

## 6. Important Considerations

### A. Sequence Length Impact
- **Current**: Loses 1 frame → effective sequence length = `seq_len - 1`
- **New**: Loses 2 frames → effective sequence length = `seq_len - 2`
- **Impact**: Shorter sequences per window, may need to adjust `seq_len` in config

### B. Metric Scaling
- **Velocity**: Scales with `fps` (m/s)
- **Acceleration**: Scales with `fps²` (m/s²)
- **Update**: `acc = acc * (self.fps ** 2)` when `use_metric_velocity=True`

### C. Seed Computation in Autoregressive Generation
- **Challenge**: Need 3 consecutive frames to compute acceleration
- **Solution**: Store last 2 frames of previous window, or recompute from keypoints

### D. Classifier Guidance Interpretation
- **Current**: Guides average velocity direction
- **New**: Guides average acceleration direction (change in velocity)
- **Consideration**: May need different guidance weights or formulation

### E. Inertialization
- **Current**: Uses velocities for smooth transitions
- **New**: May need to integrate accelerations to velocities first, or adapt inertialization formula

### F. Visualization
- **Current**: Visualizes velocity rays
- **New**: Either recompute velocities from keypoints for visualization, or visualize accelerations directly

---

## 7. Summary Checklist

- [ ] Update `forward_one()`: Compute accelerations, update concatenation, update loss
- [ ] Update `generate_perpetual_navigation()`: All seed computations, guidance, output decomposition
- [ ] Update all subclass methods (`ARDiffusionAction`, `ARDiffusionSpatial`)
- [ ] Update config: Rename `weight_vel` → `weight_acc`
- [ ] Update variable names throughout (vel → acc)
- [ ] Adjust sequence length handling (lose 2 frames instead of 1)
- [ ] Update metric scaling (fps² instead of fps)
- [ ] Test seed computation in autoregressive generation
- [ ] Update visualization code
- [ ] Consider inertialization adaptation

---

## 8. Testing Considerations

1. **Training**: Verify loss decreases, check that acceleration values are reasonable
2. **Generation**: Test autoregressive generation, check smoothness between windows
3. **Guidance**: Test classifier guidance with accelerations
4. **Visualization**: Ensure visualization still works correctly
5. **Edge Cases**: Very short sequences, boundary conditions

---

## Notes

- The dimension `x_dim` stays the same (still n_kpts × 3 for accelerations)
- The model architecture doesn't need to change (same input/output size)
- Main changes are in data preprocessing and loss computation
- Be careful with frame indexing (lose 2 frames instead of 1)

