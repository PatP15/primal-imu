# Complete Guide to Creating a Custom ARDiffusion Class

This guide explains the entire architecture, data flow, preprocessing, and necessary components for creating your own custom version of the ARDiffusion class.

## Table of Contents
1. [Data Loading and Format](#data-loading-and-format)
2. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
3. [Motion Representation](#motion-representation)
4. [Model Architecture](#model-architecture)
5. [Training Process](#training-process)
6. [Inference/Generation](#inferencegeneration)
7. [Required Functions and Methods](#required-functions-and-methods)
8. [Key Considerations](#key-considerations)

---

## 1. Data Loading and Format

### Input Data Structure

The model expects data in the following format from the dataset:

```python
batch = {
    "betas": torch.Tensor,  # Shape: [batch_size, seq_len, 16] - SMPL-X shape parameters
    "xb": torch.Tensor,     # Shape: [batch_size, seq_len, 69] - SMPL-X pose parameters
    # Optional:
    "action_label": torch.Tensor,  # For action-conditioned models
    "ori": torch.Tensor,           # For spatial control models
}
```

### SMPL-X Parameter Breakdown (`xb`)

The `xb` tensor contains SMPL-X body parameters concatenated:
- **First 3 dims**: Root translation `[tx, ty, tz]` (global position)
- **Next 3 dims**: Root orientation (global rotation) in axis-angle format `[rx, ry, rz]`
- **Next 63 dims**: Body pose parameters (21 joints × 3 axis-angle) = `[j1_rx, j1_ry, j1_rz, ..., j21_rx, j21_ry, j21_rz]`

**Total**: 3 + 3 + 63 = 69 dimensions

### Dataset Classes

The codebase uses two main dataset classes:

1. **`AMASS_SMPLX_NEUTRAL`**: Loads AMASS dataset from `.npz` files
   - Reads: `trans`, `root_orient`, `pose_body`, `betas`, `jts_body`
   - Applies frame rate downsampling
   - Randomly samples sequences of fixed length

2. **`CustomizedActionMC`**: Loads custom `.smpl` files
   - Requires naming convention: `[action]_[idx].smpl`
   - Includes action labels for action-conditioned training

---

## 2. Data Preprocessing Pipeline

### Step-by-Step Preprocessing Flow

The preprocessing happens in the `forward_one()` method (or `forward()` for subclasses):

#### Step 1: Rotation Continuity Conversion (Optional)
```python
if 'rotcont' in self.mrepr:
    xb_all = self.aa2rotcont(xb_all)
```
- Converts axis-angle rotations to 6D rotation continuity representation
- **Why**: Better for neural networks (smoother gradients, no gimbal lock)
- **Conversion**: 3D axis-angle → 6D rotation matrix → 6D continuous representation

#### Step 2: Canonicalization
```python
xb = self.canonicalization(betas, xb_all, return_transf=False)
```
- **Purpose**: Transform motion to a canonical coordinate frame
- **Process**:
  1. Extract canonical frame from first frame's joint positions
  2. Compute rotation matrix and translation based on pelvis and shoulder positions
  3. Transform all SMPL-X parameters to this local frame
- **Result**: Motion is now in a body-centric coordinate system (Y-up, pelvis at origin)

#### Step 3: Forward Kinematics
```python
kpts = self._fwd_smplx_seq(betas, xb)
```
- **Purpose**: Compute joint/keypoint positions from SMPL-X parameters
- **Uses**: SMPL-X body model (SMPLXParser or SMPLXParserRotcont)
- **Output**: Joint positions `[batch, time, 22, 3]` (22 body joints × 3D coordinates)

#### Step 4: Velocity Computation
```python
vel = kpts[:,1:] - kpts[:,:-1]  # Finite difference
if self.use_metric_velocity:
    vel = vel * self.fps  # Convert to m/s if needed
```
- **Purpose**: Compute velocities from joint positions
- **Method**: Finite difference (next frame - current frame)
- **Optional**: Scale by FPS to get metric velocity (m/s)

#### Step 5: Sequence Alignment
```python
xb = xb[:,:-1]      # Remove last frame (no velocity for it)
kpts = kpts[:,:-1]  # Remove last frame
```
- **Why**: Velocities are computed between frames, so we lose one frame

#### Step 6: Concatenate Motion Representation
```python
xs = torch.cat([
    xb,                                    # SMPL-X params: [b, t, 69 or 135]
    kpts.reshape(nb, nt, -1),              # Joint positions: [b, t, 66] (22×3)
    vel.reshape(nb, nt, -1)                # Velocities: [b, t, 66] (22×3)
], dim=-1)
```
- **Final representation**: `[batch, time, x_dim]`
- **x_dim depends on representation**:
  - `smplx_jts_locs_velocity`: 3+3+63 + 66 + 66 = 201
  - `smplx_jts_locs_velocity_rotcont`: 3+6+126 + 132 + 132 = 399

---

## 3. Motion Representation

### Supported Representations

The model supports different motion representations via `motion_repr` config:

1. **`smplx_jts_locs_velocity`**:
   - SMPL-X params: 3 (trans) + 3 (root rot) + 63 (pose) = 69 dims
   - Joint locations: 22 joints × 3 = 66 dims
   - Velocities: 22 joints × 3 = 66 dims
   - **Total x_dim**: 201

2. **`smplx_jts_locs_velocity_rotcont`**:
   - SMPL-X params: 3 (trans) + 6 (root rot) + 126 (pose) = 135 dims
   - Joint locations: 22 joints × 3 = 66 dims
   - Velocities: 22 joints × 3 = 66 dims
   - **Total x_dim**: 267

3. **`smplx_ssm67_locs_velocity`**:
   - Uses SSM2-67 surface markers instead of joints
   - **Total x_dim**: 3+3+63 + 67×3 + 67×3 = 477

### Why This Representation?

- **SMPL-X params**: Direct control over body pose
- **Joint locations**: Spatial constraints, easier to enforce physical consistency
- **Velocities**: Temporal smoothness, momentum information

---

## 4. Model Architecture

### Base Class: `MotionDiffuserBase`

All diffusion models inherit from this base class which provides:

#### Core Components:
1. **EMA (Exponential Moving Average)**: Stabilizes training
2. **SMPL-X Parser**: Handles body model forward/backward kinematics
3. **Noise Scheduler**: DDPM scheduler for diffusion process
4. **Optimizer**: AdamW optimizer

#### Key Methods:
- `_setup_ema()`: Configure EMA weights
- `_setup_tokenizers()`: Initialize SMPL-X parser
- `_setup_noisescheduler()`: Initialize DDPM scheduler
- `configure_optimizers()`: Setup optimizer
- `training_step()`: Training loop
- `validation_step()`: Validation loop

### ARDiffusion Architecture

#### Denoiser Network

The denoiser is a Transformer-based model. Two main types:

1. **`TransformerAdaLN0`**:
   - Uses Adaptive Layer Normalization (AdaLN) conditioned on time + condition
   - DiT-style architecture
   - Input: `[batch, time, x_dim]`
   - Output: `[batch, time, x_dim]`

2. **`TransformerInContext`**:
   - Standard Transformer encoder
   - Condition tokens concatenated with input
   - Supports separate condition tokens or merged

#### Architecture Details:

```
Input [b, t, x_dim]
  ↓
Linear Projection → [b, t, h_dim]
  ↓
Positional Encoding (optional)
  ↓
Concatenate with Condition Embedding
  ↓
Transformer Encoder (N layers)
  ↓
Output Projection → [b, t, x_dim]
```

#### Condition Embedding

The motion seed (first frame) is embedded:
```python
xs_seed = xs[:,:1]  # First frame
cond_emb = self.emb_motionseed(xs_seed)  # [b, 1, h_dim]
```

This condition guides the generation of the rest of the sequence.

---

## 5. Training Process

### Forward Pass (Training)

```python
def forward_one(self, batch):
    # 1. Preprocess data (as described above)
    xs = ...  # [b, t, x_dim]
    xs_seed = xs[:,:1]  # Motion seed
    
    # 2. Sample random timestep
    timesteps = torch.randint(0, num_train_timesteps, (batch_size,))
    
    # 3. Add noise to clean data
    xs_noise = self.noise_scheduler.add_noise(xs, noise, timesteps)
    
    # 4. Forward pass through denoiser
    cond_emb = self.emb_motionseed(xs_seed)
    xs_pred = self.denoiser(timesteps, xs_noise, c_emb=cond_emb)
    
    # 5. Compute losses
    losses = {
        'loss_simple': MSE(xs_pred, xs),  # Direct prediction loss
        'loss_fk': MSE(kpts_pred_fk, kpts),  # Forward kinematics loss
        'loss_vel': MSE(vel_pred_fk, vel),  # Velocity consistency loss
    }
    losses['loss'] = weighted_sum(losses)
    
    return losses
```

### Loss Components

1. **`loss_simple`**: Direct MSE between predicted and target motion representation
2. **`loss_fk`**: Forward kinematics loss - ensures predicted SMPL-X params produce correct joint positions
3. **`loss_vel`**: Velocity consistency - ensures velocities match joint position differences

### Loss Weights

Configurable via:
- `weight_simple`: Weight for direct prediction loss
- `weight_fk`: Weight for forward kinematics loss
- `weight_vel`: Weight for velocity consistency loss

---

## 6. Inference/Generation

### Autoregressive Generation

The model generates motion autoregressively in windows:

```python
def generate_perpetual_navigation(self, batch, n_inference_steps=10, nt_max=1200):
    while tt < nt_max:
        # 1. Canonicalization: Transform to local coordinate
        xb_seed_c, rotmat_c, transl_c = self.canonicalization(...)
        
        # 2. Control Embedding: Encode motion seed
        cond_emb = self.emb_motionseed(xs_seed)
        
        # 3. Reverse Diffusion: Denoise random noise
        input = torch.randn(batch, nt_tw, x_dim)
        for t in scheduler.timesteps:
            model_output = self.denoiser(t, input, c_emb=cond_emb)
            input = scheduler.step(model_output, t, input).prev_sample
        
        # 4. Post-processing: Transform back to world coordinates
        xb_gen_w = self.smplx_parser.update_transl_glorot_seq(
            rotmat_c, transl_c, betas, xb_gen_c, fwd_transf=True
        )
        
        # 5. Inertialization (optional): Smooth transition between windows
        if switch_on_inertialization:
            output = inertialize(xs_seed, input, omega=10.0)
```

### Key Inference Steps:

1. **Canonicalization**: Each window is canonicalized to local frame
2. **Conditioning**: First frame of window is used as condition
3. **Diffusion**: Reverse diffusion process denoises random noise
4. **Coordinate Transform**: Transform back to world coordinates
5. **Inertialization**: Smooth blending between consecutive windows

---

## 7. Required Functions and Methods

### Must Implement in Custom Class

#### 1. `_setup_denoiser(self, cfg)`
**Purpose**: Initialize the denoiser network

**Required**:
- Set `self.mrepr` (motion representation type)
- Set `self.x_dim` (input/output dimension)
- Set `self.n_dim_rot` (rotation dimension: 3 for axis-angle, 6 for rotcont)
- Set `self.n_kpts` (number of keypoints: 22 for joints, 67 for markers)
- Initialize `self.denoiser` (Transformer model)

**Example**:
```python
def _setup_denoiser(self, cfg):
    self.mrepr = cfg.get('motion_repr', 'smplx_jts_locs_velocity_rotcont')
    if self.mrepr == 'smplx_jts_locs_velocity_rotcont':
        self.x_dim = 3 + 6 + 21*6 + 22*6  # 267
        self.n_dim_rot = 6
        self.n_kpts = 22
    
    self.denoiser = TransformerInContext(
        self.x_dim, self.x_dim,
        cfg.network.h_dim,
        cfg.network.n_layer,
        cfg.network.n_head,
        dropout=cfg.network.dropout,
        n_time_embeddings=cfg.scheduler.num_train_timesteps,
    )
```

#### 2. `_setup_controller(self, cfg)`
**Purpose**: Initialize condition embedding

**Required**:
- Initialize `self.emb_motionseed` (embeds motion seed to condition)

**Example**:
```python
def _setup_controller(self, cfg):
    self.emb_motionseed = nn.Linear(self.x_dim, cfg.network.h_dim)
```

#### 3. `_setup_additional(self, cfg)`
**Purpose**: Setup additional hyperparameters

**Required**:
- Set `self.canonical_tidx` (which frame to use for canonicalization)
- Set `self.use_l1_norm_fk` and `self.use_l1_norm_vel` (loss function types)
- Set `self.fps` (frame rate)
- Set `self.use_metric_velocity` (whether to scale velocities)

**Example**:
```python
def _setup_additional(self, cfg):
    self.canonical_tidx = 0  # Use first frame
    self.use_l1_norm_fk = cfg.get('use_l1_norm_fk', False)
    self.use_l1_norm_vel = cfg.get('use_l1_norm_vel', False)
    self.fps = cfg.data.framerate
    self.use_metric_velocity = cfg.get('use_metric_velocity', False)
```

#### 4. `forward_one(self, batch)` or `forward(self, batch)`
**Purpose**: Training forward pass

**Required Steps**:
1. Extract `betas` and `xb` from batch
2. Convert to rotcont if needed (`aa2rotcont`)
3. Canonicalize (`canonicalization`)
4. Forward kinematics (`_fwd_smplx_seq`)
5. Compute velocities
6. Concatenate representation (`xs`)
7. Extract seed (`xs_seed`)
8. Sample timesteps and add noise
9. Forward through denoiser
10. Compute losses (simple, FK, velocity)
11. Return losses dict

#### 5. Rotation Conversion Methods (if using rotcont)

**Required**:
- `aa2rotcont(xb)`: Convert axis-angle to rotation continuity
- `rotcont2aa(xb)`: Convert rotation continuity to axis-angle

**Example**:
```python
def aa2rotcont(self, xb):
    nb, nt = xb.shape[:2]
    transl, glorot_aa = xb[:,:,:3], xb[:,:,3:6]
    global_rotcont = RotConverter.aa2cont(glorot_aa)
    pose_aa = xb[:,:,6:]
    pose_rotcont = RotConverter.aa2cont(pose_aa.reshape(nb,nt,-1,3)).reshape(nb,nt,-1)
    return torch.cat([transl, global_rotcont, pose_rotcont], dim=-1)
```

### Inherited Methods (from Base Class)

These are already implemented but you should understand them:

1. **`canonicalization(betas, xb, return_transf=False)`**:
   - Transforms motion to canonical coordinate frame
   - Uses `smplx_parser.get_new_coordinate()` and `update_transl_glorot_seq()`

2. **`_fwd_smplx_seq(betas, xb, return_ssm2=False)`**:
   - Forward kinematics: SMPL-X params → joint positions
   - Returns joint positions or markers

3. **`snap_init_cond_to_ground(betas, xb)`**:
   - Projects initial condition to ground plane
   - Useful for preventing floating artifacts

---

## 8. Key Considerations

### Coordinate Systems

- **World Coordinates**: Original motion data
- **Canonical Coordinates**: Body-centric frame (Y-up, pelvis at origin)
- **Transformation**: Always canonicalize before processing, transform back after generation

### Motion Representation Choice

- **Axis-angle (`smplx_jts_locs_velocity`)**: Simpler, but has gimbal lock issues
- **Rotation continuity (`smplx_jts_locs_velocity_rotcont`)**: Better for neural networks, smoother gradients

### Loss Function Design

- **Direct loss (`loss_simple`)**: Ensures model learns motion representation
- **FK loss (`loss_fk`)**: Ensures physical consistency (SMPL-X params → joints)
- **Velocity loss (`loss_vel`)**: Ensures temporal smoothness

### Autoregressive Generation

- **Window size**: Typically `seq_len - 1` frames per window
- **Overlap**: Last frame of previous window becomes seed for next
- **Inertialization**: Critical for smooth transitions between windows

### Control Mechanisms

For conditional generation, you can add:

1. **Action Conditioning**: Embed action labels (see `ARDiffusionAction`)
2. **Spatial Control**: Embed target locations (see `ARDiffusionSpatial`)
3. **Classifier Guidance**: Use gradients to guide generation (see `generate_perpetual_navigation`)

### Performance Tips

1. **Use EMA**: Exponential moving average improves stability
2. **Metric Velocity**: Scale velocities by FPS for better physical consistency
3. **Ground Snapping**: Project to ground to prevent floating
4. **Inertialization**: Always use for smooth long-horizon generation

---

## Example: Minimal Custom ARDiffusion

```python
class MyCustomARDiffusion(ARDiffusion):
    def _setup_denoiser(self, cfg):
        # Set motion representation
        self.mrepr = 'smplx_jts_locs_velocity_rotcont'
        self.x_dim = 3 + 6 + 21*6 + 22*6  # 267
        self.n_dim_rot = 6
        self.n_kpts = 22
        
        # Initialize denoiser
        self.denoiser = TransformerInContext(
            self.x_dim, self.x_dim,
            cfg.network.h_dim,
            cfg.network.n_layer,
            cfg.network.n_head,
            dropout=cfg.network.dropout,
            n_time_embeddings=cfg.scheduler.num_train_timesteps,
            separate_condition_token=cfg.network.get('separate_condition_token', True),
            use_positional_encoding=cfg.network.get('use_positional_encoding', True),
        )
    
    def _setup_controller(self, cfg):
        # Condition embedding
        self.emb_motionseed = nn.Linear(self.x_dim, cfg.network.h_dim)
    
    def _setup_additional(self, cfg):
        # Inherit from parent
        super()._setup_additional(cfg)
        # Add any custom settings here
    
    # forward_one() is inherited from ARDiffusion base class
    # You can override it if you need custom training behavior
```

---

## Summary Checklist

When creating a custom ARDiffusion class, ensure you:

- [ ] Set `motion_repr` and compute `x_dim` correctly
- [ ] Initialize denoiser network in `_setup_denoiser()`
- [ ] Initialize condition embedding in `_setup_controller()`
- [ ] Configure hyperparameters in `_setup_additional()`
- [ ] Implement `forward_one()` or `forward()` for training
- [ ] Handle rotation conversion if using rotcont representation
- [ ] Implement generation method if needed (or inherit from base)
- [ ] Test canonicalization and coordinate transformations
- [ ] Verify loss computation (simple, FK, velocity)
- [ ] Test autoregressive generation with inertialization

---

## Additional Resources

- **SMPL-X Model**: https://github.com/vchoutas/smplx
- **Diffusion Models**: DDPM paper (Ho et al., 2020)
- **DiT Architecture**: Scalable Diffusion Models with Transformers
- **Rotation Continuity**: Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"

