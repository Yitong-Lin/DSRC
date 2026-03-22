# DSRC Paper Notes

> Paper: "DSRC: Learning Density-insensitive and Semantic-aware V2X Collaborative Representation against Corruptions" (AAAI 2025)
> arXiv: 2412.10739

---

## 1. Paper Module → Code Mapping

### Overall Architecture
The paper proposes a **teacher-student knowledge distillation** framework. Both branches share identical network structures but receive different inputs.

```
Input Point Cloud
    └─→ Encoder (PointPillars)
           └─→ Fusion (Multi-scale Feature Attention)
                  └─→ Detection Head (cls + reg)
```

---

### Teacher Model
**Paper:** Top branch in Figure 2. Takes multi-view dense "painted" point cloud P^T (x, y, z, r, s) where `s` is a semantic indicator (1=object, 0=background). Used only during training.

**Code:** `opencood/models/point_pillar_base_multi_scale_teacher.py`
- Class: `PointPillarBaseMultiScaleTeacher`
- `self.pillar_vfe_teacher` — PointPillar feature encoder, takes 5-channel input (x,y,z,r,s)
- `self.scatter_teacher` — scatters pillar features to BEV grid
- `self.backbone_teacher` — `ResNetBEVBackbone`, extracts multi-scale BEV features
- `self.cls_head` / `self.reg_head` — detection heads

---

### Student Model
**Paper:** Bottom branch in Figure 2. Takes original sparse point cloud P^S (x, y, z, r). This is the **only model retained at inference**.

**Code:** `opencood/models/point_pillar_base_multi_scale_student.py`
- Class: `PointPillarBaseMultiScaleStudent`
- `self.pillar_vfe` — PointPillar feature encoder, takes 4-channel input (x,y,z,r)
- `self.scatter` — scatters pillar features to BEV grid
- `self.backbone` — `ResNetBEVBackbone`, same architecture as teacher
- `self.cls_head` / `self.reg_head` — detection heads (shared with teacher)

Note: The student model file **contains both student and teacher** submodules during training. The teacher submodules are discarded at inference.

---

### Encoder — PointPillars
**Paper:** Converts raw point cloud into Bird's Eye View (BEV) feature maps F ∈ R^(H×W×C).

**Code:** `opencood/models/sub_modules/pillar_vfe.py`
- Used as `self.pillar_vfe` (student, 4 features) and `self.pillar_vfe_teacher` (teacher, 5 features)

---

### Backbone — ResNetBEVBackbone
**Paper:** Feature encoder that produces multi-scale features at different spatial resolutions.

**Code:** `opencood/models/sub_modules/base_bev_backbone_resnet.py`
- Class: `ResNetBEVBackbone`
- Uses **Res2Net** with `Bottle2neck` blocks for multi-scale feature extraction
- `get_multiscale_feature()` — extracts features at multiple scales before fusion (used for DAE distillation)
- `decode_multiscale_feature()` — upsamples and concatenates multi-scale features into final BEV feature map (used for DAF distillation)

---

### Multi-scale Feature Attention Fusion
**Paper:** Fuses BEV features from all collaborating agents using attention-based fusion. Produces fused feature H_i for ego agent.

**Code:** Inside `PointPillarBaseMultiScaleStudent.forward()`
- `self.fusion_net` — multi-scale attention fusion module
- Input: list of per-agent feature maps
- Output: fused feature map H^S_i (student) / H^T_i (teacher)

---

### Three-Stage Distillation (DAE → DAF → DAP)

#### DAE — Distillation After Encoding
**Paper:** Aligns per-agent local features F^S_j ≈ F^T_j using foreground mask M_j. Loss: L_d = Σ M_j · ||F^T_j − F^S_j||²

**Code:** `PointPillarBaseMultiScaleStudent.forward()`
- `fused_feature_list_student` vs `fused_feature_list_teacher` — pre-fusion multi-scale features
- `self.object_mask` (`Communication` module) — generates foreground binary mask M_j from detection scores
- KD loss computed between these feature lists → logged as `KD Loss` in training output

#### DAF — Distillation After Fusion
**Paper:** Aligns fused representations H^S_i ≈ H^T_i. Loss: L_h = ||H^T_i − H^S_i||²

**Code:** `PointPillarBaseMultiScaleStudent.forward()`
- `fused_feature` (student) vs `fused_feature_teacher` (teacher) — post-fusion feature maps
- MSE loss between these → logged as `Fused Loss` in training output

#### DAP — Distillation After Prediction
**Paper:** Aligns classification and regression outputs. Loss: L_p = D_KL(C^T_i, C^S_i) + D_KL(R^T_i, R^S_i)

**Code:** `PointPillarBaseMultiScaleStudent.forward()`
- `output_dict['cls_preds']` vs teacher cls output
- `output_dict['reg_preds']` vs teacher reg output
- KL divergence loss between student and teacher predictions → logged as part of total loss

---

### Point Cloud Reconstruction Module (PCR)
**Paper:** If fused features are reliable, they should be able to reconstruct the full scene. Decoupled into: (1) occupancy mask V_m prediction, (2) point offsets O_p prediction. Reconstructed cloud: P_c = (O_p + V_c) × V_m

**Code:**
- Class: `PCR` (referenced as `self.pcr` in student model)
- `mask_offset_loss()` in `PointPillarBaseMultiScaleStudent`:
  - Binary cross-entropy for mask prediction → logged as `mask Loss`
  - L1 loss for offset prediction → logged as `off Loss`
- Combined: `REC Loss` in training output = L_rec = L_m + L_o

---

### Detection Head
**Paper:** Outputs classification scores C and regression targets R (7-element: x, y, z, w, l, h, θ).

**Code:** `opencood/loss/point_pillar_loss.py`
- Class: `PointPillarLoss`
- Focal loss for classification
- Weighted Smooth L1 for regression
- `add_sin_difference()` — encodes rotation angle θ with sin/cos for periodicity
- Logged as `Conf Loss` + `Loc Loss` in training output

---

### Training Loss Summary

| Paper Symbol | Description | Code Log Label |
|---|---|---|
| L_detect | Detection loss (focal + smooth L1) | `Conf Loss` + `Loc Loss` |
| L_d (DAE) | Encoding-level KD (L2 with foreground mask) | `KD Loss` |
| L_h (DAF) | Fusion-level KD (L2) | `Fused Loss` |
| L_p (DAP) | Prediction-level KD (KL divergence) | part of total `Loss` |
| L_m | Occupancy mask BCE loss | `mask Loss` |
| L_o | Point offset L1 loss | `off Loss` |
| L_rec = L_m + L_o | Reconstruction loss | `REC Loss` |
| **L = L_detect + L_kd + L_rec** | **Total loss** | **`Loss`** |

Hyperparameters: α=1, β=1, γ=0.5 for L_kd = α·L_d + β·L_h + γ·L_p

---

## 2. Version Dependencies & Compatibility

### Core Dependencies

| Package | Required Version | Notes |
|---|---|---|
| Python | 3.7 | Strict — many packages drop 3.7 support |
| PyTorch | 1.9.1 | Strict — API changes in 1.10+ break code |
| CUDA | 11.1 / 11.3 | Must match torch build |
| torchvision | 0.10.1 | Paired with torch 1.9.1 |
| spconv | 2.x (cu113) | Voxel/pillar sparse convolution |
| numpy | 1.21.x | 1.24+ removes deprecated aliases |

### Most Likely Problem Areas

**1. PyTorch + CUDA mismatch (highest risk)**
The code uses `torch 1.9.1` built for CUDA 11.1. If your CUDA driver is newer (e.g. CUDA 12.x), the torch binary may still work but spconv and custom CUDA extensions may fail to compile or run.

Typical error:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
# or
RuntimeError: CUDA error: device-side assert triggered
# or during .cuda() call:
RuntimeError: CUDA error: out of memory   ← (misleading, often actually an arch mismatch)
```

**2. spconv version**
spconv 1.x and 2.x have completely different APIs. This repo uses spconv 2.x. Installing spconv 1.x will cause:
```
ImportError: cannot import name 'SparseConv3d' from 'spconv'
# or
AttributeError: module 'spconv' has no attribute 'xxx'
```

**3. numpy >= 1.24 drops deprecated type aliases**
numpy removed `np.float`, `np.int`, `np.bool` aliases in 1.24. OpenCOOD and older dependencies use these:
```
AttributeError: module 'numpy' has no attribute 'float'
```
Fix: pin `numpy==1.21.6` or use `numpy<1.24`.

**4. Python 3.7 EOL**
Many packages no longer publish wheels for Python 3.7, requiring compilation from source or using mirrors (this repo uses Tsinghua mirrors).

**5. open3d ml submodule dependencies**
`open3d 0.17.0` imports `sklearn` and `addict` at module load time via `open3d.ml`, even when you don't use those features. This causes cascading import errors if those packages are missing (encountered in this project).

---

## 3. Errors Encountered & Solutions

### Error 1: `ModuleNotFoundError: No module named 'torch'`
**When:** Running `python opencood/tools/train.py` directly
**Why:** The `dsrc` conda environment was not activated — system Python was used instead
**Fix:**
```bash
conda activate dsrc
# or
conda run -n dsrc python opencood/tools/train.py ...
```

---

### Error 2: `RuntimeError: CUDA out of memory`
**When:** Starting training on GPU 0
**Why:** A stale Python process from a previous disconnected session was still holding ~17 GB on GPU 0, leaving only ~5 GB for the new job
**Fix:**
```bash
nvidia-smi                          # identify the stale process PID
kill <PID>                          # free the GPU
# then restart training
```
**Prevention:** Always use `tmux` so sessions persist across disconnects and you can reattach instead of leaving orphan processes.

---

### Error 3: `ModuleNotFoundError: No module named 'sklearn'` (via open3d)
**When:** Running `inference.py`
**Why:** `open3d 0.17.0` unconditionally imports `open3d.ml` at the top level, which in turn imports `sklearn`. The `dsrc` env had open3d but not scikit-learn installed.
**Fix:**
```bash
pip install scikit-learn
```
But since `open3d` is only used for visualization (guarded by `--show_sequence` flag), a better fix was to move the import inside the conditional block:
```python
# Before (line 10, always runs):
import open3d as o3d

# After (only runs if visualization requested):
if opt.show_sequence:
    import open3d as o3d
    ...
```

---

### Error 4: Hardcoded `CUDA_VISIBLE_DEVICES=2` in `inference.py`
**When:** Running `CUDA_VISIBLE_DEVICES=3 python inference.py ...` — it still used GPU 2
**Why:** Line 5 of `inference.py` had:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 使得程序只看到2号GPU
```
This overrides any environment variable set before launching Python. GPU 2 was nearly full (24 GB used), causing OOM.
**Fix:** Remove the hardcoded line from `inference.py` so the external `CUDA_VISIBLE_DEVICES` takes effect.

---

### Error 5: `KeyError: 'gt_range'` during inference
**When:** Inference runs a few iterations then crashes
**Why:** `inference.py` has `--save_vis_n` defaulting to 10, which triggers visualization code that calls `hypes['postprocess']['gt_range']`. This key does not exist in `student_train_folder/config.yaml`.
**Fix:** Disable visualization by passing `--save_vis_n 0`:
```bash
CUDA_VISIBLE_DEVICES=3 python opencood/tools/inference.py \
    --model_dir /data/yitonglin/workspace/dsrc_test/student_train_folder \
    --save_vis_n 0
```
