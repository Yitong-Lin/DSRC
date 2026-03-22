# Project Log

## 2026-03-21 — Environment Setup & Teacher Training

### Clone Repository

```bash
mkdir -p ~/workspace/dsrc_test
cd ~/workspace/dsrc_test
git clone git@github.com:Yitong-Lin/DSRC.git
```

### Create Conda Environment

```bash
conda create -n dsrc python=3.7 -y
conda activate dsrc
conda install -n dsrc numpy -y

# PyTorch with CUDA 11.1
conda install -n dsrc pytorch==1.9.1=cuda111py37he371307_3 cudatoolkit=11.1 -c conda-forge -y

# Other dependencies
conda install -n dsrc tqdm tensorboard scikit-image matplotlib scipy -c conda-forge -y
```

### Install pip Packages

```bash
cd ~/workspace/dsrc_test/DSRC
pip install spconv-cu113
pip install -r requirements.txt
pip install open3d shapely

# Build bbx nms CUDA extension
python opencood/utils/setup.py build_ext --inplace

# Install opencood into environment
python setup.py develop
```

### Train Teacher Model

```bash
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
    --hypes_yaml opencood/hypes_yaml/point_pillar_base_multi_scale_teacher.yaml
```

Completed 8 epochs (checkpoints saved every 2 epochs: net_epoch1/3/5/7.pth).

### Setup Student Training Folder

Per README: copy teacher checkpoint folder, keep only last checkpoint renamed as `net_epoch1.pth`, update `core_method` to `point_pillar_base_multi_scale_student`.

```bash
cp -r teacher_train_folder student_train_folder
# kept only net_epoch7.pth, renamed to net_epoch1.pth
# edited student_train_folder/config.yaml: core_method -> point_pillar_base_multi_scale_student
```

---

## 2026-03-22 — Student Training & Inference

### Context
- Teacher model was already trained (`teacher_train_folder/`, epochs 1/3/5/7)
- Student model had been partially trained (epoch 7 interrupted at 57%)
- Goal: complete student training and run inference

---

### Resume Student Training

```bash
# Attempt 1 — failed: no conda env activated
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py --model_dir /data/yitonglin/workspace/dsrc_test/student_train_folder

# Attempt 2 — failed: CUDA OOM (stale process from previous session holding 17GB on GPU 0)
conda run -n dsrc bash -c "CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py --model_dir /data/yitonglin/workspace/dsrc_test/student_train_folder"

# Kill stale process
kill 1027520

# Attempt 3 — tried GPU 3 (then killed to switch back to GPU 0)
conda run -n dsrc bash -c "CUDA_VISIBLE_DEVICES=3 python opencood/tools/train.py --model_dir /data/yitonglin/workspace/dsrc_test/student_train_folder"

# Final: launched in tmux on GPU 0
tmux new-session -d -s dsrc_train
# inside tmux:
conda activate dsrc
cd /data/yitonglin/workspace/dsrc_test/DSRC
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py --model_dir /data/yitonglin/workspace/dsrc_test/student_train_folder
```

Training completed epoch 7 (3187/3187) in ~28 minutes.

---

### Fixes Applied to `inference.py`

1. Removed top-level `import open3d as o3d` (caused `ModuleNotFoundError`) — moved import inside `if opt.show_sequence:` block
2. Removed hardcoded `os.environ["CUDA_VISIBLE_DEVICES"] = "2"` at line 5

---

### Run Inference

```bash
# inside tmux session dsrc_train:
CUDA_VISIBLE_DEVICES=3 python opencood/tools/inference.py \
    --model_dir /data/yitonglin/workspace/dsrc_test/student_train_folder \
    --save_vis_n 0
```

Note: Used GPU 3 because GPU 0 was occupied by another user during inference.
`--save_vis_n 0` disables visualization (avoids `KeyError: gt_range`).

---

### Results

```
Epoch: 7 | AP @0.5: 0.8794 | AP @0.7: 0.7752
```

Output saved to: `student_train_folder/result.txt`
