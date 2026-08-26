# WristMimic

Official repository for **WristMimic** — physics-based whole-body human-object interaction control with
wrist-centric grasp supervision, built on top of
[InterMimic](https://github.com/Sirui-Xu/InterMimic) (CVPR 2025 Highlight) and trained on the
**ParaHome** dataset with SMPL-X humanoids.

---

## Installation

```bash
# 1. Create the conda environment and install PyTorch
conda create -n intermimic python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirement.txt

# (alternatively, build from the full environment file)
conda env create -f environment.yml
```

Then download and set up [Isaac Gym](https://developer.nvidia.com/isaac-gym), and activate the environment:

```bash
conda activate intermimic
```

## Data & Assets

| Item | Location |
|------|----------|
| Motion data (ParaHome) | `InterAct/Parahome/<sequence_name>/*.pt` |
| Humanoid XML (SMPL-X) | `intermimic/data/assets/parahome/sim_human/` |
| Object URDF / meshes | `intermimic/data/assets/parahome/sim_object/` |
| Checkpoints | `checkpoints/<experiment>/nn/` |

Each sequence folder under `InterAct/Parahome/` is named
`<subject>_<clip>_<object>_<action>` (e.g. `s79_1_chair_desk2table`), and the humanoid XML must match
the subject ID of the sequence (e.g. `s79_intermimic_ROM.xml` for `s79_*`).

### Object assets

The simulation-ready object assets are included in this repository under
`intermimic/data/assets/parahome/sim_object/`, one folder per object:

```
sim_object/<object>/
├── <object>.urdf       # rigid-body definition loaded by Isaac Gym
├── base_visual.obj     # visual mesh
└── base_collision.obj  # collision mesh
```

These meshes are derived from the object scans distributed with the
[ParaHome](https://github.com/snuvclab/ParaHome) dataset, so the original geometry for any of these
objects can also be obtained directly from the ParaHome release if you need the raw scans, additional
object categories, or per-part meshes. Articulated objects keep an extra full-articulation variant
(e.g. `drawer/drawer_full.urdf`) alongside the single-body URDF.

## Training

```bash
bash scripts/train_parahome.sh
```

Or run it directly:

```bash
python intermimic/run.py \
--task InterMimic_MULTI_OBJ \
--cfg_env intermimic/data/cfg/parahome_train.yaml \
--cfg_train intermimic/data/cfg/train/rlg/parahome.yaml \
--output checkpoints \
--num_envs 2048 \
--minibatch_size 16384 \
--motion_file InterAct/Parahome/s110_0_kettle_table2desk \
--robot_type sim_human/s110_intermimic_ROM.xml \
--experiment wristmimic_s110_0_kettle_table2desk \
--headless \
--num_position_iterations 20 \
--num_velocity_iterations 0
```

To resume from a checkpoint, add `--resume_from checkpoints/<experiment>/nn/<checkpoint>.pth`.

## Testing

```bash
bash scripts/test_parahome.sh
```

Or run it directly:

```bash
python intermimic/run.py \
--task InterMimic_MULTI_OBJ \
--cfg_env intermimic/data/cfg/parahome_test.yaml \
--cfg_train intermimic/data/cfg/train/rlg/parahome.yaml \
--test \
--checkpoint checkpoints/<experiment>/nn/<checkpoint>.pth \
--num_envs 1 \
--motion_file InterAct/Parahome/s110_0_kettle_table2desk \
--robot_type sim_human/s110_intermimic_ROM.xml \
--num_position_iterations 20 \
--num_velocity_iterations 0
--test_no_reset
```

## Key Flags

| Flag | Description |
|------|-------------|
| `--task` | Task class to instantiate (`InterMimic_MULTI_OBJ`) |
| `--cfg_env` / `--cfg_train` | Environment and PPO config YAMLs |
| `--motion_file` | ParaHome sequence directory to train / evaluate on |
| `--robot_type` | Humanoid XML, relative to `intermimic/data/assets/parahome/` |
| `--num_envs` | Number of parallel environments (2048 for training, 1–4 for testing) |
| `--experiment` | Experiment name used for checkpoint and log directories |
| `--headless` | Run without a viewer (server training) |
| `--test` | Inference mode |
| `--checkpoint` / `--resume_from` | Load weights for evaluation / resume training |

All grasp and reset parameters — grasp/finger windows, wrist position and rotation reset thresholds,
reset transition frames, and reward weights — are configured in
[parahome_train.yaml](intermimic/data/cfg/parahome_train.yaml) and
[parahome_test.yaml](intermimic/data/cfg/parahome_test.yaml), not via the command line.

## Monitoring

- **TensorBoard**: `tensorboard --logdir checkpoints/<experiment>/summaries`
- **Weights & Biases**: `wandb online` before training, `wandb offline` for evaluation

## Acknowledgement

This project builds on [InterMimic](https://github.com/Sirui-Xu/InterMimic) by Xu et al.
(CVPR 2025 Highlight) and uses the [ParaHome](https://github.com/snuvclab/ParaHome) dataset.
Please cite the original works if you use this code.

## License

See [LICENSE](LICENSE).
