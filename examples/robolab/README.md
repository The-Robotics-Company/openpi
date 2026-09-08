# Fine-tuning π0.5 on RoboLab (IsaacLab) data

Pipeline: RoboLab recordings → LeRobot v2.1 dataset → openpi fine-tune → openpi policy server → RoboLab eval.
Everything here uses openpi's pinned LeRobot (v2.1); do not point it at v3.0 datasets.

## 1. Record in RoboLab

Run any RoboLab runner with image recording on so the HDF5 carries the exact policy observations:

```bash
cd ~/git/RoboLab
uv run python policies/pi0_family/run.py --policy pi05 --headless --num-envs 10 \
    --record-image-data --output-folder-name <run> --task <Task ...>
```

Without `--record-image-data` the converter falls back to post-step states plus the side-by-side mp4
(what our `pi05_isaac51_all120_n10` eval run has). It works, but the images are half resolution and
sample alignment is shifted by one step.

## 2. Convert to LeRobot v2.1

```bash
cd ~/git/openpi-train
uv run examples/robolab/convert_robolab_to_lerobot.py \
    --output-dir ~/git/RoboLab/output/<run> --repo-id trc/robolab_piperx --robot piperx --only-success
# Franka self-distillation from the pi05 eval run:
uv run examples/robolab/convert_robolab_to_lerobot.py \
    --output-dir ~/git/RoboLab/output/pi05_isaac51_all120_n10 --repo-id trc/robolab_franka --robot franka
```

Writes `$HF_LEROBOT_HOME/<repo_id>` (default `~/.cache/huggingface/lerobot/`). Set `HF_LEROBOT_HOME=/mnt/efs/datasets/lerobot_v21`
to keep datasets on EFS. Features: `exterior_image`, `wrist_image` (180×320), `joint_position`, `gripper_position`,
`actions` (absolute joint targets + continuous gripper, 0 open .. 1 closed, binarised only at execution), `task`. 15 fps.

## 2b. Existing LeRobot v3.0 Piper X sets (lerobot 0.4.x exports)

openpi's pinned LeRobot cannot read v3.0. Convert with:

```bash
HF_LEROBOT_HOME=/mnt/efs/datasets/lerobot_v21 uv run examples/robolab/convert_lerobot_v30_to_v21.py \
    --src /mnt/efs/datasets/lerobot_v21/_incoming/piperx_rubiks_cube_bowl --repo-id trc/robolab_piperx
```

It maps `observation.images.exo/wrist`, `observation.state.joint_position`, `action.joint_position` to the feature
names above, flips the gripper to RoboLab polarity (0 open, 1 closed) while keeping it continuous like DROID
(release 0.0, approach 0.18, hold-on-cube 0.27-0.33, close 0.97; the Piper eval client thresholds at ~0.22 with hysteresis, not DROID's 0.5), and drops frames whose joint action is all zeros (the first ~15 frames of each
episode in `piperx_rubiks_cube_bowl`). Current dataset of record: `trc/robolab_piperx` = 10 episodes, 1,397 frames,
task "put the rubiks cube in the bowl", from `s3://piperx-pick-cube-cup/piperx_rubiks_cube_bowl_lerobot.tar.gz`.

## 3. Norm stats

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_piperx        # -> assets/pi05_piperx/trc/robolab_piperx/norm_stats.json (done for the current set)
```

`pi05_robolab_franka` reuses the DROID stats from the sim checkpoint and needs no stats pass.

## 4. Train

| Config | Robot | Init | Trainable | GPU |
|---|---|---|---|---|
| `pi05_piperx` | Piper X, 6+1 | π0.5-base | everything | 80 GB (A100/H100) |
| `pi05_piperx_lora` | Piper X, 6+1 | π0.5-base | LoRA on both Gemma towers, EMA off | 48 GB L40S |
| `pi05_robolab_franka` | DROID Franka, 7+1 | `pi05_droid_jointpos` sim checkpoint | everything | 80 GB |

| `pi05_piperx_rubiks` | Piper X, 6+1 | π0.5-base | everything, **3k steps**, warmup 200, cosine to 3k, ckpt every 500 | 80 GB |
| `pi05_piperx_rubiks_lora` | Piper X, 6+1 | π0.5-base | same schedule, LoRA, batch 16, EMA off | 48 GB L40S |

The `*_rubiks` pair is the recommended run for the current 10-episode set (the 20k-step defaults would be ~450 epochs).
Pick the checkpoint by RoboLab rollouts, not by training loss.

All configs: action horizon 15 (1 s at 15 Hz), delta joint actions with absolute gripper, 20k steps,
default schedule (1k warmup → 2.5e-5, cosine → 2.5e-6 at 30k), EMA 0.99 unless LoRA.

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_piperx_lora --exp-name piperx_v1 --overwrite
# W&B is on by default (project name in TrainConfig.project_name); --no-wandb-enabled to turn off.
```

Checkpoints land in `checkpoints/<config>/<exp-name>/<step>/`. Resume with `--resume`.

## 5. Serve and evaluate

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piperx_lora \
    --policy.dir=checkpoints/pi05_piperx_lora/piperx_v1/20000
```

RoboLab talks to this server over the same websocket protocol as `pi05_droid_jointpos`; the Piper X eval needs a
client subclass that maps RoboLab's `exo_camera` / `wrist_camera` / `arm_joint_pos` / `gripper_pos` observations to
the request keys `observation/image`, `observation/wrist_image`, `observation/joint_position`,
`observation/gripper_position`, `prompt`, and executes 15 actions per query.

## Files

- `src/openpi/policies/robolab_policy.py` — `RoboLabInputs` / `RoboLabOutputs`
- `src/openpi/training/config.py` — `LeRobotRoboLabDataConfig`, the three `TrainConfig`s above
- `examples/robolab/convert_robolab_to_lerobot.py` — converter
- `examples/robolab/sky_train_piperx.yaml` — SkyPilot job for the full fine-tune (edit the dataset sync path first)
