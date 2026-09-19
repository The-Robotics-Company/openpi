"""Paired real/sim frame access for the sim-twin token experiment.

Adapted from the sim-vs-real analysis loader. The whole experiment rests on the two domains
being processed *identically*, so this reuses openpi's own dataset construction rather than
reimplementing the transform chain. Two things are pinned:

  * only `repo_id` differs between the two pipelines -- every transform, the gripper remap, the
    delta-action mask and `resize_with_pad` to 224 are shared;
  * both domains are normalized with ONE set of norm stats, so the normalizer is never a confound.

The reference config here is the SIM one, unlike the earlier analysis: the policy being adapted was
trained on sim and will normalize real state with sim statistics at deployment, so reading both
domains through the sim config is the deployment-faithful choice. For image features it makes no
difference at all -- norm stats touch state and actions, never pixels.

Frame `i` is the same moment in both domains: the sim episodes replay the real commanded actions
and the per-episode lengths match 1:1. `assert_aligned` checks that for free by comparing the
*absolute* action columns, which are bit-identical by construction.

Holdout is by whole trajectory. Adjacent frames at 30 Hz are near-duplicates, so a random frame
split would leak the answer across the boundary.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib

os.environ.setdefault("HF_LEROBOT_HOME", "/home/ubuntu/training/data")

import numpy as np  # noqa: E402
import torch.utils.data as _torch_data  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from openpi.training import data_loader as _data_loader  # noqa: E402

SIM_CONFIG = "pi05_piperx_sim_expert"
REAL_REPO_ID = "teleop-data-hugo/piper_x_pick_cube_v1"
SIM_REPO_ID = "sim-data-1/piper_x_pick_cube_v1"
REAL_ROOT = pathlib.Path("/home/ubuntu/training/data/teleop-data-hugo/piper_x_pick_cube_v1")

# right_wrist_0_rgb is an unused padding slot for this robot (all -1, masked off).
CAMERAS = ("base_0_rgb", "left_wrist_0_rgb")
NUM_PATCHES = 256  # 224 / 14 = 16 per side
FEATURE_DIM = 2048  # post-`head`: what Gemma actually consumes
NUM_EPISODES = 49
# Every 5th episode, so the holdout spans the whole recording session rather than one end of it.
HOLDOUT_EPISODES = tuple(range(0, NUM_EPISODES, 5))


def _build(repo_id: str, config_name: str, *, norm_stats=None):
    train_cfg = _config.get_config(config_name)
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    data_cfg = dataclasses.replace(
        data_cfg,
        repo_id=repo_id,
        norm_stats=norm_stats if norm_stats is not None else data_cfg.norm_stats,
    )
    ds = _data_loader.create_torch_dataset(data_cfg, train_cfg.model.action_horizon, train_cfg.model)
    return _data_loader.transform_dataset(ds, data_cfg), data_cfg, train_cfg


class PairedFrames:
    """Aligned real/sim views of the same 18431 frames."""

    def __init__(self, config_name: str = SIM_CONFIG):
        sim_ds, sim_cfg, train_cfg = _build(SIM_REPO_ID, config_name)
        # The real pass inherits the SIM norm stats on purpose -- see the module docstring.
        real_ds, _, _ = _build(REAL_REPO_ID, config_name, norm_stats=sim_cfg.norm_stats)
        if len(real_ds) != len(sim_ds):
            raise RuntimeError(f"length mismatch: real {len(real_ds)} vs sim {len(sim_ds)}")
        self.real_ds = real_ds
        self.sim_ds = sim_ds
        self.train_cfg = train_cfg
        self.norm_stats = sim_cfg.norm_stats
        self.use_quantile_norm = sim_cfg.use_quantile_norm

    def __len__(self) -> int:
        return len(self.real_ds)


def episode_bounds():
    """(lengths, starts) over the 49 episodes; real and sim agree by construction."""
    lengths = np.array([json.loads(line)["length"] for line in (REAL_ROOT / "meta/episodes.jsonl").open()])
    return lengths, np.concatenate([[0], np.cumsum(lengths)])


def episode_of(frames, starts):
    return np.searchsorted(starts, frames, side="right") - 1


def split_frames(holdout=HOLDOUT_EPISODES):
    """Frame indices for the train and holdout trajectory splits."""
    lengths, starts = episode_bounds()
    held = np.zeros(int(starts[-1]), dtype=bool)
    for ep in holdout:
        held[starts[ep] : starts[ep + 1]] = True
    all_frames = np.arange(len(held))
    return all_frames[~held], all_frames[held]


def denormalize(x, stats, *, use_quantiles: bool):
    x = np.asarray(x)
    if use_quantiles:
        q01, q99 = np.asarray(stats.q01), np.asarray(stats.q99)
        n = q01.shape[-1]
        return (x[..., :n] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    mean, std = np.asarray(stats.mean), np.asarray(stats.std)
    n = mean.shape[-1]
    return x[..., :n] * (std + 1e-6) + mean


class _Pairs(_torch_data.Dataset):
    def __init__(self, pf: PairedFrames, frames, *, domains=("real", "sim")):
        self.pf = pf
        # Plain ints: the LeRobot/datasets index rejects numpy integer scalars.
        self.frames = [int(i) for i in frames]
        self.domains = domains

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, k):
        i = self.frames[k]
        out = []
        for d in self.domains:
            out.append((self.pf.real_ds if d == "real" else self.pf.sim_ds)[i])
        return tuple(out)


def _collate(batch):
    n = len(batch[0])
    return tuple(_data_loader._collate_fn([b[k] for b in batch]) for k in range(n))  # noqa: SLF001


def batches(pf: PairedFrames, frames, *, domains=("real", "sim"), batch_size=16, num_workers=8):
    """Yield (indices, {domain: (Observation, actions)}) batches, decoded in worker processes.

    Video decoding, not the GPU, is the bottleneck: a single-process loop ran at 1.1 frames/s with
    the GPU idle at 0%.
    """
    ds = _Pairs(pf, frames, domains=domains)
    loader = _torch_data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    start = 0
    for parts in loader:
        n = len(parts[0]["actions"])
        idx = np.asarray(ds.frames[start : start + n])
        start += n
        yield idx, {
            d: (_model.Observation.from_dict(p), p["actions"]) for d, p in zip(ds.domains, parts, strict=True)
        }


def assert_aligned(pf: PairedFrames, frames, *, atol: float = 1e-4) -> float:
    """Frame i must be the same moment in both domains.

    The absolute action columns are bit-identical by construction, so recovering them from the
    delta targets (adding the state back on the arm joints) is a free alignment check that does not
    depend on anything the experiment is measuring.
    """
    _, batch = next(batches(pf, frames, batch_size=len(frames), num_workers=0))
    q = pf.use_quantile_norm

    def absolute(obs, act):
        act = denormalize(act, pf.norm_stats["actions"], use_quantiles=q).copy()
        state = denormalize(obs.state, pf.norm_stats["state"], use_quantiles=q)
        act[..., :6] += state[..., None, :6]
        return act

    real, sim = batch["real"], batch["sim"]
    diff = float(np.abs(absolute(*real) - absolute(*sim)).max())
    if diff > atol:
        raise RuntimeError(f"real/sim frames are NOT aligned: absolute action max|diff| = {diff:.3e}")
    return diff
