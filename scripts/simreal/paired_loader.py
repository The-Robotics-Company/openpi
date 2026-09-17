"""Paired real/sim frame loader for the pi0.5 sim-vs-real analysis.

The whole analysis rests on the two domains being processed *identically*, so this
module deliberately reuses openpi's own dataset construction rather than
reimplementing the transform chain. Two things are pinned:

  * only `repo_id` differs between the two pipelines -- every transform, the
    gripper remap, the delta-action mask and `resize_with_pad` to 224 are shared;
  * both domains are normalized with ONE set of norm stats (the real ones), so
    the normalizer is never a confound.

Frame `i` is the same moment in both domains: the sim episodes are a replay of the
real commanded actions, and the per-episode lengths match 1:1. `assert_aligned`
checks that for free by comparing the *absolute* action columns, which are
bit-identical by construction.

NOTE on the targets: the model does not regress absolute actions. The piperx data
config applies `DeltaActions` over the 6 arm joints, so the target is
`action[t+k] - state[t]`, and only the gripper stays absolute. Because sim and real
state differ, the *targets* differ too. Downstream code treats that as a first-class
variable rather than an accident: every ablation is a tensor swap on the pair, so
the label can be held fixed independently of the observation.
"""

from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("HF_LEROBOT_HOME", "/home/ubuntu/training/data")

import numpy as np  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from openpi.training import data_loader as _data_loader  # noqa: E402

# The real config is the reference for BOTH domains: its transform chain and its norm
# stats are reused verbatim, and only repo_id is swapped for the sim pass.
REAL_CONFIG = "pi05_piperx_teleop_expert"
REAL_REPO_ID = "teleop-data-hugo/piper_x_pick_cube_v1"
SIM_REPO_ID = "sim-data-1/piper_x_pick_cube_v1"


def _build(repo_id: str, *, norm_stats=None):
    """Build the transformed dataset for one domain, using the real transform chain."""
    train_cfg = _config.get_config(REAL_CONFIG)
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

    def __init__(self):
        real_ds, real_cfg, train_cfg = _build(REAL_REPO_ID)
        # The sim pass inherits the REAL norm stats on purpose.
        sim_ds, _, _ = _build(SIM_REPO_ID, norm_stats=real_cfg.norm_stats)
        if len(real_ds) != len(sim_ds):
            raise RuntimeError(f"length mismatch: real {len(real_ds)} vs sim {len(sim_ds)}")
        self.real_ds = real_ds
        self.sim_ds = sim_ds
        self.train_cfg = train_cfg
        self.norm_stats = real_cfg.norm_stats
        # pi0.5 normalizes by quantiles, not z-score: config.py sets use_quantile_norm
        # for every model type except PI0. De-normalizing therefore needs q01/q99.
        self.use_quantile_norm = real_cfg.use_quantile_norm

    def __len__(self) -> int:
        return len(self.real_ds)

    def batch(self, indices):
        """Return ((real_obs, real_actions), (sim_obs, sim_actions)) for `indices`."""
        out = []
        for ds in (self.real_ds, self.sim_ds):
            items = [ds[int(i)] for i in indices]
            collated = _data_loader._collate_fn(items)  # noqa: SLF001 - reuse the training collate exactly
            out.append((_model.Observation.from_dict(collated), collated["actions"]))
        return out[0], out[1]


def denormalize(x, stats, *, use_quantiles: bool):
    """Invert openpi's Normalize over the leading dims that the stats cover."""
    x = np.asarray(x)
    if use_quantiles:
        q01, q99 = np.asarray(stats.q01), np.asarray(stats.q99)
        n = q01.shape[-1]
        return (x[..., :n] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    mean, std = np.asarray(stats.mean), np.asarray(stats.std)
    n = mean.shape[-1]
    return x[..., :n] * (std + 1e-6) + mean


def assert_aligned(pf: PairedFrames, indices, *, atol: float = 1e-4) -> float:
    """Frame i must be the same moment in both domains.

    The absolute action columns are bit-identical by construction, so recovering them
    from the delta targets (adding the state back on the arm joints) gives a free
    alignment check that does not depend on anything the analysis is measuring.
    """
    (r_obs, r_act), (s_obs, s_act) = pf.batch(indices)
    q = pf.use_quantile_norm

    def absolute(act, obs):
        act = denormalize(act, pf.norm_stats["actions"], use_quantiles=q).copy()
        state = denormalize(obs.state, pf.norm_stats["state"], use_quantiles=q)
        act[..., :6] += state[..., None, :6]  # undo DeltaActions on the arm joints
        return act

    diff = float(np.abs(absolute(r_act, r_obs) - absolute(s_act, s_obs)).max())
    if diff > atol:
        raise RuntimeError(f"real/sim frames are NOT aligned: absolute action max|diff| = {diff:.3e}")
    return diff
