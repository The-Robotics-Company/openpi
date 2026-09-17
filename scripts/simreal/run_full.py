"""Pass B: action-output difference and loss difference through the full model.

Everything that can be held fixed is held fixed, so the only thing varying between
the two branches is the observation:

  * the flow-matching noise and timestep are DERIVED from the frame index, not
    sampled, so real and sim see the same (noise, t);
  * the sampler gets the same initial noise array in both branches;
  * both domains are normalized with the real norm stats.

The one thing that cannot simply be held fixed is the target. The piperx config
regresses DELTA actions (action[t+k] - state[t]) over the 6 arm joints, and sim and
real state differ, so the targets differ by the state gap. That matters twice over,
because x_t = t*noise + (1-t)*actions is itself built from the target -- so even the
velocity prediction is contaminated unless the label is pinned. Hence two variants:

  label_fixed  both branches use the REAL target. The only difference is the
               observation, so ||dv|| and the loss gap are pure perception effects.
               This is the variant the headline numbers come from.
  as_trained   each branch uses its own target, i.e. what training/eval actually
               sees. The gap between this and label_fixed IS the label-shift term.

Sampled actions involve no target at all, so the action delta is pure-observation in
both variants and is reported once.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from paired_loader import PairedFrames, denormalize

PI05_BASE = "/home/ubuntu/training/models/openpi-assets/checkpoints/pi05_base/params"
SEED = 0
CAMERAS = ("base_0_rgb", "left_wrist_0_rgb")


def frame_rngs(indices, action_shape, *, offset=0):
    """Noise and timestep derived from the frame index -- identical across branches.

    `offset` shifts the seed, which is how the noise-floor control is produced: the
    same observation sampled from different initial noise. Any real-vs-sim delta
    smaller than that floor is below the model's own sampling variability.
    """
    noise, time = [], []
    for i in indices:
        k = jax.random.key(SEED + offset * 1_000_003 + int(i))
        k_n, k_t = jax.random.split(k)
        noise.append(jax.random.normal(k_n, action_shape))
        time.append(jax.random.beta(k_t, 1.5, 1) * 0.999 + 0.001)
    return jnp.stack(noise), jnp.stack(time)


@nnx.jit
def velocity_and_loss(model, obs, actions, noise, time):
    """Mirrors Pi0.compute_loss, but also returns v_t so ||dv|| can be measured."""
    obs = _model.preprocess_observation(None, obs, train=False)
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(obs, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
    loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # (b, horizon)
    return v_t, loss


@nnx.jit
def sample(model, obs, noise):
    return model.sample_actions(jax.random.key(0), obs, num_steps=10, noise=noise)


def swap(obs, other, *, images=None, prompt=False):
    """Build an observation from `obs` with selected channels taken from `other`."""
    new_images = dict(obs.images)
    new_masks = dict(obs.image_masks)
    for cam in images or ():
        new_images[cam] = other.images[cam]
        new_masks[cam] = other.image_masks[cam]
    kw = {"images": new_images, "image_masks": new_masks}
    if prompt:
        kw["tokenized_prompt"] = other.tokenized_prompt
        kw["tokenized_prompt_mask"] = other.tokenized_prompt_mask
        kw["state"] = other.state
    return dataclasses.replace(obs, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--params", default=PI05_BASE)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--ablations", action="store_true", help="also run the channel-swap ablations")
    ap.add_argument("--out", default="/home/ubuntu/training/analysis/simreal/passB")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pf = PairedFrames()
    print(f"loading {args.params} ...", flush=True)
    model = pf.train_cfg.model.load(_model.restore_params(args.params, dtype=jnp.bfloat16))
    model.eval()

    a_stats = pf.norm_stats["actions"]
    frames = list(range(0, len(pf), args.stride))
    if args.limit:
        frames = frames[: args.limit]
    print(f"frames: {len(frames)}", flush=True)

    records = []
    for start in range(0, len(frames), args.batch_size):
        idx = frames[start : start + args.batch_size]
        (r_obs, r_act), (s_obs, s_act) = pf.batch(idx)
        noise, time = frame_rngs(idx, np.asarray(r_act).shape[1:])

        # --- label_fixed: both branches regress the REAL target ---
        v_r, l_r = velocity_and_loss(model, r_obs, r_act, noise, time)
        v_s, l_s = velocity_and_loss(model, s_obs, r_act, noise, time)
        # --- as_trained: each branch regresses its own target ---
        _, l_s_own = velocity_and_loss(model, s_obs, s_act, noise, time)

        # --- sampled actions (no target involved), de-normalized to rad / m ---
        samp_noise = noise[..., : pf.train_cfg.model.action_dim]
        a_r = denormalize(sample(model, r_obs, samp_noise), a_stats, use_quantiles=pf.use_quantile_norm)
        a_s = denormalize(sample(model, s_obs, samp_noise), a_stats, use_quantiles=pf.use_quantile_norm)
        d_act = np.abs(a_r - a_s)

        # Noise floor: the SAME real observation, different initial noise. This is the
        # yardstick every other delta has to beat to mean anything.
        noise2, _ = frame_rngs(idx, np.asarray(r_act).shape[1:], offset=1)
        a_r2 = denormalize(sample(model, r_obs, noise2[..., : pf.train_cfg.model.action_dim]),
                           a_stats, use_quantiles=pf.use_quantile_norm)
        d_floor = np.abs(a_r - a_r2)

        dv = np.linalg.norm(np.asarray(v_r, np.float32) - np.asarray(v_s, np.float32), axis=-1)

        row = {
            "frame": idx,
            "time": np.asarray(time, np.float32),
            "loss_real": np.asarray(l_r, np.float32).mean(-1),
            "loss_sim_label_fixed": np.asarray(l_s, np.float32).mean(-1),
            "loss_sim_as_trained": np.asarray(l_s_own, np.float32).mean(-1),
            "dv_mean": dv.mean(-1),
            "dv_max": dv.max(-1),
            "dact_arm_mean_rad": d_act[..., :6].mean((1, 2)),
            "dact_arm_max_rad": d_act[..., :6].max((1, 2)),
            "dact_grip_mean_m": d_act[..., 6].mean(-1),
            "floor_arm_mean_rad": d_floor[..., :6].mean((1, 2)),
            "floor_grip_mean_m": d_floor[..., 6].mean(-1),
        }

        if args.ablations:
            # real images onto sim state, and each camera swapped on its own
            for name, obs in (
                ("state_from_real", swap(s_obs, r_obs, prompt=True)),
                ("ext_from_real", swap(s_obs, r_obs, images=["base_0_rgb"])),
                ("wrist_from_real", swap(s_obs, r_obs, images=["left_wrist_0_rgb"])),
            ):
                v_a, l_a = velocity_and_loss(model, obs, r_act, noise, time)
                row[f"loss_{name}"] = np.asarray(l_a, np.float32).mean(-1)
                row[f"dv_{name}"] = np.linalg.norm(
                    np.asarray(v_r, np.float32) - np.asarray(v_a, np.float32), axis=-1
                ).mean(-1)

        records.append(pd.DataFrame(row))
        if start % (args.batch_size * 20) == 0:
            print(f"  {start + len(idx)}/{len(frames)}", flush=True)

    df = pd.concat(records, ignore_index=True)
    path = out / f"passB_{args.tag}_stride{args.stride}.parquet"
    df.to_parquet(path)
    print(f"\nwrote {path}  ({len(df)} rows)")
    print(df.describe().T[["mean", "50%", "max"]].round(5))


if __name__ == "__main__":
    main()
