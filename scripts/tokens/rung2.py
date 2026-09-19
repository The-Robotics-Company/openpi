"""Rung 2: do the policy's ACTIONS track the twin's once the tokens are applied?

Rung 1 says whether the features moved. This says whether that mattered to the thing downstream of
them. The policy is frozen and is never in any loss -- it is only being asked, on held-out frames,
whether `real + tokens` produces the action chunk that the twin frame would have produced.

Six conditions, all against the twin's own prediction as the reference:

  sim                      the reference: what the policy does on the twin frame
  real                     the deployment condition, untreated -- the gap to beat
  real+tokens              the deployment condition, treated
  real+random              control: is it the learning, or merely the presence of tokens?
  simstate+realimg         perception only, untreated (pi0.5 carries state in the prompt, so this
                           swaps pixels while holding the discretized state fixed)
  simstate+realimg+tokens  perception only, treated -- the cleanest read on what tokens can fix

Reported in normalized action space, which is what the model actually emits and is comparable
across conditions without a denormalization step that would differ between the domains.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import jax
import jax.numpy as jnp
import numpy as np

import pairs
from openpi.models import model as _model
from openpi.shared import nnx_utils

TOKENS = pathlib.Path("/home/ubuntu/training/tokens")
DEPLOY_CKPT = "/home/ubuntu/training/runs/pi05_piperx_sim_expert/pi0.5-ft01-simdata/9999"


def combined_tokens(k: int, suffix: str = "") -> np.ndarray | None:
    """Stack the per-camera runs for token count `k` into one (cameras, k, width) matrix."""
    sets = []
    for cam in pairs.CAMERAS:
        f = TOKENS / "runs" / f"{cam}-K{k}{suffix}" / "tokens.npz"
        if not f.exists():
            return None
        sets.append(np.load(f)["tokens"][0])
    return np.stack(sets).astype(np.float32)


def build_model(k: int, tokens: np.ndarray | None):
    """The policy, optionally with a prompt-token matrix spliced into the frozen tower."""
    from openpi.training import config as _config

    train_cfg = _config.get_config(pairs.SIM_CONFIG)
    if k:
        train_cfg = dataclasses.replace(
            train_cfg,
            model=dataclasses.replace(
                train_cfg.model, num_prompt_tokens=k, prompt_token_cameras=pairs.CAMERAS
            ),
        )
    params = _model.restore_params(pathlib.Path(DEPLOY_CKPT) / "params", dtype=jnp.bfloat16)
    if k:
        params = dict(params)
        params["PaliGemma"] = dict(params["PaliGemma"])
        img = dict(params["PaliGemma"]["img"])
        img["prompt_tokens"] = jnp.asarray(tokens, jnp.float32)
        params["PaliGemma"]["img"] = img
    model = train_cfg.model.load(params)
    model.eval()
    return model


def swap_images(target_obs, source_obs):
    """target's state/prompt with source's pixels."""
    return dataclasses.replace(target_obs, images=dict(source_obs.images))


def to_jax(obs):
    """The loader hands back numpy; the model's type contract wants jax arrays."""
    return dataclasses.replace(
        obs,
        images={k: jnp.asarray(v) for k, v in obs.images.items()},
        image_masks={k: jnp.asarray(v) for k, v in obs.image_masks.items()},
        state=jnp.asarray(obs.state),
        tokenized_prompt=None if obs.tokenized_prompt is None else jnp.asarray(obs.tokenized_prompt, jnp.int32),
        tokenized_prompt_mask=None
        if obs.tokenized_prompt_mask is None
        else jnp.asarray(obs.tokenized_prompt_mask),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-steps", type=int, default=10, help="flow-matching integration steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tokens = combined_tokens(args.k)
    if tokens is None:
        sys.exit(f"no complete token set for K={args.k}")
    rng_np = np.random.default_rng(args.seed)
    random_tokens = rng_np.normal(0, 0.02, tokens.shape).astype(np.float32)

    _, holdout = pairs.split_frames()
    stride = max(1, len(holdout) // args.max_frames)
    frames = holdout[::stride][: args.max_frames]
    print(f"Rung 2 on {len(frames)} held-out frames, K={args.k}", flush=True)

    pf = pairs.PairedFrames()

    def collect(model, conditions):
        """conditions: name -> fn(real_obs, sim_obs) -> Observation."""
        out = {name: [] for name in conditions}
        key = jax.random.key(args.seed)
        # The same wrapper the policy server uses. Calling sample_actions directly runs the 3.3B
        # model eagerly, op by op, at ~14 s per batch; this also makes the numbers here come from
        # the exact code path that serves the robot.
        sample = nnx_utils.module_jit(model.sample_actions)
        for i, (idx, batch) in enumerate(
            pairs.batches(pf, frames, batch_size=args.batch_size, num_workers=6)
        ):
            real_obs, _ = batch["real"]
            sim_obs, _ = batch["sim"]
            for name, fn in conditions.items():
                obs = to_jax(fn(real_obs, sim_obs))
                # Same noise draw for every condition, so differences are the observation alone.
                actions = sample(key, obs, num_steps=args.num_steps)
                out[name].append(np.asarray(actions, np.float32))
            if i % 10 == 0:
                print(f"  batch {i} ({(i + 1) * args.batch_size}/{len(frames)})", flush=True)
        return {k: np.concatenate(v) for k, v in out.items()}

    plain_conditions = {
        "sim": lambda r, s: s,
        "real": lambda r, s: r,
        "simstate+realimg": lambda r, s: swap_images(s, r),
        # The complement: real state with the twin's pixels. Tokens cannot touch the state channel,
        # so this is the part of the gap that is out of reach by construction, and having both
        # halves lets the total be decomposed instead of guessed at.
        "realstate+simimg": lambda r, s: swap_images(r, s),
    }
    token_conditions = {
        "real+tokens": lambda r, s: r,
        "simstate+realimg+tokens": lambda r, s: swap_images(s, r),
    }

    def run_stage(label, k, tok, conditions):
        """One policy, then let go of it: three 3.3B models resident at once is needless pressure."""
        print(f"loading policy: {label} ...", flush=True)
        model = build_model(k, tok)
        try:
            return collect(model, conditions)
        finally:
            del model
            gc.collect()
            jax.clear_caches()

    results = run_stage("untreated", 0, None, plain_conditions)
    results.update(run_stage(f"K={args.k} learned tokens", args.k, tokens, token_conditions))
    results.update(
        run_stage(f"K={args.k} RANDOM tokens", args.k, random_tokens, {"real+random": lambda r, s: r})
    )

    ref = results["sim"]
    report = {"k": args.k, "frames": len(frames), "num_steps": args.num_steps, "conditions": {}}
    baseline = float(np.abs(results["real"] - ref).mean())
    perc_baseline = float(np.abs(results["simstate+realimg"] - ref).mean())
    for name, a in results.items():
        d = np.abs(a - ref)
        entry = {"mean_abs_diff_vs_sim": float(d.mean()), "p95_abs_diff_vs_sim": float(np.percentile(d, 95))}
        if name in ("real+tokens", "real+random"):
            entry["pct_of_gap_removed"] = 100.0 * (1.0 - entry["mean_abs_diff_vs_sim"] / baseline)
        if name == "simstate+realimg+tokens":
            entry["pct_of_perceptual_gap_removed"] = 100.0 * (
                1.0 - entry["mean_abs_diff_vs_sim"] / perc_baseline
            )
        report["conditions"][name] = entry

    out = pathlib.Path(args.out or TOKENS / "reports" / f"rung2_K{args.k}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
