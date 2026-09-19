"""Stage 1: cache the digital-twin features the prompt tokens will regress onto.

One decode pass over the paired dataset produces everything Stage 2 needs:

  * `z_sim`  -- the frozen encoder's patch features for every SIM frame, per camera. These are the
    regression targets and they never change, so they are computed once.
  * `real_images` -- the preprocessed REAL frames. Real features cannot be cached: the tokens live
    inside the tower, so every training step has to run the real frame through the ViT with the
    tokens attached and backprop through it. Caching the *pixels* is what makes that affordable --
    video decode, not the GPU, was the bottleneck (17 frames/s with the GPU idle), and decoding
    once turns each later epoch from ~14 minutes into ~2.
  * per-dimension mean/std of the targets, over TRAIN frames only, for the standardized L2. ViT
    features carry a few very high-norm dimensions that otherwise dominate raw L2.

The encoder is loaded from the deployment checkpoint rather than from pi05_base. Finetuning did not
move the vision tower -- it is bit-identical `bfloat16(pi05_base)` in every checkpoint -- but
pi05_base stores fp32, so loading it would round differently from what actually serves.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import jax.numpy as jnp
import numpy as np
from flax import nnx

import pairs
from openpi.models import model as _model

DEPLOY_CKPT = "/home/ubuntu/training/runs/pi05_piperx_sim_expert/pi0.5-ft01-simdata/9999/params"
OUT = pathlib.Path("/home/ubuntu/training/tokens")


@nnx.jit
def encode(model, images):
    tokens, _ = model.PaliGemma.img(images, train=False)
    return tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="smoke test on the first N frames")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--checkpoint", default=DEPLOY_CKPT)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "targets").mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(parents=True, exist_ok=True)

    pf = pairs.PairedFrames()
    frames = np.arange(len(pf)) if args.limit is None else np.arange(args.limit)
    train_frames, holdout_frames = pairs.split_frames()
    is_train = np.zeros(len(pf), dtype=bool)
    is_train[train_frames] = True

    print(f"alignment check: max|diff| = {pairs.assert_aligned(pf, [0, 500, 9000, len(pf) - 1]):.3e}", flush=True)
    print(f"frames {len(frames)}  train {is_train[frames].sum()}  holdout {(~is_train[frames]).sum()}", flush=True)

    print(f"loading encoder from {args.checkpoint} ...", flush=True)
    model = pf.train_cfg.model.load(_model.restore_params(args.checkpoint, dtype=jnp.bfloat16))
    model.eval()

    n = len(frames)
    shape = (n, pairs.NUM_PATCHES, pairs.FEATURE_DIM)
    z = {
        cam: np.lib.format.open_memmap(
            out / "targets" / f"z_sim__{cam}.npy", mode="w+", dtype=np.float16, shape=shape
        )
        for cam in pairs.CAMERAS
    }
    imgs = {
        cam: np.lib.format.open_memmap(
            out / "cache" / f"real__{cam}.npy", mode="w+", dtype=np.float16, shape=(n, 224, 224, 3)
        )
        for cam in pairs.CAMERAS
    }
    # float64 accumulators: 2048 dims is nothing, and the sums run over ~3.8e6 patch vectors.
    acc = {cam: {"n": 0, "sum": np.zeros(pairs.FEATURE_DIM), "sq": np.zeros(pairs.FEATURE_DIM)} for cam in pairs.CAMERAS}
    peak = {cam: 0.0 for cam in pairs.CAMERAS}

    t0 = time.time()
    done = 0
    for idx, batch in pairs.batches(
        pf, frames, domains=("real", "sim"), batch_size=args.batch_size, num_workers=args.num_workers
    ):
        real_obs, _ = batch["real"]
        sim_obs, _ = batch["sim"]
        sl = slice(done, done + len(idx))
        for cam in pairs.CAMERAS:
            feats = np.asarray(encode(model, sim_obs.images[cam]), dtype=np.float32)
            peak[cam] = max(peak[cam], float(np.abs(feats).max()))
            z[cam][sl] = feats.astype(np.float16)
            imgs[cam][sl] = np.asarray(real_obs.images[cam], dtype=np.float32).astype(np.float16)

            tr = is_train[idx]
            if tr.any():
                f = feats[tr].reshape(-1, pairs.FEATURE_DIM).astype(np.float64)
                acc[cam]["n"] += f.shape[0]
                acc[cam]["sum"] += f.sum(0)
                acc[cam]["sq"] += (f * f).sum(0)

        done += len(idx)
        if done % (args.batch_size * 20) < args.batch_size or done == n:
            rate = done / (time.time() - t0)
            print(f"  {done}/{n}  {rate:.1f} frames/s  eta {(n - done) / rate / 60:.1f} min", flush=True)

    stats = {}
    for cam in pairs.CAMERAS:
        a = acc[cam]
        mean = a["sum"] / a["n"]
        var = np.maximum(a["sq"] / a["n"] - mean**2, 0.0)
        std = np.sqrt(var)
        np.savez(out / "targets" / f"stats__{cam}.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))
        stats[cam] = {
            "n_patch_vectors": int(a["n"]),
            "peak_abs_feature": peak[cam],
            "mean_of_means": float(mean.mean()),
            "std_min": float(std.min()),
            "std_max": float(std.max()),
            "std_ratio_max_over_median": float(std.max() / np.median(std)),
        }
        z[cam].flush()
        imgs[cam].flush()

    meta = {
        "checkpoint": args.checkpoint,
        "frames": int(n),
        "cameras": list(pairs.CAMERAS),
        "num_patches": pairs.NUM_PATCHES,
        "feature_dim": pairs.FEATURE_DIM,
        "holdout_episodes": list(pairs.HOLDOUT_EPISODES),
        "train_frames": int(is_train[frames].sum()),
        "holdout_frames": int((~is_train[frames]).sum()),
        "storage_dtype": "float16",
        "stats": stats,
        "wall_s": time.time() - t0,
    }
    (out / "targets" / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\n" + json.dumps(meta, indent=2))
    for cam in pairs.CAMERAS:
        if peak[cam] > 60000:
            print(f"WARNING: {cam} peak |feature| {peak[cam]:.0f} is near the float16 ceiling (65504)")


if __name__ == "__main__":
    main()
