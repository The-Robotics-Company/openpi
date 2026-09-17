"""Pass A: per-patch real-vs-sim divergence through the SigLIP tower alone.

No actions and no loss here -- SigLIP only turns images into patch embeddings.
For each paired frame this records, per camera and per patch:

    d_sr[p]  cosine distance between the real and sim embedding of patch p
    d_rr[p]  the same distance against a REAL control frame -- a different
             episode at the same normalized phase

d_rr is what makes d_sr readable. Raw cosine distance is confounded by patch
content (a flat, low-texture patch has low-norm features and always looks
"similar"), so the quantity worth reading is the ratio d_sr / d_rr: how far apart
the domains are *relative to* how far apart two real frames of the same moment
already are. A ratio near 1 means the sim is within the spread of real data at
that patch; much greater than 1 means the domains genuinely separate there.

PaliGemma uses SigLIP So400m/14, so 224px gives a 16x16 grid = 256 patches per
camera (not the 14x14/patch-16 the siglip.py default suggests -- the variant
string overrides it). That is 256 floats per map, small enough to run the full set.
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

from openpi.models import model as _model
from paired_loader import PairedFrames, paired_batches

PI05_BASE = "/home/ubuntu/training/models/openpi-assets/checkpoints/pi05_base/params"
REAL_META = "/home/ubuntu/training/data/teleop-data-hugo/piper_x_pick_cube_v1/meta/episodes.jsonl"
# right_wrist_0_rgb is an unused padding slot for this robot (all -1, masked off).
CAMERAS = ("base_0_rgb", "left_wrist_0_rgb")
# 224 / 14 = 16 patches per side. Verified empirically: the tower returns 256 tokens.
GRID = 16


def episode_index():
    lengths = [json.loads(line)["length"] for line in open(REAL_META)]
    starts = np.cumsum([0] + lengths)
    return np.array(lengths), starts


def control_index(i, lengths, starts):
    """A real frame from a different episode at the same normalized phase."""
    ep = int(np.searchsorted(starts, i, side="right") - 1)
    t = int(i - starts[ep])
    phase = t / max(lengths[ep] - 1, 1)
    ep2 = (ep + 1) % len(lengths)
    t2 = int(round(phase * (lengths[ep2] - 1)))
    return int(starts[ep2] + t2)


@nnx.jit
def encode(model, images):
    tokens, _ = model.PaliGemma.img(images, train=False)
    return tokens


def cosine_distance(a, b):
    """Per-patch cosine distance, computed in float32. a, b: (batch, patches, width)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
    return 1.0 - num / den


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--limit", type=int, default=None, help="stop after N frames")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", default="/home/ubuntu/training/analysis/simreal/passA")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("loading pi05_base ...", flush=True)
    pf = PairedFrames()
    model = pf.train_cfg.model.load(_model.restore_params(PI05_BASE, dtype=jnp.bfloat16))
    model.eval()

    lengths, starts = episode_index()
    frames = list(range(0, len(pf), args.stride))
    if args.limit:
        frames = frames[: args.limit]
    print(f"frames to process: {len(frames)} (stride {args.stride})", flush=True)

    rows, d_sr_all, d_rr_all = [], {c: [] for c in CAMERAS}, {c: [] for c in CAMERAS}
    controls = [control_index(i, lengths, starts) for i in frames]
    t0 = time.time()
    done = 0
    for idx, (r_obs, _), (s_obs, _), (c_obs, _) in paired_batches(
        pf, frames, controls, batch_size=args.batch_size, num_workers=args.num_workers
    ):
        for cam in CAMERAS:
            r = encode(model, r_obs.images[cam])
            s = encode(model, s_obs.images[cam])
            c = encode(model, c_obs.images[cam])
            d_sr_all[cam].append(cosine_distance(r, s))
            d_rr_all[cam].append(cosine_distance(r, c))

        for i in idx:
            ep = int(np.searchsorted(starts, i, side="right") - 1)
            rows.append((i, ep, int(i - starts[ep]), (i - starts[ep]) / max(lengths[ep] - 1, 1),
                         control_index(i, lengths, starts)))

        done += len(idx)
        if done % (args.batch_size * 20) < args.batch_size:
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(frames)}  {rate:.1f} frames/s  eta {(len(frames) - done) / rate / 60:.1f} min", flush=True)

    meta = np.array(rows, dtype=np.float64)
    payload = {"frame": meta[:, 0], "episode": meta[:, 1], "t": meta[:, 2], "phase": meta[:, 3], "control_frame": meta[:, 4]}
    for cam in CAMERAS:
        payload[f"d_sr__{cam}"] = np.concatenate(d_sr_all[cam]).astype(np.float32)
        payload[f"d_rr__{cam}"] = np.concatenate(d_rr_all[cam]).astype(np.float32)

    path = out / f"patches_stride{args.stride}.npz"
    np.savez_compressed(path, **payload)
    print(f"\nwrote {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    for cam in CAMERAS:
        sr = payload[f"d_sr__{cam}"]
        rr = payload[f"d_rr__{cam}"]
        print(f"  {cam:18s} patches {sr.shape[1]}  mean d_sr {sr.mean():.4f}  mean d_rr {rr.mean():.4f}  ratio {sr.mean() / rr.mean():.2f}")


if __name__ == "__main__":
    main()
