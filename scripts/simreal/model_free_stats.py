"""Everything about the sim/real gap that needs no forward pass.

Bounds how much of any downstream difference is distributional rather than
perceptual, and quantifies the two channels that are NOT images:

  * the state gap, in radians, in normalized units, and in the 256 discrete bins
    pi0.5 actually sees (state is spliced into the prompt as text);
  * the delta-action TARGET gap, which exists because DeltaActions subtracts the
    per-domain state from a commanded action that is otherwise bit-identical.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

REAL = "/home/ubuntu/training/data/teleop-data-hugo/piper_x_pick_cube_v1"
SIM = "/home/ubuntu/training/data/sim-data-1/piper_x_pick_cube_v1"
ASSETS = "/home/ubuntu/training/assets"
REAL_NS = f"{ASSETS}/pi05_piperx_teleop_expert/teleop-data-hugo/piper_x_pick_cube_v1/norm_stats.json"
SIM_NS = f"{ASSETS}/pi05_piperx_sim_expert/sim-data-1/piper_x_pick_cube_v1/norm_stats.json"
H, N_ARM = 30, 6
JOINTS = [f"j{i + 1}" for i in range(6)] + ["grip"]


def load(root, ep):
    df = pd.read_parquet(f"{root}/data/chunk-000/episode_{ep:06d}.parquet")
    st = np.stack(df["observation.state"].values).astype(np.float64)
    ac = np.stack(df["action"].values).astype(np.float64)
    st[:, 6] = np.clip(1.0 - st[:, 6] / 0.07, 0, 1)
    ac[:, 6] = np.clip(1.0 - ac[:, 6] / 0.07, 0, 1)
    return st, ac


def qnorm(x, q01, q99):
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def main() -> None:
    rns = json.load(open(REAL_NS))["norm_stats"]
    sq01 = np.array(rns["state"]["q01"])
    sq99 = np.array(rns["state"]["q99"])
    aq01 = np.array(rns["actions"]["q01"])
    aq99 = np.array(rns["actions"]["q99"])

    print("=" * 74)
    print("1. NORM STATS: how far apart are the two datasets before any model runs")
    print("=" * 74)
    try:
        sns_ = json.load(open(SIM_NS))["norm_stats"]
        for key in ("state", "actions"):
            r, s = rns[key], sns_[key]
            rr = np.array(r["q99"]) - np.array(r["q01"])
            sr = np.array(s["q99"]) - np.array(s["q01"])
            print(f"\n  {key}:")
            print(f"    q01 |real - sim| : {np.round(np.abs(np.array(r['q01']) - np.array(s['q01'])), 4)}")
            print(f"    q99 |real - sim| : {np.round(np.abs(np.array(r['q99']) - np.array(s['q99'])), 4)}")
            print(f"    range ratio sim/real : {np.round(sr / (rr + 1e-9), 3)}")
    except FileNotFoundError:
        print("  (sim norm stats not found -- skipping)")

    st_r, st_s, tg_r, tg_s = [], [], [], []
    for ep in range(49):
        sr, ar = load(REAL, ep)
        ss, asim = load(SIM, ep)
        assert np.array_equal(ar, asim), f"episode {ep}: absolute actions are not identical"
        st_r.append(sr)
        st_s.append(ss)
        T = len(sr)
        idx = np.clip(np.arange(T)[:, None] + np.arange(H)[None, :], 0, T - 1)
        for st, acc, dst in ((sr, ar, tg_r), (ss, asim, tg_s)):
            ch = acc[idx].copy()
            ch[:, :, :N_ARM] -= st[:, None, :N_ARM]
            dst.append(qnorm(ch, aq01, aq99))

    SR, SS = np.concatenate(st_r), np.concatenate(st_s)
    TR, TS = np.concatenate(tg_r), np.concatenate(tg_s)

    print("\n" + "=" * 74)
    print("2. STATE GAP (absolute actions are bit-identical; only state differs)")
    print("=" * 74)
    d = np.abs(SR - SS)
    print(f"\n  radians          mean {np.round(d.mean(0), 4)}")
    print(f"                   p95  {np.round(np.percentile(d, 95, axis=0), 4)}")
    print(f"                   max  {np.round(d.max(0), 4)}")
    dn = np.abs(qnorm(SR, sq01, sq99) - qnorm(SS, sq01, sq99))
    print(f"\n  normalized       mean {np.round(dn.mean(0), 4)}   (range is 2.0 wide)")

    bins = np.linspace(-1, 1, 257)[:-1]
    br = np.digitize(qnorm(SR, sq01, sq99), bins) - 1
    bs = np.digitize(qnorm(SS, sq01, sq99), bins) - 1
    db = np.abs(br - bs)
    print(f"\n  256-bin tokens   mean {np.round(db.mean(0), 2)}")
    print(f"                   p50  {np.percentile(db, 50, axis=0).astype(int)}")
    print(f"                   p95  {np.percentile(db, 95, axis=0).astype(int)}")
    print(f"    frames whose full state-token string is identical: {(db == 0).all(1).sum()}/{len(db)}")

    print("\n" + "=" * 74)
    print("3. TARGET GAP (DeltaActions turns the state gap into a LABEL difference)")
    print("=" * 74)
    dt = np.abs(TS - TR)
    print(f"\n  quantile-normalized (range 2.0 wide), per dim:")
    for i, name in enumerate(JOINTS):
        print(f"    {name:5s} mean {dt[..., i].mean():.4f}  p95 {np.percentile(dt[..., i], 95):.4f}  max {dt[..., i].max():.4f}")
    nd = np.linalg.norm(TS - TR, axis=-1).mean()
    nr = np.linalg.norm(TR, axis=-1).mean()
    print(f"\n  mean ||target_sim - target_real|| : {nd:.4f}")
    print(f"  mean ||target_real||              : {nr:.4f}")
    print(f"  ratio                             : {nd / nr:.3f}")
    print(f"  gripper is absolute, not delta, so its target gap is exactly {dt[..., 6].max():.1e}")

    print("\n" + "=" * 74)
    print("4. STRATIFICATION (does the gap live in a heavy tail?)")
    print("=" * 74)
    per_frame = db[:, :6].mean(1)
    qs = np.percentile(per_frame, [50, 75, 90, 95, 99])
    print(f"\n  per-frame mean arm state-bin distance: p50 {qs[0]:.1f}  p75 {qs[1]:.1f}  "
          f"p90 {qs[2]:.1f}  p95 {qs[3]:.1f}  p99 {qs[4]:.1f}")
    lo = per_frame <= qs[0]
    hi = per_frame >= qs[3]
    print(f"  target gap ||d|| in the bottom half : {np.linalg.norm(TS - TR, axis=-1)[lo].mean():.4f}")
    print(f"  target gap ||d|| in the top 5%      : {np.linalg.norm(TS - TR, axis=-1)[hi].mean():.4f}")
    print("\n  -> any real-vs-sim number that does not control for state is dominated by that tail.")


if __name__ == "__main__":
    main()
