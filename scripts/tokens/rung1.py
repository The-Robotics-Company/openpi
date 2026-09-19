"""Rung 1: did the sim->real feature gap actually close, on held-out trajectories?

Two measurements, both on the 10 holdout episodes the tokens never saw.

  * Cosine distance between the real and sim embedding of each patch, `d_sr`, reported against the
    distance between two *real* frames of the same moment in different episodes, `d_rr`. Raw cosine
    distance is confounded by patch content -- a flat, low-texture patch has low-norm features and
    always looks "similar" -- so the readable quantity is the ratio. A ratio near 1 means the sim
    is inside the spread real data already has; much above 1 means the domains genuinely separate.

  * A linear probe trained to tell sim from real. If the domains stop being linearly separable the
    accuracy falls toward chance, which is a different and harder-to-fake kind of evidence than a
    distance going down.

Both are reported over all 256 patches and over the 192 that carry image content: 480x640 padded
to 224x224 leaves two patch rows of constant grey at top and bottom. Those rows are NOT inert --
attention carries content into them -- but they are the one region where a token's effect is purely
attention-mediated, so they are worth separating rather than dropping.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import jax
import jax.numpy as jnp
import numpy as np

import pairs
import train_tokens as _tt

TOKENS = pathlib.Path("/home/ubuntu/training/tokens")
GRID = 16
CONTENT = np.zeros((GRID, GRID), bool)
CONTENT[2:14] = True
CONTENT = CONTENT.ravel()


def control_frames(frames, lengths, starts):
    """For each frame, a real frame at the same normalized phase in a different episode."""
    out = []
    for i in frames:
        ep = int(np.searchsorted(starts, i, side="right") - 1)
        t = int(i - starts[ep])
        phase = t / max(lengths[ep] - 1, 1)
        ep2 = (ep + 1) % len(lengths)
        out.append(int(starts[ep2] + round(phase * (lengths[ep2] - 1))))
    return np.array(out)


def cosine_distance(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
    return 1.0 - num / den


def encode_frames(module, weights, tok, images, frames, batch_size, prompt_index):
    def fwd(w, t, im):
        feats, _ = module.apply({"params": {**w, "prompt_tokens": t}}, im, prompt_index=prompt_index)
        return feats.astype(jnp.float32)

    jfwd = jax.jit(fwd)
    out = np.empty((len(frames), pairs.NUM_PATCHES, pairs.FEATURE_DIM), np.float16)
    for i in range(0, len(frames), batch_size):
        fr = frames[i : i + batch_size]
        im = jnp.asarray(np.asarray(images[fr], np.float32))
        out[i : i + len(fr)] = np.asarray(jfwd(weights, tok, im), np.float16)
    return out


def probe_accuracy(real, sim, episodes, seed=0, steps=400, l2=1e-3):
    """Linear probe, sim vs real, on mean-pooled patch features.

    Plain logistic regression trained with Adam -- a dependency-free stand-in for sklearn, which
    is not in this environment. The split is by EPISODE, not by frame: adjacent frames at 30 Hz are
    near-duplicates and a frame-level split would let the probe memorise rather than generalise.

    Chance is 0.5 by construction (one sim frame per real frame). An accuracy that falls toward
    0.5 means the domains have stopped being linearly separable.
    """
    eps = np.unique(episodes)
    rng = np.random.default_rng(seed)
    test_eps = set(rng.choice(eps, size=max(1, len(eps) // 2), replace=False).tolist())
    is_test = np.array([e in test_eps for e in episodes])

    x = np.concatenate([real.mean(1).astype(np.float32), sim.mean(1).astype(np.float32)])
    y = np.concatenate([np.zeros(len(real), np.float32), np.ones(len(sim), np.float32)])
    m = np.concatenate([is_test, is_test])

    mu, sd = x[~m].mean(0), x[~m].std(0) + 1e-6
    xtr, xte = jnp.asarray((x[~m] - mu) / sd), jnp.asarray((x[m] - mu) / sd)
    ytr, yte = jnp.asarray(y[~m]), jnp.asarray(y[m])

    def loss(p):
        logits = xtr @ p["w"] + p["b"]
        bce = jnp.mean(jnp.logaddexp(0.0, logits) - ytr * logits)
        return bce + l2 * jnp.sum(p["w"] ** 2)

    import optax

    params = {"w": jnp.zeros(xtr.shape[1]), "b": jnp.zeros(())}
    opt = optax.adam(1e-2)
    state = opt.init(params)

    @jax.jit
    def step(p, st):
        g = jax.grad(loss)(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st

    for _ in range(steps):
        params, state = step(params, state)

    pred = (xte @ params["w"] + params["b"]) > 0
    return float(jnp.mean((pred == (yte > 0.5)).astype(jnp.float32)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="run tags under tokens/runs; default all")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-frames", type=int, default=2000, help="subsample holdout for speed")
    ap.add_argument("--out", default=str(TOKENS / "reports" / "rung1.json"))
    args = ap.parse_args()

    lengths, starts = pairs.episode_bounds()
    _, holdout = pairs.split_frames()
    stride = max(1, len(holdout) // args.max_frames)
    frames = holdout[::stride][: args.max_frames]
    ctrl = control_frames(frames, lengths, starts)
    episodes = pairs.episode_of(frames, starts)
    print(f"holdout frames {len(holdout)} -> evaluating on {len(frames)} (stride {stride})", flush=True)

    run_dirs = sorted((TOKENS / "runs").iterdir()) if args.runs is None else [TOKENS / "runs" / r for r in args.runs]
    run_dirs = [d for d in run_dirs if (d / "tokens.npz").exists()]
    print(f"runs: {[d.name for d in run_dirs]}", flush=True)

    report = {"frames": len(frames), "holdout_episodes": list(pairs.HOLDOUT_EPISODES), "cameras": {}}
    weights = None
    modules: dict[int, object] = {}

    for cam in pairs.CAMERAS:
        imgs = np.load(TOKENS / "cache" / f"real__{cam}.npy", mmap_mode="r")
        z_sim_all = np.load(TOKENS / "targets" / f"z_sim__{cam}.npy", mmap_mode="r")
        z_sim = np.asarray(z_sim_all[frames])

        cam_runs = [d for d in run_dirs if json.loads((d / "summary.json").read_text())["camera"] == cam]
        # Any run's K works for the untreated pass: prompt_index=None skips the tokens entirely.
        k_for_base = json.loads((cam_runs[0] / "summary.json").read_text())["num_tokens"] if cam_runs else 4
        # The weights are identical for every K -- only the module structure differs -- so they are
        # restored once and shared across the whole sweep rather than per run.
        if weights is None:
            weights = _tt.load_img_params(_tt.DEPLOY_CKPT)
        module = modules.setdefault(k_for_base, _tt.make_module(k_for_base))
        dummy = jnp.zeros((1, k_for_base, 1152), jnp.float32)

        base = encode_frames(module, weights, dummy, imgs, frames, args.batch_size, None)
        ctrl_feats = encode_frames(module, weights, dummy, imgs, ctrl, args.batch_size, None)
        d_rr = cosine_distance(base, ctrl_feats)
        d_sr_base = cosine_distance(base, z_sim)

        def summarize(d_sr):
            return {
                "d_sr_all": float(d_sr.mean()),
                "d_sr_content": float(d_sr[:, CONTENT].mean()),
                "ratio_all": float(d_sr.mean() / d_rr.mean()),
                "ratio_content": float(d_sr[:, CONTENT].mean() / d_rr[:, CONTENT].mean()),
            }

        entry = {
            "d_rr_all": float(d_rr.mean()),
            "d_rr_content": float(d_rr[:, CONTENT].mean()),
            "no_tokens": summarize(d_sr_base),
            "runs": {},
        }
        entry["no_tokens"]["probe_acc"] = probe_accuracy(base, z_sim, episodes)
        print(f"\n[{cam}] NO TOKENS  ratio_content {entry['no_tokens']['ratio_content']:.3f}  "
              f"probe {entry['no_tokens']['probe_acc']:.3f}", flush=True)
        del ctrl_feats

        for d in cam_runs:
            meta = json.loads((d / "summary.json").read_text())
            tok = np.load(d / "tokens.npz")["tokens"]
            mod_k = modules.setdefault(meta["num_tokens"], _tt.make_module(meta["num_tokens"]))
            feats = encode_frames(mod_k, weights, jnp.asarray(tok), imgs, frames, args.batch_size, 0)
            r = summarize(cosine_distance(feats, z_sim))
            r["probe_acc"] = probe_accuracy(feats, z_sim, episodes)
            r["gap_closed_pct"] = 100.0 * (1.0 - (r["ratio_content"] - 1.0) /
                                           max(entry["no_tokens"]["ratio_content"] - 1.0, 1e-9))
            r["num_tokens"] = meta["num_tokens"]
            r["steps"] = meta["steps"]
            r["shuffled_control"] = meta["shuffled_control"]
            entry["runs"][d.name] = r
            print(f"[{cam}] {d.name:32s} ratio_content {r['ratio_content']:.3f}  "
                  f"probe {r['probe_acc']:.3f}  gap closed {r['gap_closed_pct']:+.1f}%", flush=True)
            del feats

        report["cameras"][cam] = entry
        del base, z_sim

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
