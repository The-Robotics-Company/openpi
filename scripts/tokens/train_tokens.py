"""Stage 2: train visual prompt tokens to make real features look like their digital twin's.

Only the K x D token matrix has gradients. The gradient still flows backwards through all 27 ViT
layers to reach it, but nothing else is updated and the policy is never loaded -- Gemma and the
action expert play no part, which is what keeps the job on one GPU and makes a positive result mean
one thing only (perception was realigned) rather than "behaviour drifted toward the real demos".

The tower is driven as a plain Flax module over the `PaliGemma/img` subtree of the deployment
checkpoint, rather than through the full Pi0 model, so 2.9B irrelevant parameters are never
touched.

Controls share this script so they differ from the treatment in exactly one flag:
  --steps 0    untrained random tokens: is it the learning, or merely the presence of tokens?
  --shuffle    targets permuted across frames: is it the pairing, or any feature-space pull?
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn  # noqa: F401  (kept for parity with siglip's namespace)

import pairs
from openpi.models import model as _model
from openpi.models import siglip as _siglip

TOKENS = pathlib.Path("/home/ubuntu/training/tokens")
DEPLOY_CKPT = "/home/ubuntu/training/runs/pi05_piperx_sim_expert/pi0.5-ft01-simdata/9999/params"


def load_img_params(checkpoint: str):
    """Just the SigLIP subtree of a checkpoint -- 2.9B irrelevant parameters are dropped."""
    params = _model.restore_params(checkpoint, dtype=jnp.bfloat16, restore_type=np.ndarray)
    img = jax.tree.map(jnp.asarray, params["PaliGemma"]["img"])
    del params
    return img


def make_module(num_prompt_tokens: int):
    """The tower's structure. Independent of the weights, which are the same for every K."""
    return _siglip.Module(
        num_classes=pairs.FEATURE_DIM,
        variant="So400m/14",
        pool_type="none",
        scan=True,
        dtype_mm="bfloat16",
        num_prompt_tokens=num_prompt_tokens,
        num_prompt_sets=1,
    )


def load_tower(checkpoint: str, num_prompt_tokens: int):
    return make_module(num_prompt_tokens), load_img_params(checkpoint)


def losses(pred, target, mean, std, *, cos_weight):
    """Standardized L2 plus a cosine term.

    ViT features carry a handful of very high-norm dimensions -- std ranges over ~11x here -- which
    would otherwise dominate a raw L2 and let the fit ignore most of the representation. The cosine
    term matches direction rather than magnitude, which is what downstream attention is primarily
    sensitive to.
    """
    zp = (pred - mean) / std
    zt = (target - mean) / std
    mse = jnp.mean(jnp.square(zp - zt))
    cos = 1.0 - jnp.mean(
        jnp.sum(pred * target, -1)
        / (jnp.linalg.norm(pred, axis=-1) * jnp.linalg.norm(target, axis=-1) + 1e-8)
    )
    return mse + cos_weight * cos, mse, cos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True, choices=pairs.CAMERAS)
    ap.add_argument("--num-tokens", type=int, required=True)
    ap.add_argument("--tag", default=None, help="run name; defaults to <camera>-K<k>")
    ap.add_argument("--steps", type=int, default=4000, help="0 = save the random init untrained")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--cos-weight", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--shuffle", action="store_true", help="control: break the real<->sim pairing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=DEPLOY_CKPT)
    args = ap.parse_args()

    tag = args.tag or f"{args.camera}-K{args.num_tokens}"
    out = TOKENS / "runs" / tag
    out.mkdir(parents=True, exist_ok=True)
    cam = args.camera

    imgs = np.load(TOKENS / "cache" / f"real__{cam}.npy", mmap_mode="r")
    tgts = np.load(TOKENS / "targets" / f"z_sim__{cam}.npy", mmap_mode="r")
    st = np.load(TOKENS / "targets" / f"stats__{cam}.npz")
    mean = jnp.asarray(st["mean"], jnp.float32)
    # A floor keeps a near-constant dimension from exploding when standardized.
    std = jnp.asarray(np.maximum(st["std"], 1e-3), jnp.float32)

    train_frames, holdout_frames = pairs.split_frames()
    rng = np.random.default_rng(args.seed)
    # The shuffled control keeps the same marginal distribution of targets and destroys only the
    # frame-to-frame correspondence, which is the single thing the treatment relies on.
    target_of = np.arange(len(imgs))
    if args.shuffle:
        target_of[train_frames] = rng.permutation(train_frames)

    module, img_params = load_tower(args.checkpoint, args.num_tokens)
    tokens = jax.random.normal(jax.random.key(args.seed), (1, args.num_tokens, 1152), jnp.float32) * 0.02

    # `img_params` is passed in rather than captured: 400M frozen weights baked into the jaxpr as
    # constants blow up compile time for no benefit.
    def forward(tok, weights, images, prompt_index):
        feats, _ = module.apply(
            {"params": {**weights, "prompt_tokens": tok}}, images, prompt_index=prompt_index
        )
        return feats.astype(jnp.float32)

    def loss_fn(tok, weights, images, target):
        total, _, _ = losses(forward(tok, weights, images, 0), target, mean, std, cos_weight=args.cos_weight)
        return total

    opt = optax.adam(args.lr)
    opt_state = opt.init(tokens)

    @jax.jit
    def step(tok, weights, state, images, target):
        loss, grad = jax.value_and_grad(loss_fn)(tok, weights, images, target)
        updates, state = opt.update(grad, state, tok)
        return optax.apply_updates(tok, updates), state, loss

    @jax.jit
    def eval_with(tok, weights, images, target):
        return losses(forward(tok, weights, images, 0), target, mean, std, cos_weight=args.cos_weight)

    @jax.jit
    def eval_without(tok, weights, images, target):
        # prompt_index=None: the parameter exists but no tokens are prepended, which is exactly the
        # untreated baseline this run has to beat.
        return losses(forward(tok, weights, images, None), target, mean, std, cos_weight=args.cos_weight)

    def batch_of(frames):
        return (
            jnp.asarray(np.asarray(imgs[frames], np.float32)),
            jnp.asarray(np.asarray(tgts[target_of[frames]], np.float32)),
        )

    eval_frames = holdout_frames[:: max(1, len(holdout_frames) // (args.eval_batches * args.batch_size))]
    eval_frames = eval_frames[: args.eval_batches * args.batch_size]
    eval_chunks = [eval_frames[i : i + args.batch_size] for i in range(0, len(eval_frames), args.batch_size)]

    def holdout_loss(tok, *, with_tokens=True):
        fn = eval_with if with_tokens else eval_without
        acc = np.zeros(3)
        for fr in eval_chunks:
            im, tg = batch_of(fr)
            acc += np.array([float(v) for v in fn(tok, img_params, im, tg)])
        return acc / len(eval_chunks)

    print(f"[{tag}] train {len(train_frames)}  holdout {len(holdout_frames)}  eval on {len(eval_frames)}", flush=True)
    base = holdout_loss(tokens, with_tokens=False)
    print(f"[{tag}] NO TOKENS  holdout total {base[0]:.5f}  mse {base[1]:.5f}  cos {base[2]:.5f}", flush=True)

    history = [{"step": 0, "holdout_total": base[0], "holdout_mse": base[1], "holdout_cos": base[2], "what": "no_tokens"}]
    best = (float("inf"), tokens, 0)
    t0 = time.time()

    # Measured unconditionally: with --steps 0 this IS the untrained-random-token control, and its
    # holdout number is that control's headline result rather than a discarded starting point.
    init = holdout_loss(tokens)
    history.append({"step": 0, "holdout_total": init[0], "holdout_mse": init[1], "holdout_cos": init[2], "what": "random_init"})
    print(f"[{tag}] RANDOM     holdout total {init[0]:.5f}  mse {init[1]:.5f}  cos {init[2]:.5f}", flush=True)
    best = (init[0], tokens, 0)

    if args.steps > 0:
        order = rng.permutation(train_frames)
        cursor = 0
        for s in range(1, args.steps + 1):
            if cursor + args.batch_size > len(order):
                order = rng.permutation(train_frames)
                cursor = 0
            fr = order[cursor : cursor + args.batch_size]
            cursor += args.batch_size
            im, tg = batch_of(fr)
            tokens, opt_state, loss = step(tokens, img_params, opt_state, im, tg)

            if s % args.eval_every == 0 or s == args.steps:
                h = holdout_loss(tokens)
                history.append({"step": s, "train_total": float(loss), "holdout_total": h[0],
                                "holdout_mse": h[1], "holdout_cos": h[2], "what": "train"})
                flag = ""
                if h[0] < best[0]:
                    best = (h[0], tokens, s)
                    flag = "  *best"
                rate = s / (time.time() - t0)
                print(f"[{tag}] step {s:5d}  train {float(loss):.5f}  holdout {h[0]:.5f} "
                      f"(mse {h[1]:.5f} cos {h[2]:.5f})  {rate:.2f} it/s{flag}", flush=True)

    final = np.asarray(best[1], np.float32)
    np.savez(out / "tokens.npz", tokens=final, cameras=np.array([cam]))
    summary = {
        "tag": tag, "camera": cam, "num_tokens": args.num_tokens, "steps": args.steps,
        "shuffled_control": args.shuffle, "lr": args.lr, "cos_weight": args.cos_weight,
        "batch_size": args.batch_size, "seed": args.seed, "checkpoint": args.checkpoint,
        "best_step": best[2], "eval_frames": len(eval_frames),
        "no_tokens": {"total": base[0], "mse": base[1], "cos": base[2]},
        "best_holdout": {"total": best[0]},
        "wall_s": time.time() - t0, "history": history,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[{tag}] saved {out}/tokens.npz  best holdout {best[0]:.5f} at step {best[2]} "
          f"(no tokens {base[0]:.5f})", flush=True)


if __name__ == "__main__":
    main()
