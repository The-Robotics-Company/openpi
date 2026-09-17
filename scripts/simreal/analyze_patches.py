"""Turn the Pass A dump into heatmaps and a ranked list of divergent regions.

The quantity plotted is the RATIO d_sr / d_rr, not raw cosine distance. Raw
distance is confounded by patch content: a flat, low-texture patch has low-norm
features and always looks "similar", while a busy one always looks different.
Dividing by the real-vs-real control (a different episode at the same normalized
phase) asks the question that actually matters -- is the sim further from the real
frame than two real frames of the same moment already are?

  ratio ~ 1   sim sits inside the spread of real data at this patch
  ratio >> 1  the domains genuinely separate here
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GRID = 16  # SigLIP So400m/14 at 224px
CAMERAS = ("base_0_rgb", "left_wrist_0_rgb")


def load_frame_images(frame_idx):
    """Fetch one real/sim image pair for the overlay background."""
    from paired_loader import PairedFrames

    pf = PairedFrames()
    (r_obs, _), (s_obs, _) = pf.batch([frame_idx])
    to_rgb = lambda x: ((np.asarray(x)[0] + 1.0) / 2.0).clip(0, 1)
    return {c: (to_rgb(r_obs.images[c]), to_rgb(s_obs.images[c])) for c in CAMERAS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="/home/ubuntu/training/analysis/simreal/figures")
    ap.add_argument("--overlay-frame", type=int, default=None, help="frame index to use as heatmap background")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(args.npz)
    phase = d["phase"]
    print(f"frames in dump: {len(phase)}\n")

    overlay_idx = args.overlay_frame if args.overlay_frame is not None else int(d["frame"][len(phase) // 2])
    try:
        backgrounds = load_frame_images(overlay_idx)
    except Exception as exc:  # noqa: BLE001 - the overlay is a nicety, the numbers are not
        print(f"(could not load overlay images: {exc})")
        backgrounds = None

    fig, axes = plt.subplots(len(CAMERAS), 3, figsize=(13, 4.4 * len(CAMERAS)), constrained_layout=True)
    axes = np.atleast_2d(axes)

    summary = {}
    for row, cam in enumerate(CAMERAS):
        sr = d[f"d_sr__{cam}"]           # (frames, 256)
        rr = d[f"d_rr__{cam}"]
        ratio = sr.mean(0) / (rr.mean(0) + 1e-8)
        summary[cam] = (sr, rr, ratio)

        print(f"--- {cam} ---")
        print(f"  mean d_sr {sr.mean():.4f}   mean d_rr {rr.mean():.4f}   overall ratio {sr.mean() / rr.mean():.2f}")
        print(f"  per-patch ratio: min {ratio.min():.2f}  median {np.median(ratio):.2f}  max {ratio.max():.2f}")
        top = np.argsort(ratio)[::-1][:8]
        print(f"  most divergent patches (row,col,ratio): "
              + ", ".join(f"({p // GRID},{p % GRID},{ratio[p]:.1f})" for p in top))
        frac = (ratio > 2).mean()
        print(f"  patches where the domains separate by >2x the real-real control: {100 * frac:.1f}%\n")

        if backgrounds is not None:
            axes[row, 0].imshow(backgrounds[cam][0])
            axes[row, 0].set_title(f"{cam}\nreal (frame {overlay_idx})", fontsize=9)
            axes[row, 1].imshow(backgrounds[cam][1])
            axes[row, 1].set_title("sim", fontsize=9)
            for a in axes[row, :2]:
                a.axis("off")

        hm = ratio.reshape(GRID, GRID)
        im = axes[row, 2].imshow(hm, cmap="magma", interpolation="nearest")
        axes[row, 2].set_title("sim-real divergence / real-real control", fontsize=9)
        axes[row, 2].axis("off")
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046)

    path = out / "patch_divergence.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")

    # Does the gap track where you are in the episode?
    fig2, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    bins = np.linspace(0, 1, 21)
    which = np.digitize(phase, bins) - 1
    for cam in CAMERAS:
        sr, rr, _ = summary[cam]
        per_frame = sr.mean(1) / (rr.mean(1) + 1e-8)
        curve = [per_frame[which == b].mean() if (which == b).any() else np.nan for b in range(len(bins) - 1)]
        ax.plot(bins[:-1] + 0.025, curve, marker="o", ms=3, label=cam)
    ax.axhline(1.0, color="0.6", lw=1, ls="--")
    ax.set_xlabel("normalized episode phase")
    ax.set_ylabel("divergence ratio")
    ax.set_title("Where in the episode do the domains separate?")
    ax.legend(fontsize=8)
    path2 = out / "patch_divergence_by_phase.png"
    fig2.savefig(path2, dpi=130)
    print(f"wrote {path2}")


if __name__ == "__main__":
    main()
