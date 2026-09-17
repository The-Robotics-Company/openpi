"""Turn the Pass A dump into heatmaps and a ranked list of divergent regions.

The quantity plotted is the RATIO d_sr / d_rr, not raw cosine distance. Raw
distance is confounded by patch content: a flat, low-texture patch has low-norm
features and always looks "similar", while a busy one always looks different.
Dividing by the real-vs-real control (a different episode at the same normalized
phase) asks the question that matters -- is sim further from the real frame than
two real frames of the same moment already are?

  ratio ~ 1   sim sits inside the spread of real data at this patch
  ratio >> 1  the domains genuinely separate here

TWO THINGS THIS SCRIPT HAS TO CORRECT FOR:

1. Letterbox padding. resize_with_pad maps 480x640 onto 224x224, leaving 28px
   bars top and bottom -- patch rows 0, 1, 14, 15 contain NO scene content, and
   their pixels are bit-identical between the domains. They still show the
   HIGHEST divergence ratios in the image, because SigLIP attends globally and
   those tokens end up carrying a summary of the whole frame. They are real
   signal about the image as a whole but say nothing about that location, so
   including them in a spatial map is actively misleading. They are masked out
   of the map and reported separately.

2. Small denominators. Where the real-real control is tiny (a static region that
   looks the same in every episode), the ratio is fragile. Raw d_sr is reported
   alongside so a large ratio built on a near-zero denominator is visible.
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

GRID = 16
PATCH = 14
CAMERAS = ("base_0_rgb", "left_wrist_0_rgb")


def load_frame(frame_idx):
    from paired_loader import PairedFrames

    pf = PairedFrames()
    (r_obs, _), (s_obs, _) = pf.batch([frame_idx])
    to_rgb = lambda x: ((np.asarray(x)[0] + 1.0) / 2.0).clip(0, 1)
    return {c: (to_rgb(r_obs.images[c]), to_rgb(s_obs.images[c])) for c in CAMERAS}


def content_rows(img):
    """Patch rows that actually contain scene content, from the letterbox bars."""
    rowvar = img.reshape(img.shape[0], -1).std(1)
    nz = np.where(rowvar > 1e-6)[0]
    return nz.min() // PATCH, nz.max() // PATCH


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="/home/ubuntu/training/analysis/simreal/figures")
    ap.add_argument("--overlay-frame", type=int, default=None)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(args.npz)
    phase = d["phase"]
    print(f"frames in dump: {len(phase)}")

    overlay_idx = args.overlay_frame if args.overlay_frame is not None else int(d["frame"][len(phase) // 2])
    frames = load_frame(overlay_idx)

    fig, axes = plt.subplots(len(CAMERAS), 3, figsize=(13, 4.4 * len(CAMERAS)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    summary = {}

    for row, cam in enumerate(CAMERAS):
        sr_f = d[f"d_sr__{cam}"]
        rr_f = d[f"d_rr__{cam}"]
        sr = sr_f.mean(0).reshape(GRID, GRID)
        rr = rr_f.mean(0).reshape(GRID, GRID)
        r0, r1 = content_rows(frames[cam][0])
        mask = np.zeros((GRID, GRID), bool)
        mask[r0 : r1 + 1] = True
        summary[cam] = (sr_f, rr_f, mask)

        ratio = sr / (rr + 1e-8)
        c_sr, c_rr = sr[mask].mean(), rr[mask].mean()
        p_sr, p_rr = sr[~mask].mean(), rr[~mask].mean()

        print(f"\n--- {cam} ---")
        print(f"  content patch rows {r0}..{r1}  ({mask.sum()} of {GRID * GRID} patches)")
        print(f"  CONTENT   d_sr {c_sr:.5f}   d_rr {c_rr:.5f}   ratio {c_sr / c_rr:.2f}")
        print(f"  padding   d_sr {p_sr:.5f}   d_rr {p_rr:.5f}   ratio {p_sr / p_rr:.2f}"
              f"   <- no scene content; global summary tokens, excluded from the map")

        rc = ratio.copy()
        rc[~mask] = np.nan
        flat = np.where(mask.ravel())[0]
        order = flat[np.argsort(ratio.ravel()[flat])[::-1]][:8]
        print(f"  per-patch ratio (content only): min {np.nanmin(rc):.2f}  median {np.nanmedian(rc):.2f}  max {np.nanmax(rc):.2f}")
        print("  most divergent content patches (row,col | ratio | d_sr | d_rr):")
        for p in order:
            rr_v = rr.ravel()[p]
            warn = "  <- small denominator" if rr_v < 0.5 * c_rr else ""
            print(f"    ({p // GRID:2d},{p % GRID:2d})  {ratio.ravel()[p]:6.2f}   {sr.ravel()[p]:.5f}   {rr_v:.5f}{warn}")
        print(f"  content patches separating by >2x the control: {100 * (ratio[mask] > 2).mean():.1f}%")

        y0, y1 = r0 * PATCH, (r1 + 1) * PATCH
        axes[row, 0].imshow(frames[cam][0][y0:y1])
        axes[row, 0].set_title(f"{cam}\nreal (frame {overlay_idx})", fontsize=9)
        axes[row, 1].imshow(frames[cam][1][y0:y1])
        axes[row, 1].set_title("sim", fontsize=9)
        for a in axes[row, :2]:
            a.axis("off")
        im = axes[row, 2].imshow(ratio[r0 : r1 + 1], cmap="magma", interpolation="nearest")
        axes[row, 2].set_title("divergence / real-real control\n(letterbox rows dropped; aligned to the crop)", fontsize=9)
        axes[row, 2].axis("off")
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046)

    path = out / "patch_divergence.png"
    fig.savefig(path, dpi=130)
    print(f"\nwrote {path}")

    # Plot the numerator and denominator separately. The ratio alone spikes hard at
    # phase 0 and it is pure artifact: every episode starts from the same home pose, so
    # two real episodes are nearly identical there and the control collapses. The
    # sim-real distance itself is flat across the episode.
    fig2, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    bins = np.linspace(0, 1, 21)
    which = np.digitize(phase, bins) - 1
    centers = bins[:-1] + 0.025
    for cam, style in zip(CAMERAS, ("-", "--"), strict=True):
        sr_f, rr_f, mask = summary[cam]
        m = mask.ravel()
        sr = sr_f[:, m].mean(1)
        rr = rr_f[:, m].mean(1)
        binned = lambda v: [v[which == b].mean() if (which == b).any() else np.nan for b in range(len(bins) - 1)]
        axa.plot(centers, binned(sr), style, marker="o", ms=3, label=f"{cam}  sim-vs-real")
        axa.plot(centers, binned(rr), style, marker="s", ms=3, alpha=0.6, label=f"{cam}  real-vs-real control")
        axb.plot(centers, np.array(binned(sr)) / np.array(binned(rr)), style, marker="o", ms=3, label=cam)
    axa.set_xlabel("normalized episode phase")
    axa.set_ylabel("mean cosine distance")
    axa.set_title("Numerator and denominator separately")
    axa.legend(fontsize=7)
    axb.axhline(1.0, color="0.6", lw=1, ls="--")
    axb.set_xlabel("normalized episode phase")
    axb.set_ylabel("divergence ratio")
    axb.set_title("Ratio -- the phase-0 spike is the control\ncollapsing, not a larger gap")
    axb.legend(fontsize=8)
    path2 = out / "patch_divergence_by_phase.png"
    fig2.savefig(path2, dpi=130)
    print(f"wrote {path2}")


if __name__ == "__main__":
    main()
