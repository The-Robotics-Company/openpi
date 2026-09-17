"""Step 0: verify the paired pipeline before spending any GPU time.

Checks that frame i is the same moment in both domains, that both branches come out
of the transform chain with identical shapes, and reports which parts of the model
input actually differ between the domains.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
from paired_loader import PairedFrames, assert_aligned


def main() -> None:
    pf = PairedFrames()
    print(f"paired frames: {len(pf)}")

    idx = [0, 1, 500, 5000, 12345, len(pf) - 1]
    assert_aligned(pf, idx)
    print(f"alignment OK on {idx} (absolute actions bit-identical)")

    (r_obs, r_act), (s_obs, s_act) = pf.batch(idx)

    print("\n--- shapes (must match) ---")
    for name in sorted(r_obs.images):
        r, s = np.asarray(r_obs.images[name]), np.asarray(s_obs.images[name])
        assert r.shape == s.shape, f"{name}: {r.shape} vs {s.shape}"
        print(f"  image {name:16s} {r.shape} {r.dtype}  range real [{r.min():.2f},{r.max():.2f}] sim [{s.min():.2f},{s.max():.2f}]")
    print(f"  state            {np.asarray(r_obs.state).shape}")
    print(f"  tokenized_prompt {np.asarray(r_obs.tokenized_prompt).shape}")
    print(f"  actions          {np.asarray(r_act).shape}")

    print("\n--- what differs between the domains ---")
    for name in sorted(r_obs.images):
        d = np.abs(np.asarray(r_obs.images[name]) - np.asarray(s_obs.images[name]))
        print(f"  image {name:16s} mean|d| {d.mean():.4f}  max|d| {d.max():.4f}   (inputs are in [-1,1])")

    rs, ss = np.asarray(r_obs.state)[:, :7], np.asarray(s_obs.state)[:, :7]
    print(f"  state (normalized)   mean|d| {np.abs(rs - ss).mean():.4f}  max|d| {np.abs(rs - ss).max():.4f}")

    rt, st = np.asarray(r_obs.tokenized_prompt), np.asarray(s_obs.tokenized_prompt)
    same = (rt == st).all(axis=1)
    print(f"  prompt tokens        identical on {same.sum()}/{len(same)} of these frames "
          f"(state is discretized INTO the prompt for pi0.5)")

    ra, sa = np.asarray(r_act)[..., :7], np.asarray(s_act)[..., :7]
    d = np.abs(ra - sa)
    print(f"  TARGET (delta act)   mean|d| {d.mean():.4f}  max|d| {d.max():.4f}   <- non-zero: DeltaActions")
    print(f"    per-dim mean|d|:   {np.round(d.mean((0, 1)), 4)}")
    print(f"    gripper (dim 6) is absolute, so its target diff is exactly {d[..., 6].max():.1e}")

    print("\n--- sanity: the two branches are NOT accidentally the same object ---")
    print(f"  images identical? {np.array_equal(np.asarray(r_obs.images[sorted(r_obs.images)[0]]), np.asarray(s_obs.images[sorted(s_obs.images)[0]]))}")
    print("\nstep 0 OK")


if __name__ == "__main__":
    main()
