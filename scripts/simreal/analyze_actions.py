"""Turn the Pass B dump into the headline tables.

Three questions, in order of how directly they answer "how different are the
actions when sim is passed through instead of real":

  1. the sampled action delta, in radians and mm, against the model's own
     seed-to-seed noise floor. A delta below the floor is not a finding.
  2. the velocity sensitivity ||dv||, which is the same effect measured inside the
     flow field rather than after 10 integration steps.
  3. the loss gap, in both variants. label_fixed is the perception gap;
     as_trained also carries the label shift that DeltaActions introduces, and
     the difference between them isolates that shift.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

RAD2DEG = 180.0 / np.pi


def line(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"rows: {len(df)}")

    line("1. SAMPLED ACTION DELTA (no target involved -- pure observation effect)")
    arm = df["dact_arm_mean_rad"]
    floor = df["floor_arm_mean_rad"]
    print(f"\n  real vs sim, arm    mean {arm.mean():.5f} rad ({arm.mean() * RAD2DEG:.3f} deg)"
          f"   p95 {arm.quantile(0.95):.5f}   max {df['dact_arm_max_rad'].max():.5f}")
    print(f"  noise floor, arm    mean {floor.mean():.5f} rad ({floor.mean() * RAD2DEG:.3f} deg)")
    ratio = arm.mean() / max(floor.mean(), 1e-9)
    print(f"  ratio to floor      {ratio:.2f}x")
    verdict = ("ABOVE the model's own sampling noise -- the domains genuinely differ"
               if ratio > 1.5 else
               "at or BELOW the model's own sampling noise -- not a meaningful gap")
    print(f"  -> {verdict}")

    g, gf = df["dact_grip_mean_mm"], df["floor_grip_mean_mm"]
    print(f"\n  gripper real vs sim mean {g.mean():.2f} mm   floor {gf.mean():.2f} mm"
          f"   ratio {g.mean() / max(gf.mean(), 1e-9):.2f}x")

    if "dcmd_arm_mean_rad" in df:
        cmd = df["dcmd_arm_mean_rad"]
        print(f"\n  The rows above are the model's raw DELTA output. What the arm would")
        print(f"  actually be commanded is delta + state, and the state differs too:")
        print(f"    commanded arm     mean {cmd.mean():.5f} rad ({cmd.mean() * RAD2DEG:.3f} deg)"
              f"   max {df['dcmd_arm_max_rad'].max():.5f}")

    line("2. VELOCITY SENSITIVITY ||dv||  (label pinned to real)")
    print(f"\n  mean {df['dv_mean'].mean():.5f}   p95 {df['dv_mean'].quantile(0.95):.5f}   max {df['dv_max'].max():.5f}")
    if df["time"].nunique() > 5:
        df["_tbin"] = pd.cut(df["time"], np.linspace(0, 1, 6))
        print("\n  by flow timestep (t->0 is near the data, t->1 near pure noise):")
        for b, grp in df.groupby("_tbin", observed=True):
            print(f"    t in {str(b):14s} n={len(grp):5d}  ||dv|| {grp['dv_mean'].mean():.5f}")

    line("3. LOSS GAP")
    lr = df["loss_real"]
    lf = df["loss_sim_label_fixed"]
    lt = df["loss_sim_as_trained"]
    print(f"\n  loss real                 {lr.mean():.5f}")
    print(f"  loss sim (label_fixed)    {lf.mean():.5f}   gap {lf.mean() - lr.mean():+.5f}  <- perception only")
    print(f"  loss sim (as_trained)     {lt.mean():.5f}   gap {lt.mean() - lr.mean():+.5f}  <- what training sees")
    print(f"\n  label-shift contribution  {lt.mean() - lf.mean():+.5f}"
          f"  ({100 * abs(lt.mean() - lf.mean()) / max(abs(lt.mean() - lr.mean()), 1e-9):.0f}% of the as-trained gap)")
    sign = "LOWER" if lf.mean() < lr.mean() else "HIGHER"
    print(f"\n  sim loss is {sign} than real.")
    if lf.mean() < lr.mean():
        print("  Sim being the EASIER domain predicts a sim-trained policy that flatters")
        print("  itself in sim and degrades on real hardware.")

    abl = [c for c in df.columns if c.startswith("loss_") and c not in
           ("loss_real", "loss_sim_label_fixed", "loss_sim_as_trained")]
    if abl:
        line("4. ABLATIONS (which channel carries the gap)")
        base = lf.mean() - lr.mean()
        print(f"\n  full sim observation      gap {base:+.5f}   (100%)")
        for c in abl:
            gap = df[c].mean() - lr.mean()
            print(f"  {c.replace('loss_', ''):24s} gap {gap:+.5f}   ({100 * gap / base:5.1f}% of it remains)")
        print("\n  A row near 0% means swapping that channel back to real removes the gap,")
        print("  i.e. that channel was carrying it.")


if __name__ == "__main__":
    main()
