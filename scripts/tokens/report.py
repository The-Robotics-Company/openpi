"""Render the experiment's results as one readable table."""

from __future__ import annotations

import json
import pathlib

TOKENS = pathlib.Path("/home/ubuntu/training/tokens")


def load(p):
    try:
        return json.loads(pathlib.Path(p).read_text())
    except Exception:
        return None


def stage2():
    print("=" * 100)
    print("STAGE 2 -- token training, held-out feature loss (standardized L2 + cosine)")
    print("=" * 100)
    print(f"{'run':34s} {'K':>4s} {'steps':>6s} {'no tokens':>11s} {'random':>10s} {'trained':>10s} {'drop':>8s}")
    rows = sorted((TOKENS / "runs").glob("*/summary.json"))
    for f in rows:
        d = load(f)
        if not d:
            continue
        rnd = next((h["holdout_total"] for h in d["history"] if h["what"] == "random_init"), float("nan"))
        base, best = d["no_tokens"]["total"], d["best_holdout"]["total"]
        drop = 100.0 * (1 - best / base)
        note = "  [shuffled-pairs control]" if d["shuffled_control"] else ("  [random control]" if d["steps"] == 0 else "")
        print(f"{d['tag']:34s} {d['num_tokens']:4d} {d['steps']:6d} {base:11.4f} {rnd:10.4f} {best:10.4f} {drop:7.1f}%{note}")
    print()


def rung1():
    d = load(TOKENS / "reports" / "rung1.json")
    if not d:
        print("RUNG 1: not available yet\n")
        return
    print("=" * 100)
    print(f"RUNG 1 -- feature gap on {d['frames']} held-out frames (episodes {d['holdout_episodes']})")
    print("  ratio = cosine distance to the twin / distance between two REAL frames of the same moment")
    print("  content = the 192 patches carrying image content (the other 64 are resize padding)")
    print("=" * 100)
    for cam, e in d["cameras"].items():
        base_d = e["no_tokens"]["d_sr_content"]
        print(f"\n-- {cam}   (d_rr, the real-vs-real floor: {e['d_rr_content']:.4f})")
        print(f"   {'condition':40s} {'d_sr':>8s} {'reduced':>9s} {'ratio':>8s} {'probe':>7s}")

        def line(name, r, probe=None):
            # Reduction in raw distance is the honest headline. `ratio` divides by a noise floor
            # that is much closer to the wrist's baseline than to the external camera's, so a
            # "% of ratio excess removed" figure inflates wildly on the wrist and is not
            # comparable between the two cameras.
            red = 100.0 * (1.0 - r["d_sr_content"] / base_d)
            pr = f"{probe:7.3f}" if probe is not None else " " * 7
            print(f"   {name:40s} {r['d_sr_content']:8.4f} {red:8.1f}% {r['ratio_content']:8.3f} {pr}")

        line("no tokens (baseline)", e["no_tokens"], e["no_tokens"]["probe"]["best_test_acc"])
        if "oracle_constant_offset" in e:
            line("oracle: best constant offset", e["oracle_constant_offset"])
            line("oracle: best per-patch offset", e["oracle_per_patch_offset"])
        for tag, r in sorted(e["runs"].items(), key=lambda kv: -kv[1]["d_sr_content"]):
            kind = " [shuffled]" if r["shuffled_control"] else (" [random]" if r["steps"] == 0 else "")
            line(tag + kind, r, r["probe"]["best_test_acc"])
    print()


def rung2():
    for k in (64, 16):
        d = load(TOKENS / "reports" / f"rung2_K{k}.json")
        if not d:
            continue
        print("=" * 100)
        print(f"RUNG 2 -- action chunks vs the twin's own prediction, K={k}, {d['frames']} held-out frames")
        print("=" * 100)
        print(f"   {'condition':30s} {'mean |a - a_sim|':>18s} {'p95':>10s} {'gap removed':>13s}")
        for name, e in d["conditions"].items():
            if name == "sim":
                continue
            rem = e.get("pct_of_gap_removed", e.get("pct_of_perceptual_gap_removed"))
            rs = f"{rem:12.1f}%" if rem is not None else " " * 13
            print(f"   {name:30s} {e['mean_abs_diff_vs_sim']:18.5f} {e['p95_abs_diff_vs_sim']:10.5f} {rs}")
        print()


if __name__ == "__main__":
    stage2()
    rung1()
    rung2()
