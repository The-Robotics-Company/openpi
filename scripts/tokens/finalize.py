"""Emit deployable token files: one npz per configuration, both cameras, self-describing.

These are the artifacts that go to the robot. `serve_policy --policy.prompt-tokens <file>` is the
only change needed to serve the adapted policy; the checkpoint, the train config and the arm-side
client are all untouched.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pairs

TOKENS = pathlib.Path("/home/ubuntu/training/tokens")
OUT = TOKENS / "deploy"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    suffixes = sorted({d.name.split("-K", 1)[1] for d in (TOKENS / "runs").iterdir() if "-K" in d.name})
    manifest = []
    for suf in suffixes:
        sets, metas = [], []
        for cam in pairs.CAMERAS:
            f = TOKENS / "runs" / f"{cam}-K{suf}"
            if not (f / "tokens.npz").exists():
                sets = None
                break
            sets.append(np.load(f / "tokens.npz")["tokens"][0])
            metas.append(json.loads((f / "summary.json").read_text()))
        if not sets:
            print(f"skip K{suf}: incomplete camera set")
            continue
        tokens = np.stack(sets).astype(np.float32)
        path = OUT / f"tokens_K{suf}.npz"
        np.savez(path, tokens=tokens, cameras=np.array(pairs.CAMERAS))
        entry = {
            "file": str(path),
            "shape": list(tokens.shape),
            "cameras": list(pairs.CAMERAS),
            "params": int(tokens.size),
            "kb": round(path.stat().st_size / 1024, 1),
            "per_camera": {
                m["camera"]: {
                    "steps": m["steps"],
                    "shuffled_control": m["shuffled_control"],
                    "holdout_no_tokens": round(m["no_tokens"]["total"], 5),
                    "holdout_best": round(m["best_holdout"]["total"], 5),
                    "best_step": m["best_step"],
                }
                for m in metas
            },
        }
        manifest.append(entry)
        print(f"{path.name:28s} {str(tokens.shape):18s} {entry['params']:>8,d} params  {entry['kb']:>7.1f} KB")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {OUT}/manifest.json")


if __name__ == "__main__":
    main()
