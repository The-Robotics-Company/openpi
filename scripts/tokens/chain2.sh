#!/usr/bin/env bash
cd /home/ubuntu/github-repos/openpi
LOGS=/home/ubuntu/training/tokens/logs

for K in 64 16; do
  echo "### $(date -u +%H:%M:%S)  RUNG 2  K=$K"
  .venv/bin/python scripts/tokens/rung2.py --k $K --max-frames 600 > "$LOGS/rung2_K$K.log" 2>&1
  echo "   rc=$?"
done

# Both K=64 runs hit their best holdout loss at the FINAL step, so 3500 steps is where training
# stopped, not where it converged. These separate "the ceiling of a shared token set" from "we ran
# out of budget" -- the question the escalate-to-VPT-deep decision turns on.
for cam in base_0_rgb left_wrist_0_rgb; do
  tag="$cam-K64-long"
  [ -f "/home/ubuntu/training/tokens/runs/$tag/tokens.npz" ] && { echo "== skip $tag"; continue; }
  echo "### $(date -u +%H:%M:%S) $tag"
  .venv/bin/python scripts/tokens/train_tokens.py --camera "$cam" --num-tokens 64 \
      --tag "$tag" --steps 12000 --eval-every 500 > "$LOGS/$tag.log" 2>&1
  echo "   rc=$? $(grep saved "$LOGS/$tag.log" | tail -1)"
done

echo "### $(date -u +%H:%M:%S) RUNG 1 (final, including the long runs)"
.venv/bin/python scripts/tokens/rung1.py > "$LOGS/rung1_final.log" 2>&1
echo "   rc=$?"
echo "### $(date -u +%H:%M:%S) CHAIN2 DONE"
