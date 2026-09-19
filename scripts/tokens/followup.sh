#!/usr/bin/env bash
# Runs after the main chain. Both K=64 runs had their best holdout loss at the FINAL step, so 3500
# steps is where training stopped, not where it converged. These longer runs separate "this is the
# ceiling of a shared token set" from "we ran out of budget" -- which is exactly the question the
# escalate-to-VPT-deep decision turns on.
cd /home/ubuntu/github-repos/openpi
LOGS=/home/ubuntu/training/tokens/logs

until grep -q "ALL DONE" "$LOGS/run_all.log" 2>/dev/null; do sleep 60; done
echo "### $(date -u +%H:%M:%S) main chain done, starting long runs"

for cam in base_0_rgb left_wrist_0_rgb; do
  tag="$cam-K64-long"
  if [ -f "/home/ubuntu/training/tokens/runs/$tag/tokens.npz" ]; then echo "== skip $tag"; continue; fi
  echo "== $(date -u +%H:%M:%S) $tag"
  .venv/bin/python scripts/tokens/train_tokens.py --camera "$cam" --num-tokens 64 \
      --tag "$tag" --steps 12000 --eval-every 500 > "$LOGS/$tag.log" 2>&1
  echo "   rc=$? $(grep saved "$LOGS/$tag.log" | tail -1)"
done

echo "### $(date -u +%H:%M:%S) RUNG 1 (full, including long runs)"
.venv/bin/python scripts/tokens/rung1.py > "$LOGS/rung1_final.log" 2>&1
echo "   rc=$?"

echo "### $(date -u +%H:%M:%S) FOLLOWUP DONE"
