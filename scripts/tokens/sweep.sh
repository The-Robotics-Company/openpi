#!/usr/bin/env bash
# Stage 2 sweep: one token set per (camera, K), plus the two controls.
#
# Ordered so the headline result lands first. K=64 is the most capacity, so if the gap does not
# close there it will not close at 16 or 4 -- and if the night is cut short, the most informative
# rows are already on disk. Every run is skipped if its tokens.npz already exists, so the sweep is
# resumable.
set -u
cd /home/ubuntu/github-repos/openpi
STEPS="${STEPS:-3500}"
LOGS=/home/ubuntu/training/tokens/logs
mkdir -p "$LOGS"

run () {  # run <tag> <camera> <k> <extra args...>
  local tag=$1 cam=$2 k=$3; shift 3
  if [ -f "/home/ubuntu/training/tokens/runs/$tag/tokens.npz" ]; then
    echo "== skip $tag (done)"; return
  fi
  echo "== $(date -u +%H:%M:%S) $tag"
  .venv/bin/python scripts/tokens/train_tokens.py \
      --camera "$cam" --num-tokens "$k" --tag "$tag" "$@" > "$LOGS/$tag.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "   FAILED rc=$rc -- last lines:"; tail -12 "$LOGS/$tag.log" | sed 's/^/     /'
  else
    grep -E "NO TOKENS|RANDOM |saved" "$LOGS/$tag.log" | sed 's/^/   /'
  fi
}

for cam in base_0_rgb left_wrist_0_rgb; do run "$cam-K64" "$cam" 64 --steps "$STEPS"; done
for cam in base_0_rgb left_wrist_0_rgb; do run "$cam-K64-random" "$cam" 64 --steps 0; done
for cam in base_0_rgb left_wrist_0_rgb; do run "$cam-K16" "$cam" 16 --steps "$STEPS"; done
for cam in base_0_rgb left_wrist_0_rgb; do run "$cam-K16-shuffled" "$cam" 16 --steps "$STEPS" --shuffle; done
for cam in base_0_rgb left_wrist_0_rgb; do run "$cam-K4"  "$cam" 4  --steps "$STEPS"; done
for cam in base_0_rgb left_wrist_0_rgb; do run "$cam-K16-random" "$cam" 16 --steps 0; done

echo "== $(date -u +%H:%M:%S) SWEEP COMPLETE"
ls -la /home/ubuntu/training/tokens/runs/
