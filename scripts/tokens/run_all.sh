#!/usr/bin/env bash
# The whole overnight chain. Each stage is resumable and logs separately, and a failure in one
# does not take down the ones after it -- a partial result is worth more in the morning than a
# clean abort.
cd /home/ubuntu/github-repos/openpi
LOGS=/home/ubuntu/training/tokens/logs
mkdir -p "$LOGS"

echo "### $(date -u +%H:%M:%S)  STAGE 2 SWEEP"
STEPS=3500 bash scripts/tokens/sweep.sh 2>&1 | tee "$LOGS/sweep.log"

echo "### $(date -u +%H:%M:%S)  RUNG 1"
.venv/bin/python scripts/tokens/rung1.py > "$LOGS/rung1.log" 2>&1
echo "  rc=$?"; grep -E "NO TOKENS|ratio_content|wrote" "$LOGS/rung1.log" | tail -30

for K in 64 16; do
  echo "### $(date -u +%H:%M:%S)  RUNG 2  K=$K"
  .venv/bin/python scripts/tokens/rung2.py --k $K > "$LOGS/rung2_K$K.log" 2>&1
  echo "  rc=$?"; tail -30 "$LOGS/rung2_K$K.log"
done

echo "### $(date -u +%H:%M:%S)  ALL DONE"
