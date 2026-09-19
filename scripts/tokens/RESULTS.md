# Sim-twin token alignment — offline results (2026-09-19)

Base policy: `pi05_piperx_sim_expert` (vision frozen, Gemma frozen, action expert trained),
checkpoint `pi0.5-ft01-simdata/9999`. Sim-in-sim success for this recipe is 96.7–100% over 30
episodes, so the precondition "competent in sim" holds.

Encoder: SigLIP So400m/14, 27 layers, width 1152, 256 patches, frozen. Verified bit-identical
(`bfloat16(pi05_base)`) across every checkpoint, so the targets do not depend on which one is used.

Data: the 49 paired real/sim episodes, 18,431 frames, 2 cameras. Holdout is 10 whole episodes
(every 5th), 3,761 frames. Pair alignment verified at max|diff| 3e-8 on recovered absolute actions.

## Rung 1 — feature gap (2,000 held-out frames)

`d_sr` is cosine distance to the twin over the 192 content patches; `reduced` is against the
untreated baseline.

| condition | external d_sr | reduced | wrist d_sr | reduced |
|---|---|---|---|---|
| no tokens | 0.0265 | — | 0.0177 | — |
| random tokens (control) | 0.0271 | −2.3% | 0.0179 | −1.1% |
| shuffled pairs (control, K=16) | 0.0132 | 50.1% | 0.0159 | 10.6% |
| oracle: best constant offset | 0.0214 | 19.2% | 0.0160 | 9.9% |
| oracle: best per-patch offset | 0.0095 | 64.3% | 0.0135 | 24.2% |
| K=4 | 0.0167 | 37.2% | 0.0116 | 34.8% |
| K=16 | 0.0096 | 63.9% | 0.0089 | 49.9% |
| K=64 | 0.0059 | 77.7% | 0.0076 | 57.3% |
| K=64, 12k steps | 0.0045 | **83.2%** | 0.0068 | **61.9%** |

Linear probe (sim vs real, split by episode, L2 swept, best test accuracy reported) stays at 1.000
in every condition except the wrist K=64 long run, which falls to 0.942.

## Rung 2 — action chunks (600 held-out frames, K=64)

Reference is the policy's own prediction on the twin frame.

| condition | mean abs diff | gap removed |
|---|---|---|
| real, untreated | 0.02422 | — |
| real + random tokens (control) | 0.02440 | −0.8% |
| images only, untreated | 0.02171 | — |
| state only, untreated | 0.01380 | — |
| real + K=64 tokens | 0.01629 | 32.7% |
| images only + K=64 tokens | 0.00890 | 59.0% |

K=16: 26.4% / 49.5%.

## What these say

1. **The gap closes and the controls are clean.** Random tokens move nothing (−2.3% to +0.2%); the
   learned tokens remove 83% of the external camera's feature distance. Rung 1 and Rung 2 both pass.

2. **The tokens beat the oracle additive correction.** A constant offset fitted on the evaluation
   frames themselves reaches 19% (external) and a per-patch offset 64%; the tokens reach 83%. A
   shared token set is therefore NOT equivalent to a constant feature shift — attention makes its
   effect content-dependent. The plan's "image-independent modulation" framing understates the
   mechanism.

3. **The pairing matters, and it matters unevenly.** On the static external camera the shuffled
   control reaches 50 of 78 points, so much of that gap really is a constant appearance shift. On
   the moving wrist camera it reaches only 11 of 57 — there the frame-accurate twin does nearly all
   the work. This is the strongest evidence that building the paired dataset was necessary.

4. **Capacity is not saturated.** K=4 → 16 → 64 improves monotonically, and 12k steps beats 3.5k on
   both cameras with the best holdout loss at the final step every time. The minimal-footprint
   question (RQ2) has no plateau yet: 64 is a floor, not an answer.

5. **The domains stay linearly separable.** The probe holds at 1.000 almost everywhere while
   distance falls 83%, so a small, perfectly consistent direction survives. That residual is the one
   most likely to be *content* rather than appearance — the camera-housing props hidden in the sim
   render, the cup's differing artwork — which no shared correction can supply.

6. **After treatment the state channel is the larger remaining contributor.** Untreated, images
   carry 90% of the action divergence and state 57% (they interact — the halves exceed the whole).
   After tokens the residual is 0.0163 against a state-only gap of 0.0138. Worth knowing before
   Rung 3.

## Deploying

    scripts/tokens/finalize.py          # writes ~/training/tokens/deploy/tokens_K*.npz

    uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
      --policy.config=pi05_piperx_sim_expert \
      --policy.dir=/home/ubuntu/training/runs/pi05_piperx_sim_expert/pi0.5-ft01-simdata/9999 \
      --policy.prompt-tokens=/home/ubuntu/training/tokens/deploy/tokens_K64-long.npz

`tokens_K64-long.npz` is 577 KB and 147,456 parameters — the only new parameters anywhere in the
system. The checkpoint, the train config and the arm-side client are unchanged; the token file is
the single difference between serving the base policy and the adapted one. `tokens_K64-random.npz`
and `tokens_K16-shuffled.npz` are the two controls, ready to serve the same way.

## Reproducing

    scripts/tokens/dump_targets.py      # Stage 1, ~15 min
    STEPS=3500 scripts/tokens/sweep.sh  # Stage 2, ~70 min
    scripts/tokens/rung1.py             # ~25 min
    scripts/tokens/rung2.py --k 64      # ~5 min
    scripts/tokens/report.py
