# S618 — FN-first rescue of genuinely missed guitar note attacks

Date: 2026-10-16. Research branch `codex/note-live-s618-missed-onset-rescue`,
based on `codex/note-live-s615-full-loops`.
**Official S613, S80 and S617 references/weights remain UNCHANGED.**

## Why change the optimization objective?

The correctly evaluated S613 NATIVE HYBRID (neural heads + frozen ONNX
trees) on the pinned 17 GuitarSet/URMP dev songs has:

- **TP = 1241**, **FP = 734**, **FN = 838**, NoteOn F1 ±80ms **61.2235%**.
- A S615 hybrid retraining/mixture on the same dev songs has TP1269
  (+28), FP958 (+224), FN810 (−28), F1 58.9410%.
- Thus native heads DO have recovery opportunities despite FP regressions.
  **Do not first discard newly recovered TRUE NOTE attacks** by globally
  limiting count, overblocking octave/harmonic siblings or suppressing
  K3/K4 notes.

New research objective, in order:
1. Locate individually matched S613 false negative ground-truth
   attacks on the same 17 dev recordings.
2. Cluster causes: onset head low, onset and sustain not simultaneous
   at selector threshold, or selector state/cooldown/timing.
3. Create explicitly causal recovery selectors WITHOUT future samples,
   no pitch/oracle truth in model input, and no long hold after onset.
4. Measure **recovered previously missed truth** AND
   **lost previously correct truth**, not only net TP.
5. Report full TP/FP/FN, precision/recall, F1 and grouped-onset Exact-K
   by K1,K2,K3,K4 and true pitch set. A rise in recall is not sufficient
   proof of improved F1 or realistic piano quality.
6. Only after class-specific FN evidence consider a trainable
   specialist rescue head. It must run on already computed causal
   features, preserve all existing NOTE native heads, and be judged
   against unseen independent compositions.

## First reproducible causal ablation

`scripts/s618_missed_attack_sweep.py` loads frozen S613 using
`load_heads` **not** `NativeLiveHeads` (a known previous evaluation
mistake which silently disabled the ONNX component).
It caches head probabilities from all 17 pinned dev songs then
replays 10 strictly causal selection configurations:
- baseline onset/sustain: .60/.50;
- onset-only relaxed: .55/.50, .50/.50, .45/.50;
- sustain-only relaxed: .60/.45, .60/.40;
- both relaxed: .55/.45, .50/.45, .50/.40, .45/.40.

Other selection state and weights NEVER change: harmonic expert,
native six-note max, K-state, refractory, release, ONNX and model hop
256. This is a diagnostic ablation, not the final trained rescue design.

Example in full S615 development cache:

```powershell
python -m scripts.s618_missed_attack_sweep --root data/S615_full --out results/S618_frozen_FN.json
```

Pinned real GuitarSet+URMP run:
https://github.com/Andriamarosoa/note/actions/runs/38868585898

**Important**: controls are selected and measured on the same dev songs;
none is independently generalization-validated and none may replace
the official model. The purpose is classifying rescue opportunity rather
than jumping from baseline to a lower global threshold.

## Physical interpretation

For a target false-negative attack E2, the future-aware OFFLINE AUDIT
is permitted to ask, *after prediction*, whether at some reference ±80ms
frame the model had onset>.6 and sustain>.5 simultaneously. That label
does NOT enter live inference. If not, it separates head confidence
failures from post-head selector inhibition. This distinction is
necessary before proposing a new specialist gate.

Causal LIVE gating must use only the present/past acoustic spectrum,
heads' present probabilities, prior note states and previously observed
onsets. The ground truth timestamp and pitch are NEVER inputs to
live inference. The historical 8192 FFT is a past-only receptive field;
there is no added post-onset lookahead.

## Promotion criterion

- First quantify ***recall gain*** (recover missed notes).
- Explicitly account for FP and regressions of existing TPs.
- Guard against fake octave and true E2+E3 simultaneously,
  avoiding blind suppression of harmonics at an independent attack.
- Measure K1–K6 and short NoteOff/tail as separate channels;
  grouped onset K is NOT the historic S80 Exact-K protocol.
- Confirm on new held-out music and actual iRig->FluidSynth Windows LIVE
  timing before touching the reference.
