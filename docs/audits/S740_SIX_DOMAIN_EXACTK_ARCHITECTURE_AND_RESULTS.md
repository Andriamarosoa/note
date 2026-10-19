# S740 — Generalization of Exact-K dynamic selection to six NOTE domains

Research branch: `codex/note-s740-six-domain-exactk-fragment-routers`.
Parent: S739 original Exact-K class-conditional architecture.
**No automatic promotion. No microphone LIVE change. Official reference remains S708.**

## What is implemented and actually validated

- **Six separate neural parameter sets** using the original 3-head attention, K0–K6 class gate and separate KEEP/risk path. The networks do **not** share a global combination or learned weights.
- Each one chooses from the **32 H0-anchored combinations independently for each causal four-hop fragment (~23.22 ms)**. One fragment-level mask covers 49 pitches, but each pitch retains its own probability.
- Onset uses S739 existing six heads. Sustain and Release have six causal candidate heads each, plus an independent FIT-only training and full MIDI evaluation script.
- Offset has an explicit **S707 note-instance hazard adapter** and replays the true `render_release` NoteOff emitter. It cannot be replaced by a binary Release-channel edit.
- Note identity has five action distributions KEEP/ADD/REMOVE/CHANGE_PITCH/REATTACK, a causal stateful event decoder, and a mandatory independently proposed destination for CHANGE_PITCH.
- Exact-K has seven-class distributions and five additional causal, source-derived proposals: native K, previous K, Onset acoustic density, Sustain acoustic density, and already emitted active-note count. Implemented in `s741_exactk_sources.py` and tested; these sources are NOT five independently pretrained neural networks, are NOT calibrated/selected yet and cannot be promoted. The selector normalizes seven-class probabilities after the fragment mask.
- Unit tests verify separate parameters, class-conditional routing, 32 combinations per fragment, note instances, causal prefix, six-note MIDI active limit, target isolation, safety and no synthetic score promotion.

## Real-world training status

- S739 trained Onset: all 18 CAL selections rejected as non-improving or regressive; S708 remains official.
- S740 `s740_train_posterior_specialists.py`: reproducible FIT80 / CAL32 training for Sustain and Release **separately**, with 2 seeds x 3 checkpoints x 3 confidence gates, 18 rounds *per target*, real complete MIDI decode. Each stage measures:
  1. Onset F1;
  2. Exact-K;
  3. active Exact-K;
  4. Offset accuracy;
  5. complete-note F1;
  6. Frame Pitch F1.
- Only candidates passing all six CAL no-regression gates, preserving all previously correct offsets/long onsets, are considered for HELD/DEV replay. HELD/DEV were previously exposed and cannot be called blind TEST.
- Offset, identity, and a newly trained Exact-K selector **still require separately supervised FIT/CAL pipelines**. S741 only adds causal Exact-K candidate evidence and tests, not new K performance. Having an adapter, action decoder, or network instance does not constitute an trained model or score improvement.
- GuitarSet CAL reference: Onset F1 74.2639%; Exact-K 62.0455%; active Exact-K 56.2762%; Offset 77.3292%; complete notes F1 57.1319%; Frame Pitch F1 76.4626%.
- **No result >80% is claimed without all six real measurements and no regression.** If no eligible improvement exists, S708 stays official.

## Workflow and source

- `.github/workflows/note-s740-six-domain-contract.yml`: code/causality/target independence.
- `.github/workflows/note-s740-train-sustain-release.yml`: real FIT/CAL learning, separately for Sustain and Release; signed S708/S734/S707 inputs and retained score reports.
- `src/causal_note/s740_multitarget_dynamic.py`, `s740_event_action_bridge.py` and `s741_exactk_sources.py`: research adapters.
- `test/test_s740_multitarget_dynamic.py` and `test/test_s740_event_action_bridge.py`: unit tests.
- A successful unit CI workflow **does not** mean >80% score or LIVE promotion.

## Structural bottleneck that remains

S739's onset learner elected H0 on >99% of CAL fragments, and other masks introduced regressions. Improving its feature coverage and class/rare-positive calibration is a research need. Moreover, increasing Onset alone cannot close the >20-point gap in complete-note F1 without independent K, note-identity and NoteOff improvements.

The next strict scientific milestone is training a *real* K0–K6 selector, Offset-instance selector, and identity-event selector on FIT, not treating their untrained architectures as successful experiments.
