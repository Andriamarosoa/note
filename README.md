# Active Exact-K ONLY, S828 true native PCM256

This Git branch contains only Active Exact-K source/readout, including its own exact S827 guided K state. It has NO pipelines/decoders for Exact-K, Onset, Offset, Frame Pitch, Full Note. Original frozen S708 is used as an acoustic backbone (originally pretrained 1024), but the S828 private HGB readout is trained and CAL-selected on each FIT80 fold at true native PCM256, not at 1024.

Signed S828 benchmark to reproduce exactly: 184143 / 249411 active frames = 73.831146% (80 GuitarSet FIT80 tracks), TRAIN 48 / CAL 16 / OUTER 16 tracks times five composition splits. This score is SHADOW ActiveK only; original MIDI remains unchanged. If the exact expected count fails in the trimmed replayer, declare revalidation failed, not promoted.

## Signed FIT80 isolation replay audit (2026-10-21)

- Workflow: [39673104335](https://github.com/Andriamarosoa/note/actions/runs/39673104335). Smoke tests PASS; signed 80-track replay completed all tracks, scientific parity assertion FAILED.
- **Isolated branch measured:** 184044/249411 = 73.79145 %.
- **Original source reference:** 184143/249411 = 73.83115 %.
- **Paired difference:** -99 correct (-0.03969 pp).
- **Verdict:** REGRESSION: rejected. Five composition-disjoint folds completed but original S828 score not reproduced.

These are genuine observed measurements on the matched signed 80-track FIT80 cohort, not a new HELD evaluation. Maintain source references; do not overwrite or promote any other axis or public MIDI. A scientifically correct next audit must localize the mismatch in copied acoustic frontend, frozen network outputs and per-axis HEAD->SELECTOR->DECODER behavior on identical PCM frames before relaxing the source-parity requirement.
