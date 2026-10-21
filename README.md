# full_note ONLY — isolated S834 native PCM256

This is ONE characteristic in its own reduced Git tree, with frozen S708 acoustic inputs retained solely as prerequisites. There are **no decoders, public output, head selectors or training scripts for the other five musical characteristics**.

S834 original 80-track true PCM256 FIT80 baseline: Full Note stream onset F1 71.108037% (TP 8224 FP 3640 FN 3043). S834's valid full-note F1 is 35.389737%, with 4093 correct offsets among 8135 eligible; do not confuse these metrics. The original weights were pretrained with hop1024 then run freshly every PCM256. Reference replay on this reduced branch must reproduce the same raw counts.

No hardware iRig latency result; no new hold, lookahead, MIDI integration or production promotion.

## Signed FIT80 isolation replay audit (2026-10-21)

- Workflow: [39672720067](https://github.com/Andriamarosoa/note/actions/runs/39672720067). Smoke tests PASS; signed 80-track replay completed all tracks, scientific parity assertion FAILED.
- **Isolated branch measured:** TP 8375, FP 4160, FN 2892; 4086 correct offsets / 8283 eligible; true Full Note F1 34.33325 %.
- **Original source reference:** TP 8224, FP 3640, FN 3043; 4093 correct offsets / 8135 eligible; true Full Note F1 35.38974 %.
- **Paired difference:** -1.05649 pp.
- **Verdict:** REGRESSION: rejected. Reference S834 71.10804% concerns matching ONSETS only, not full-note F1.

These are genuine observed measurements on the matched signed 80-track FIT80 cohort, not a new HELD evaluation. Maintain source references; do not overwrite or promote any other axis or public MIDI. A scientifically correct next audit must localize the mismatch in copied acoustic frontend, frozen network outputs and per-axis HEAD->SELECTOR->DECODER behavior on identical PCM frames before relaxing the source-parity requirement.
