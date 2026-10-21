# frame_pitch ONLY — isolated S834 native PCM256

This is ONE characteristic in its own reduced Git tree, with frozen S708 acoustic inputs retained solely as prerequisites. There are **no decoders, public output, head selectors or training scripts for the other five musical characteristics**.

S834 original 80-track true PCM256 FIT80 baseline: Frame Pitch F1 84.998850% (TP 467540 FP 71501 FN 93528) The original weights were pretrained with hop1024 then run freshly every PCM256. Reference replay on this reduced branch must reproduce the same raw counts.

No hardware iRig latency result; no new hold, lookahead, MIDI integration or production promotion.

## Signed FIT80 isolation replay audit (2026-10-21)

- Workflow: [39672713127](https://github.com/Andriamarosoa/note/actions/runs/39672713127). Smoke tests PASS; signed 80-track replay completed all tracks, scientific parity assertion FAILED.
- **Isolated branch measured:** TP 467780, FP 76122, FN 93288; F1 84.66836 %.
- **Original source reference:** TP 467540, FP 71501, FN 93528; F1 84.99885 %.
- **Paired difference:** -0.33049 pp.
- **Verdict:** REGRESSION: rejected. Native256 outputs changed after isolation, investigate shared upstream acoustic / original selector equivalence.

These are genuine observed measurements on the matched signed 80-track FIT80 cohort, not a new HELD evaluation. Maintain source references; do not overwrite or promote any other axis or public MIDI. A scientifically correct next audit must localize the mismatch in copied acoustic frontend, frozen network outputs and per-axis HEAD->SELECTOR->DECODER behavior on identical PCM frames before relaxing the source-parity requirement.
