# frame_pitch ONLY — isolated S834 native PCM256

This is ONE characteristic in its own reduced Git tree, with frozen S708 acoustic inputs retained solely as prerequisites. There are **no decoders, public output, head selectors or training scripts for the other five musical characteristics**.

S834 original 80-track true PCM256 FIT80 baseline: Frame Pitch F1 84.998850% (TP 467540 FP 71501 FN 93528) The original weights were pretrained with hop1024 then run freshly every PCM256. Reference replay on this reduced branch must reproduce the same raw counts.

No hardware iRig latency result; no new hold, lookahead, MIDI integration or production promotion.
