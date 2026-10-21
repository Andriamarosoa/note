# full_note ONLY — isolated S834 native PCM256

This is ONE characteristic in its own reduced Git tree, with frozen S708 acoustic inputs retained solely as prerequisites. There are **no decoders, public output, head selectors or training scripts for the other five musical characteristics**.

S834 original 80-track true PCM256 FIT80 baseline: Full Note stream onset F1 71.108037% (TP 8224 FP 3640 FN 3043). S834's valid full-note F1 is 35.389737%, with 4093 correct offsets among 8135 eligible; do not confuse these metrics. The original weights were pretrained with hop1024 then run freshly every PCM256. Reference replay on this reduced branch must reproduce the same raw counts.

No hardware iRig latency result; no new hold, lookahead, MIDI integration or production promotion.
