# EXACT-K ONLY — branch contract

**Branch:** `codex/note-best-exact-k-s834-256`

**Single mutable characteristic:** Exact-K (cardinality K0–K6).

**Reference:** S834 shadow FIT80 original fresh PCM256 source, 232295 correct out of 306536, 75.780659%; S708 frozen pretrained source from 1024-trained model. The score is a reference to reproduce before subsequent model improvement, not a claim about a newly evaluated combined MIDI.

**Output:** `ExactKDecision(sample,count,confidence)` only.

**Removed from this branch's Git tree:** the other five axis heads/decoders/decisions, scripts, unit tests, workflows and old series folders. They remain in their own Git branches; do not change them or their scores.

**Constraints:** 44100 Hz, exact hop=256 PCM (5.805 ms), strictly causal current/past input, zero extra buffering/hold, candidate selectors and data-driven branch decisions must not use held-out GT, no public MIDI rewrite, strict non-regression against 232295/306536 on FIT80 paired evaluation, physical p95/p99 still unknown.

**Allowed future work:** optimize and train Exact-K subheads, audit harmonic over/under-count K0..K6, test fold-disjoint CAL/OUTER, profile CPU under physical iRig setup, without modifying any other characteristic.
