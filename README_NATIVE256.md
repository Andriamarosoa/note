# ONSET — independent native PCM256 research candidate

This branch is derived from codex/note-best-onset-s708-protected without changing that protected S708/H0 reference.
The production MIDI is NOT modified. Only this axis has a live decoder in src/onset256.
The historical scorer and reference replay are retained for provenance, not for native inference.

- Audio: 44,100 Hz, mono PCM; strict input sample clock, no future audio.
- Native acoustic and S708 temporal neural inference: **one fresh call every 256 samples** (5.80498866 ms decision cadence).
- Frozen S708 network was **trained on 1024-hop data** and is *not* a newly trained PCM256 network.
- The neural/tree hybrid is restored with ONNX Runtime; no other musical branch's decoder or selection is called.
- Onset: emits only new attacks, using private active state solely for rearming.
- Offset: emits only releases, with diagnostic attack events used solely to define matched-onset eligibility in the evaluator.
- Original FIT80 GuitarSet sources and frozen S708 checkpoint are SHA-pinned; GT labels are read only after completing track inference. No ground-truth-conditioned inference or same-fold threshold selection.
- No artificial 1024-frame repetition, hold, future context, or lookahead.
- Historical S708/H0 reference: **73.2520185535% F1** on 80 explored FIT80 tracks; the new native256 score remains unverified until CI completes successfully. No external generalization claim.
- Decision cadence is not measured hardware latency; inference runtime must still be benchmarked on the actual machine.
- No promotion if regressions, leaks, skipped scoring, or CI failures occur.

Scientific workflow: `.github/workflows/onset-native256.yml` and `scripts/replay_native256.py`.
Tests: `tests/test_native256.py`. All older S708/H0 files remain historical only.
