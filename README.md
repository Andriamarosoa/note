# Exact-K S834 — isolated PCM256 branch

This branch contains **one and only one musical characteristic: Exact-K**, outputting K0 through K6 every 256 fresh audio samples at 44,100 Hz.

- Active Exact-K, Onset, Offset, Frame Pitch, Full Note decoders and their heads, selectors, tests, reports and workflows are **removed from this branch's tree**, not from the other five branches or the repository's main line.
- Frozen 1024-trained S708 acoustic backbone remains as an internal **source** of harmonically rich count evidence. Its auxiliary representation is not an independently exposed characteristic.
- The predictor returns only `ExactKDecision(sample,count,confidence)`. It emits **no NoteOn/NoteOff**, no MIDI, and never consults other heads' decisions or ground-truth labels.
- Native source .npz is retained in `model/live/strict_note/`; S708 .pt checkpoint is SHA-verified and downloaded from GitHub Action #39094682419 when running research. Neither is newly trained.
- Model's matched 80-track GuitarSet FIT80 S834 reference is **232,295 / 306,536 = 75.780659 %** (SHADOW research score, not certified as achieved yet by this trimmed branch until the replay passes).
- Native hop duration 256/44,100 = **5.805 ms**. This is decision spacing, NOT proven processing latency; Linux S834 p95 exceeded the hop budget.
- No lookahead or post-attack hold is added. Existing historical source weights were trained on an effective hop of 1024.
- No production promotion, no MIDI integration, no update to any other axis.

## Run the tests

```bash
python -m pip install numpy scipy torch
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Signed full replay

See `.github/workflows/exact-k-only-256.yml`. The workflow downloads the original S708 checkpoint and FIT80 manifests, validates source WAV SHA-256 and replays 80 tracks. The result must reproduce **exactly** 232295/306536 decisions. A failure means the extraction is not yet proven equivalent to S834.

## Scope

The only runnable model is `src/exact_k/runtime.py`. `src/exact_k/acoustics.py` and `temporal.py` contain **supporting fixed source-feature extraction** needed to preserve the S834 score; there is no second axis detector. Any future training or experiments in this branch must target Exact-K only.

All removed research history remains available on the other Git branches, especially `codex/note-s834-s809-six-axes-native256` and the five distinct `codex/note-best-*` axis branches.

## Verification

The exact-k-only source isolation smoke test passed in GitHub Actions run #39669618268. The signed FIT80 80-track K-only replay is separately triggered and must match the original S834 result before a score is revalidated for this trimmed branch.

## Signed FIT80 isolation replay audit (2026-10-21)

- Workflow: [39673135663](https://github.com/Andriamarosoa/note/actions/runs/39673135663). Smoke tests PASS; signed 80-track replay completed all tracks, scientific parity assertion FAILED.
- **Isolated branch measured:** 232332/306536 = 75.79273 %.
- **Original source reference:** 232295/306536 = 75.78066 %.
- **Paired difference:** +37 correct (+0.01207 pp).
- **Verdict:** NOT BIT-EXACT: source reference differs. Improves accuracy slightly, but scientific identity not demonstrated. GitHub workflow failed strict parity assertion.

These are genuine observed measurements on the matched signed 80-track FIT80 cohort, not a new HELD evaluation. Maintain source references; do not overwrite or promote any other axis or public MIDI. A scientifically correct next audit must localize the mismatch in copied acoustic frontend, frozen network outputs and per-axis HEAD->SELECTOR->DECODER behavior on identical PCM frames before relaxing the source-parity requirement.
