# S433–S435 — Authentic WAV guitar → names of notes, original Exact-K preserved

**14 October 2026. Status: measured development / NO promotion.**

The desired task is to recognize **musical note names** such as `E3` or `[E3,G3]` from sound, **not** to create or export a MIDI file. Live and note-end Offset work are deferred. This experiment advances the source interface to real recordings while retaining the original pretrained S80 K0–K6.

## 1. S433 — No manually typed list of K decisions

S433 joins the original S80 count archive (`row_id`, `s80_K`) to the original V88 native onset archive (`row_id`, `member`, `start_samples`, `end_samples`) by **immutable source row identity**. It sorts by measured onset within a chosen recording, emits a strict row-aligned event list, then invokes the actual S432 WAV spectral extractor and native S430 pretrained musical-note heads.

The adapter **does not** compute K from peaks, assume the NPZ arrays share an order, or silently invent K on unmatched events. Missing source IDs, duplicate events, annotation fields and incorrect cardinalities are rejected. K0 produces zero new note names, not silence.

Source: [S433](https://github.com/Andriamarosoa/note/blob/codex/s276-idmt-song-disjoint-candidate-audit/scripts/s433_join_original_S80_native_events_to_WAV_note_names.py).

[GitHub Actions #38521744655](https://github.com/Andriamarosoa/note/actions/runs/38521744655) **success**, outputs **E3**, **E3/G3**, **[]** on a controlled physical WAV with intentionally shuffled source rows and K1/K2/K0. This proves the **integration**, not the independent accuracy of S80 or the trained notes head.

CLI:
```bash
PYTHONPATH=.:src python -m scripts.s433_join_original_S80_native_events_to_WAV_note_names \
  --wav guitare.wav --member "nom_exact_du_morceau.jams" \
  --s80-npz model/S80_original_predicted_K.npz \
  --native-events model/source_V88_native_groups.npz \
  --s402-models model/S402 --s416-models model/S416 \
  --s428-models model/S428 \
  --s419-report model/S419/report.json \
  --s428-report model/S428/report.json --out model/notes_recognized
```

**Preconditions:** the historic original S45/S46/S79/S80 source model stack must previously have processed the WAV to provide event times and their **genuine S80 predicted K**; S433 only performs the protected join and the real sound-to-note identity inference. `--member` is an exact original native event source name, not an inferred song ID.

## 2. S434 — Genuine guitar recording, not synthesised input

S434 reads the real official GuitarSet **`audio_mono-mic.zip`**, MD5 **`275966d6610ac34999b58426beb119c3`**, plus untouched GuitarSet native S258 event rows and original frozen S397 S80 K for 59,309 rows. It automatically selects an original piece **without inspecting its note annotations**, converts one actual microphone recording to 44.1kHz mono PCM16, extracts original 42×49 harmonic features from the WAV, and calls trained native S402/S416/S428 note recognizers through S433.

Chosen real piece: **`00_BN2-166-Ab_comp.jams`**. Predictions for 30 native onset groups, with **7 groups where S80 predicts polyphonic K≥2**:

| Example predicted event | S80 K | Actual predicted note names |
|---|---:|---|
| Native event 579 | 3 | B3, E4, G♯4 |
| Native event 583 | 4 | D♯4, B3, D♯5, A♯3 |
| Native event 586 | 3 | E4, G♯3, G3 |
| Native event 590 | 2 | A♯3, C♯4 |
| Native K0 events | 0 | [] |

The output is a **prediction**, not ground truth. The **MIC source differs from S258 pickup-mix recording**; this distribution difference matters and is never hidden.

Code: [S434](https://github.com/Andriamarosoa/note/blob/codex/s276-idmt-song-disjoint-candidate-audit/scripts/s434_test_real_GuitarSet_MIC_WAV_to_musical_note_names_S80_source.py) · [successful run #38522246789](https://github.com/Andriamarosoa/note/actions/runs/38522246789), artifact `11874084244`.

An earlier run #38522107858 failed at Python compilation due to a missing closing parenthesis in diagnostic output. The source was fixed, and the **subsequent complete run succeeded**; the failed run is NOT cited as a completed experiment.

## 3. S435 — Independent re-scoring of frozen predictions on real JAMS notes

S435 downloads frozen S434 predicted note names **before accessing original JAMS annotation.zip**, MD5 **`b39b78e63d3446f2e54ddb7a54df9b10`**. It re-assigns reference note attacks against **all original S258 native onset groups of the piece**, reproduces the original full-piece true K count before any evaluation, and then compares exactly the same 30 selected event IDs without fitting a model or tuning a threshold.

Measured S435 diagnostics:

| Metric on chosen 30 true native events | Observed |
|---|---:|
| Correct original S80 K | **25 / 30 = 83.33 %** |
| Entire set of note names exactly correct | **22 / 30 = 73.33 %** |
| Note-instance F1 (new note names) | **63.8298 %** |
| True S80 K modified by note model | **0** |

**These 30 events are a small, deliberately identifiable example from GuitarSet, not a blind benchmark.** Many events are K0 and artificially increase the raw full-set correctness rate; the 73.33% is **not** the polyphonic chord success rate and must not be presented as a new S428 score. Original S80 has training overlap with GuitarSet, and MIC is a different recording modality from S258 pickup mix.

Example frozen failure:

| True new note names | Predicted by S434 |
|---|---|
| **A♯3, E4, G♯3** (true K3) | **B3, E4, G♯4** (pred K3) |
| **D♯4, B3, G♯3, C♯3** (true K4) | **D♯4, B3, D♯5, A♯3** (pred K4) |
| **E4, A♯3, G♯3, C♯3** (true K4) | **E4, G♯3, G3** (pred K3) |

Diagnosed residual pitch mismatches on this small sample: 4 other intervals, 4 missing fundamentals, 1 octave confusion, 1 fifth/19th harmonic case, 1 extra note (categories of greedy mismatch, not proved causal mechanisms).

Code: [S435](https://github.com/Andriamarosoa/note/blob/codex/s276-idmt-song-disjoint-candidate-audit/scripts/s435_audit_real_guitar_MIC_note_names_vs_JAMS_no_training.py) · [successful audit #38522497380](https://github.com/Andriamarosoa/note/actions/runs/38522497380), artifact `11874441410`.

## 4. Preserved best official and developmental scores

- Active S80/S397: **53,529/59,309=90.25% Exact-K all** and **6,054/7,385=81.98% poly**, development with prior GuitarSet overlap.
- Best acoustic identity challenger **S428**: **3,025/7,385 = 40.96% polyphonic complete pitch-set accuracy** on original GuitarSet development, with zero Exact-K changes.
- S431 octave-only replacement was **rejected** (10 chord corrections, 10 regressions, 0 net).
- Best complete pitch+Onset+Offset challenger S425 and reference Offset S408 both untouched. Live S424 also untouched.
- S433–S435 are **source-chain integration and real-sound diagnostics**, not a new promoted identity or K model.

## 5. Remaining blocker for complete one-command WAV-alone → E3

S433 eliminates manual K JSONL but **requires genuine original S80 predictions from historical pretrained upstream S5/S18/S45/S46/S69–S72/S79/S80 producers**, plus their corresponding native V88 onset times.

A genuine WAV-only application must invoke the original **V84→V88 native onset proposal stage**, then **16 S45 families / 193 S46 proposals / S69–S72 heads / S79–S80 final K**, preserve all model checkpoints and feature definitions and pass the native events into S433. This integration is **not yet implemented as a single generic WAV entrypoint**. Substituting harmonic peak counts, the V88 incumbent or manually typed K would silently regress to a different counting model and is not acceptable.

The next unbiased accuracy milestone requires fully *held-out from S80 training* recordings, after the full raw WAV counting chain is wired, with per-K whole-chord and individual note-name metrics plus corrections and regressions.
