# S770 — Joint causal onset research (19 October 2026)

## Motivation

S767 recovered only 97 of 1170 previously missed true onsets using the old
S765/S763 proposal pool, even with oracle matching. S768's all-pitch
hypotheses (49 MIDI pitches) improved candidate recall but the FIT-selected
threshold scored 73.7965% onset F1 on CAL32, below frozen S708's 74.2639%.
S769's independent source-onset false-positive veto scored 74.3748%, also
far from 80%. A better selector alone cannot solve missing acoustic candidates.

## What S770 tests

Two independently trained causal experts:
- S768 all-pitch missing-attack generator.
- S769 source NoteOn quality model.
- Optional partial-rise gate (present/past acoustic features only).

65 candidate shadow streams = 5 native quality thresholds x
(1 native-only + 4 extra-birth thresholds x 3 rise floors).

FIT80: composition-grouped five-fold OOF models. Freeze a single
configuration chosen by aggregate onset F1 on FIT. Retrain on FIT80 only.
CAL32: measure all 65 diagnostic variants with frozen models and selected
threshold. The CAL32 tuning history makes this split research-only, not
an untouched final test.

The original S708/S707 models and MIDI streams must reproduce all six
signed reference scores to 1e-8. Neither LIVE nor TEST/HELD changes.
These Onset-only hypotheses have no paired NoteOff and are NOT a new
official final-MIDI baseline even if Onset F1 exceeds 80%.

## Output and promotion

Workflow: .github/workflows/note-s770-joint-onset-fit80.yml
Script: scripts/s770_joint_onset_fit80.py
Tests: test/test_s770_joint_onset_fit80.py

The report includes the single OOF-selected FIT configuration and each CAL
score, strict reference parity and counts. No silent promotion.

Future work only after a genuinely improved onset candidate:
instance-aware causal NoteOn/NoteOff identity preservation, independent
new audio held-out evaluation and all six S704 final-MIDI non-regression
gates, followed by instrumented LIVE latency/CPU testing.
