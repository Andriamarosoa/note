# NOTE S764 — Real GuitarSet measurements for six isolated decoders

Verification 19 October 2026 — signed S708/S707 baseline replay, GuitarSet CAL32 composition-disjoint from FIT80.

Workflow: https://github.com/Andriamarosoa/note/actions/runs/39230990900
Implementation: scripts/s764_six_decoder_shadow_cal32.py
Source: src/causal_note/s763_independent_decoders.py

**Result:** 17/17 unit tests succeeded, all 32 real CAL recordings completed, all six signed S708 official scores reproduced exactly, no source MIDI event changed, no LIVE changes, no promotion.

## Six independent decoder diagnostics (SHADOW, not official final MIDI scores)

| Independent decoder | Definition | Measurement |
|---|---|---:|
| Exact-K | K0–K6 argmax accuracy on 31,568 1024-sample decision fragments | 69.6053% (21,973 / 31,568) |
| Active Exact-K | private acoustic sustain-hysteresis K estimate, same 31,568 fragments | 71.3824% (22,534 / 31,568) |
| Onset | independent NoteOn 80ms F1 | 72.8135% (TP 2764 / FP 745 / FN 1319) |
| Offset | independent UNPAIRED release boundaries, precision/recall F1 within 50ms | 51.2151% (TP 1570 / FP 509 / FN 2482) |
| Frame Pitch | independent binary per-pitch F1 on 1024-sample decision fragments | **84.5078%** (TP 40366 / FP 4059 / FN 10741) |
| Full Note | complete independently paired NoteOn/NoteOff event F1, original S704 formula | 55.6339% (TP 2726 / FP 708 / FN 1357 / correct Offset 2091) |

**Comparability warning:** These independent diagnostics use the four-hop **1024-sample decision cadence** (except event matching, which uses actual event samples). The official S704 Exact-K, Active Exact-K, Frame Pitch metrics use 256-sample frames. The Offset shadow metric measures UNPAIRED 50ms boundaries, while S704 official Offset is conditional accuracy among matched NoteOns; 51.215% versus 77.329% is NOT a direct regression or improvement comparison. Frame Pitch shadow >80% is not proof official frame-pitch-F1 >80%.

## Important per-K result on K6

Real CAL32 K6 decision fragments: 189.

- Frozen raw K head chose K6 correctly 0/189 = 0%.
- Independent Active Exact-K hysteresis counted six active notes in **89/189 = 47.09%**.
- K5 head correct 163/750 = 21.73%; independent Active Exact-K K5 correct 171/750 = 22.8%.
- K0 head correct 5501/6499 = 84.64%; independent ActiveK K0 correct 5337/6499 = 82.12%.
- Disagreement across all fragments: raw Exact-K vs Active-K in 9169/31568 = 29.05% of decisions.

This is the first measured evidence that independent sustain-state decoding recovers some high polyphony even when K head fails. However the 89 correct K6 decisions are not automatically correct pitches or attacks. K6 performance elsewhere can still regress if these raw counts are blindly forced into MIDI.

## Signed reference unchanged

Onset F1 74.2639%; Exact-K 62.0455%; Active Exact-K 56.2762%; Offset accuracy 77.3292%; Full Note F1 57.1319%; Frame Pitch F1 76.4626%.

These reference metrics remain the ONLY official GuitarSet CAL32 six-score results until a candidate MIDI stream is independently proven to improve ALL SIX without loss of previously correct notes. S764 did not attempt to promote. HELD and TEST not opened.

## Follow-up S765

Fit an independent causal event arbiter using the six private S763 decoders as evidence. Train/calibrate ONLY on FIT80 with group cross-validation by composition; freeze a risk threshold from FIT. On CAL32 evaluate FULL serialized MIDI and reject any candidate with six-score regression or lost prior correct offset. Source signed MIDI retains strict priority. This is research-only and doesn't change LIVE.
