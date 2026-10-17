# S655 — Per-hop Tone selection by original Exact-K and register: real result

## Verified run and verdict

GitHub Actions success:
https://github.com/Andriamarosoa/note/actions/runs/39007773271

- script `scripts/s655_k_conditioned_tone_selector.py`
- tests `test/test_s655_k_conditioned_tone_selector.py`
- archive `S655_GUITARSET_REAL_TONE_NATIVE_K_CONDITIONAL/report.json`

No promotion. The route preselected on TRAIN is S614 unchanged,
although some exploratory variants on GuitarSet DEV perform above
S641 and S652.

12 precommitted causal selection policies evaluate original,
Tone averaged, K>=2 conditional variants, K3 confidence variants,
register gated variants, uncertain K and Tone-confirmed rescue.
All use 2 simulated forward Butterworth lowpasses 2800/1100Hz
and ORIGINAL unchanged K posterior. Original pitch model, Exact-K
count head and LIVE microphone runtime are frozen.

36 GuitarSet TRAIN tracks (5 composition calibration and 4
composition internal validation), then 12 historical DEV tracks.
TEST closed.

## TRAIN decisions

Five calibration compositions:
- S614 unchanged: **77.8057% F1**.
- Triple Tone average: 76.8000%.
- Tone when native K>=2 confidence≥0.25: 76.5608%.
- Tone on high register when polyphony: 77.1422%.
- Tone when K uncertain: 77.5656%.
- Native or filtered attack when poly: 77.7360%.

**Every Tone policy underperforms S614 on TRAIN calibration.**
Protocol therefore preselects original S614; no DEV-guided promotion.

Four other TRAIN compositions:
- S614: 70.5360%.
- Triple Tone average: 71.7915%.
- Tone when native K>=2 confidence≥0.25: 71.7487%.
- Tone high-register when polyphonic: 71.0080%.
- Tone strong uncertainty: 70.9548%.
This split instability is a major generalization warning.

## DEV GuitarSet real true NoteOn F1 (±80 ms)

| Candidate | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Original S614 | 998 | 585 | 582 | 63.10465% |
| S652 triple Tone average | 974 | 509 | 606 | 63.59778% |
| Tone poly posterior≥0.25 | 976 | 505 | 604 | 63.77001% |
| Tone poly posterior≥0.50 | 973 | 501 | 607 | 63.71971% |
| Tone poly posterior≥0.75 | 974 | 502 | 606 | 63.74346% |
| Tone K3 confidence≥0.25 | 974 | 505 | 606 | 63.68094% |
| Tone K3 confidence≥0.50 | 979 | 514 | 601 | 63.71624% |
| **Tone high register on polyphony** | **978** | **506** | **602** | **63.83812%** |
| Tone low register on polyphony | 995 | 582 | 585 | 63.03453% |
| Tone when K uncertain | 996 | 577 | 584 | 63.17793% |
| Half Original+Mild only when K>=2 | 996 | 570 | 584 | 63.31850% |
| Original OR confirmed Tone on polyphony | 1001 | 587 | 579 | 63.19444% |
| S641 historical experimental | 1009 | 594 | 571 | 63.39931% |

**Important:** 63.8381% is post-hoc the best among 12
methods examined on previously exposed DEV; it does not meet
TRAIN preselection and is not generalization evidence.

The 63.8381% variant has K3+ NoteOn TP **529**
against original S614 **550** and S641 **557**.
It therefore harms real polyphony while suppressing FP.

## Interpretation for original têtes/sélections architecture

The feature is useful but globally mixed strategies are unreliable.
A register-conditional Tone view can suppress harmonic errors,
yet it can delete the very low note whose recovery motivated
the experiment. Dedicated causally triggered **simultaneous
octave rescue** from S657 needs separate evaluation S658 on
natural GuitarSet songs, with anti-doubling and no K1 regression.

No modification of original S614, S641, S80, or microphone LIVE.


## Post-hoc group-level robustness audit

The 12 DEV recordings come from **only three distinct
compositions** (BN3, Funk3, Rock1), despite 12 audio tracks.
The best observed `tone_poly_high_register` vs S614 changed:

| DEV composition | F1 difference in percentage points | TP diff | FP diff |
|---|---:|---:|---:|
| BN3 | -0.669 | -8 | -12 |
| Funk3 | +0.005 | -15 | -30 |
| Rock1 | **+2.913** | +3 | -37 |

Thus the global +0.734 point gain is **concentrated in Rock1**.
A paired cluster bootstrap resampling the three compositions
with replacement (10,000 samples, seed 1234) gives a descriptive
2.5–97.5 percentile range of about **−0.669 to +2.913 F1
points**. Only **three groups** exist, so this interval is
too unstable to be a rigorous frequentist confidence bound,
but the sign reversal in BN3 is concrete.

This confirms that selecting the maximum DEV result would be
overfitting; held-out truly independent compositions are required.
