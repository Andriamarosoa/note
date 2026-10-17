# S655 — Causal Tone/K selection, measured real GuitarSet

Workflow **success**: https://github.com/Andriamarosoa/note/actions/runs/39007773271

**No promotion:** the TRAIN five composition calibration split selected
`S614_original` (F1 77.8057%). Every conditional Tone selector
was below original on calibration. Additional TRAIN held split
(four different compositions) had better scores for some variants,
but did not override the predeclared selection criteria.
DEV below was historically exposed and is exploratory.

## Actual NoteOn GuitarSet DEV (1580 GT onsets, ±80ms)

| Selector | TP | FP | F1 | K3+ TP |
|---|---:|---:|---:|---:|
| S614 official original | 998 | 585 | 63.10465% | 550 |
| S652 average 3 views | 974 | 509 | 63.59778% | 530 |
| Tone when original K>=2 p>=0.25 | 976 | 505 | 63.77001% | 530 |
| Tone when original K>=2 p>=0.50 | 973 | 501 | 63.71971% | 528 |
| Tone when original K>=2 p>=0.75 | 974 | 502 | 63.74346% | 527 |
| Tone when original K>=3 p>=0.25 | 974 | 505 | 63.68094% | 530 |
| Tone when original K>=3 p>=0.50 | 979 | 514 | 63.71624% | 533 |
| **Tone, original K>=2 p>=0.35, only register high** | **978** | **506** | **63.83812%** | **529** |
| Tone, register low | 995 | 582 | 63.03453% | 550 |
| Tone when original K uncertain | 996 | 577 | 63.17793% | 550 |
| Tone light 50% when poly | 996 | 570 | 63.31850% | 546 |
| Original with optional filtered corroboration | 1001 | 587 | 63.19444% | 550 |

The exploratory DEV best high-register gate removes 79 false
attacks but also loses 20 true attacks versus S614, and
loses 21 true K3+ attacks (550 -> 529).
It is **not** a generalized polyphonic gain.

Source-level high-register differences versus S614 (TP/FP):
- `04_BN3-119-G_comp`: −7/−2;
- `04_BN3-154-E_comp`: −1/−11;
- `04_Funk3-112-C#_comp`: −4/−19;
- `04_Funk3-98-A_comp`: −11/−8;
- `04_Rock1-130-A_comp`: 0/−17;
- `04_Rock1-130-A_solo`: +4/−5;
- `04_Rock1-90-C#_comp`: −1/−15.

These data show **no fixed K threshold uniformly protects all chords**.
The promising structure is a task-specific head that recognizes
actual coincident/independent onset evidence and uses spectral
views only to rescue a missing note, not to filter every high note.

Official NOTE S614 model/Exact-K/runtime unchanged.
No actual Fender Stratocaster Tone-knob recording was used.
No future frames and no extra delay were intentionally introduced;
production latency was not measured.
