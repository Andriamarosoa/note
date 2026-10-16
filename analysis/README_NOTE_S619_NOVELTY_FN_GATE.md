# S619: independent acoustic novelty gate for missed NOTE onsets

Date: 2026-10-16. Research-only branch based on S618;
never merged to official S613 LIVE / S80.

## Measured priority evidence from S618 (17 real original DEV songs)

Original frozen S613 FULL hybrid (neural + ONNX) had 1241 true NoteOn,
734 false NoteOn and 838 missed NoteOn at ±80ms.

Offline audit of 838 misses:
- 702 (83.8%) onset < original .60 threshold within annotation window.
- 103 (12.3%) onset/sustain not above thresholds simultaneously.
- 33 (3.9%) selector state/refractory/temporal matching.

Causal constant-onset threshold ablations on original 17 DEV songs:
- .55/.50 rescues +30 real notes, loses 0 original correct notes,
  but adds +71 false notes.
- .50/.50 rescues +57 and loses 0 but adds +174 FP.
- .45/.50 rescues +79 and loses 0 but adds +267 FP.
- .45/.40 rescues +126, loses 2, and adds +503 FP.
These numbers are development only and describe tradeoffs, not a
new independently validated F1/Exact-K.

## Goal

Use existing NOTE 126-feature causal spectral bank to recognize
independent new attacks among weak neural onset candidates,
without lowering threshold uniformly for every note/harmonic.
Avoid harming correct baseline S613 notes or true E2+E3 octave pairs.

## Strictly causal prototype

`scripts/s619_novelty_onset_gate.py`:
- preserves every original head event whose onset>=.60;
- only considers .45/.50/.55 onset minimum when same-frame sustain>=.50;
- requires per-pitch positive score novelty (same NOTE frontend, no new FFT):
  * 2-hop physical spectral-score change, columns 60:63,
  * or contrast to causal recent minimum, columns 114:117;
- also requires absolute novelty>=.04 and relative novelty>=15/35/55%
  of most novel pitch in same frame;
- promotes the candidate to onset=.60001 for the unchanged
  CausalSelection. The selector itself retains its own
  K0..6, on/off state, refractory, duration, confirmation and budget.

Note: Relative acoustic contrast cannot by itself prove sound-source
independence, especially close pitches and harmonic siblings. This
is a strict AB test of two candidate evidences; do not turn on
the gate in production or claim that it was trained.

## Experimental validation

`scripts/s619_compare_novelty_rescue.py` measures 18 candidate gates on
the same 17 original GuitarSet/URMP development songs:
- TP/FP/FN and NoteOn F1 ±80ms;
- identity-matched recovered previously false-negative notes;
- identity-matched lost previously true-positive notes;
- grouped onset K1..6 and complete pitch sets;
- GuitarSet vs URMP per-corpus scores.

GitHub Actions run:
https://github.com/Andriamarosoa/note/actions/runs/38869340016
Read run conclusion and artifacts before claiming success or numbers.
This is development-only hyperparameter selection and cannot justify
model promotion.

## Correct next-step gate for a trained S619 head

If no deterministic novelty gate gets better FN/FP compromise:
train an explicit per-pitch *independent onset candidate* head from
original TRAIN split with native causal feature, S613 frozen network
onset probability and K-head probability as inputs. Give it positive
labels only for genuinely unmatched acoustic new note attacks, including
octaves and adjacent pitches, and hard negative labels for their
harmonic siblings, duplicated picked attacks, sustained partials.
The labels must never feed online inference. Calibrate only on
the DEV composition groups and evaluate independently on unseen
GuitarSet/URMP/IDMT and target iRig recordings.

Promotion requires positive FN recovery with acceptable extra FP,
minimum regressions of original TP, strict K3/K4 preservation,
low latency p95 and no future-sample dependency.

## COMPLETED 17-song development experiment — verified 2026-10-16

GitHub Actions [38869340016](https://github.com/Andriamarosoa/note/actions/runs/38869340016):
**SUCCESS**, all causal gate tests pass and all 18 candidate policies
were evaluated with the true frozen S613 ONNX+neural hybrid. Pinned
17 dev GuitarSet/URMP recordings; 2,079 ground-truth NoteOn events.

| Native configuration | TP | FP | FN | NoteOn F1 ±80ms | Recovered previous FN | Lost old TP | Grouped K3 | K4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen S613 | 1241 | 734 | 838 | 61.2235% | 0 | 0 | 42/183 | 3/44 |
| Global onset .50 | 1298 | 908 | 781 | 60.5834% | +57 | 0 | 41/183 | 4/44 |
| **S619 acoustic independent onset gate** (.50 onset min, 55% relative 2-hop novelty) | **1283** | **801** | **796** | **61.638%** | **+42** | **0** | **42/183** | **4/44** |
| S619 maximize recall gate (.45 onset min, 15% recent novelty) | 1313 | 946 | 766 | 60.535% | +72 | 0 | see artifact | see artifact |

Best F1 gate details:
`RescueGate(weak_onset_min=.50,
relative_novelty_min=.55, absolute_novelty_min=.04,
novelty_mode="lag")`.

**Finding:** restricting weak-onset promotion using already-computed
causal spectral novelty recovers 42 true notes while losing **zero**
previous true positives. Additional false notes are +67 versus +174
for global .50 onset threshold. F1 improves by about +0.41 percentage
points in-sample (development). The gain is small; K3 grouped exact
count stays 42/183 and K4 rises 3/44 -> 4/44; the polyphonic score
remains inadequate. This is not validation of a trainable S619 head.
Dev-policy selection is optimistic; re-test independent source groups
and real iRig input before ANY promotion.

We should now train a genuine NOTE-native independent-onset rescue
head on labeled original TRAIN sources rather than keep hand-tuning
on the same 17 DEV songs. The unchanged existing S613 note output
must remain valid in the absence of head confidence, and the
specialist must be tested on true E2+E3 and adjacent-note attack pairs.
