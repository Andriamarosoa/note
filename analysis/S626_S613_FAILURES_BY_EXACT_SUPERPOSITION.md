# S626 — which exact note combinations S613 misses (measured, not hypothesized)

Verified GitHub Actions workflow:
https://github.com/Andriamarosoa/note/actions/runs/38884677850
Follow-up immutable artifact inspection:
https://github.com/Andriamarosoa/note/actions/runs/38885299484

Protocol: original frozen *full* S613 **neural + ONNX** (loaded with
`load_heads`), same original source-disjoint 17 GuitarSet/URMP DEV
recordings, first 24s/song, MIDI40..88, timestamped **NoteOn ±80ms**
maximum cardinality one-to-one matching. No test data/labels opened.
No model retraining, new selector thresholds, augmentation or hindsight
in online decisions.

**Exact canonical comparator:** 1241 TP, 734 FP, 838 FN, 61.2235%
NoteOn F1. All accounting tests and GitHub Actions succeeded.

## What "failed pair" means

One note-instance pair has two *different* annotated MIDI pitches
sustaining simultaneously for >=10ms. A pair is **FAILED** if the
actual emitted NoteOns fail to match one or BOTH expected attacks
within ±80ms; a pair can have only one note missed or both.
Not the same as pred K=2, not necessarily simultaneous *onsets*
(e.g., the notes may begin 500ms apart), not a harmonic-source
identification. In K3/K4 a missed note can participate in multiple
failed pairs, so totals across pairs MUST NOT be summed into 838 FN.

## Exact coactive pair ranking by total failed pairs

| Exact note pair | Failed pairs / examples | Failure % | Source compositions |
|---|---:|---:|---:|
| C#3+G#3 | **28/47** | 59.6% | 2 |
| A3+D4 | **23/30** | 76.7% | 5 |
| B3+D#4 | **23/30** | 76.7% | 3 |
| G#2+D#4 | **23/26** | 88.5% | 2 |
| G#2+C4 | 22/28 | 78.6% | 2 |
| D3+D4 | 22/26 | 84.6% | 2 |
| C4+E5 | 22/23 | 95.7% | 3 (URMP only) |
| D4+F#4 | 21/41 | 51.2% | 5 |

These are exact pitch instances, not intervals generalised to all
keys or MIDI octaves.

## Conditional worst hit rate with minimum evidence

Filter: >=10 instances and >=3 independent compositions:
- C4+C5 20/20 (100%; 4 comps, URMP)
- E4+E5 11/11 (100%; 3 comps, URMP)
- F#3+D4 11/11 (100%; 3 comps, guitar and URMP)
- C4+D5 10/10 (100%; 4 comps, URMP)
- E4+G5 10/10 (100%; 3 comps, URMP)
- C4+E5 22/23 (95.7%; 3 comps, URMP)
- A3+E4 10/11 (90.9%; 4 comps, guitar+URMP)
- A3+C#4 14/18 (77.8%; 4 comps, mostly guitar)
- G3+G4 14/18 (77.8%; 4 comps, URMP)
- A3+D4 23/30 (76.7%; 5 comps, guitar+URMP)
- B3+D#4 23/30 (76.7%; 3 comps, mostly guitar)

100% failure of a PAIR means neither/only one got detected as a
*pair*, not that every component pitch was missing every time.
Even 10-20 observed pairs can be statistically fragile and repeated
within tracks.

## Simultaneous new-onset full sets (30ms grouped attacks)

| Exact attacked set | Incomplete attack groups | Number of groups | Composition groups |
|---|---:|---:|---:|
| A2+E3 | 9 | 22 | 1 |
| A3+D4+F#4 | **8** | 11 | **3** |
| C#3+G#3 | 7 | 12 | 1 |
| A2+F#3 | 6 | 11 | 1 |
| G3+G4 | 6 | 6 | 3, URMP |
| B3+D#4+G#4 | 5 | 6 | 1 |
| E3+B3+D#4+G#4 | 5 | 6 | 1 |
| E3+G#3+D4+G4 | 5 | 5 | 1 |

The high-confidence (>=10 examples, >=3 compositions) worst whole
onset group is **A3+D4+F#4**, incomplete **8/11 =72.7%**. Most
other high missed sets are seen in just ONE composition, so not
independently substantiated despite repetition.

## Missed individual pitches (absolute)

F#3 46 misses out of 82; A3 41/91; B3 39/98;
A2 38/86; C#3 37/73; D4 36/122; E4 35/115;
C4 34/105; E3 34/78; G#3 34/90. This list is by **absolute**
misses, not failure rate; one of these misses may be part of a
different overlapped chord and/or single-note condition.

## False NoteOn context classification (exclusive by priority)

Of 734 S613 false positives:
- 230 (31.3%) at interval 12/19/24 semitones from one *active
  labeled note*, possible harmonic-confusable context;
- 198 same-pitch as a still-active labeled note, possible duplicate
  retrigger / timing mismatch;
- 122 at 1–2 semitones from an active labeled note;
- 130 at other intervals from an active labeled note;
- 54 when no labeled note active.

This is an annotation-context classification, NOT proof of the
acoustic cause of an FP. Need spectrum + source-stem audit before
calling 230 "harmonic hallucinations" or 198 "repeated attacks".

## E2+E3 and chord question

In S625, DEV has **just one annotated E2+E3 overlapping pair**
(and one 30ms onset group), vs TRAIN 46 pairs in only four different
compositions. This DEV denominator is too small for a credible
conditional failure rate. Do NOT conclude that E2+E3 is solved
because it does not appear among the top pair errors, nor that
new E2+E3 training mixes necessarily improve F1.

F major+E minor union was entirely absent in TRAIN and DEV first
24 seconds; **no model miss rate can be computed for a nonexistent
DEV condition**. A chord-progression F -> Em is different from
simultaneous chord overlay.

## Decision gate, not premature recommendation

This audit proves **which exact annotated note combinations fail the
current S613 model** on original DEV music. It does not prove why
(e.g. harmonic overlap, onset head, corpus/instrument domain), or
that specific mixtures improve generalization. Next audit should
separate GuitarSet and URMP, control for note pitch individual
difficulty and active K, compare FN scores/onset heads and real
time offset, and validate any changes on independent compositions.
Only after that can targeted synthesis be justified.

Full reproducible code: `scripts/s626_audit_missed_combinations.py`.
Full JSON report preserved in Actions artifact
`s626-full-S613-exact-combination-FN-FP-audit-17dev`.
