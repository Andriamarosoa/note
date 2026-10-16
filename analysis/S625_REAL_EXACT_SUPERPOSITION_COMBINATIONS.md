# S625 — Exact superposition combinations of the ORIGINAL NOTE annotations

Audited **2026-10-16** on verified original annotations. Sources:
- [S625 GitHub Actions SUCCESS run 38883377787](https://github.com/Andriamarosoa/note/actions/runs/38883377787), 11 tests pass; complete 70-source JSON artifact: `s625-real-exact-note-combinations-53train-17dev`.
- [S624 census](https://github.com/Andriamarosoa/note/actions/runs/38881478790) independently verified **TRAIN 7,361 notes / 11,888 overlapping pairs; DEV 2,082 notes / 2,750 overlapping pairs** under identical 24-second, MIDI40–88 scope.

This answers a specific audit omission: S624 counted ONLY intervals such as octave 12 semitones and broad K; it did NOT enumerate **which notes/pitches or chord-class unions** were simultaneously present.

## Exact audited cases (53 TRAIN / 17 DEV sources)

| Case | TRAIN | DEV |
|---|---:|---:|
| E2+E3 [MIDI40,52] note-instance pairs overlapping >=10ms | **46** | **1** |
| E2+E3 distinct onset clusters within 30ms (can have other notes) | **12** | **1** |
| E2+E3 active simultaneously (seconds) | **7.43628** | **0.940** |
| E2+E3 source recordings with >=10ms pair | **4** | **1** |
| E2+E3 independent composition groups | **4** | **1** |
| E3+F3 [52,53] overlapping pairs >=10ms | **1** | **0** |
| E3+F#3 [52,54] overlapping pairs >=10ms | **1** | **0** |
| E2+E3+G3 coexistence, seconds (may include other active notes) | **5.10238** | **0.395** |
| F *single pitch-class F* plus complete E minor triad E/G/B, simultaneous pitch-class inclusion | **0.08467s; 3 active passages** | **0** |
| Full F MAJOR chord (F/A/C) plus E MINOR chord (E/G/B) pitch classes all active | **0** | **0** |
| Exact voicing F3 A3 C4 + E3 G3 B3 all active | **0** | **0** |

**Interpretation of F + Em:** the guitar chord symbol F is not the F note. We audit both:
- F *note* plus Em: pitch classes {F,E,G,B}; three brief training passages.
- F *major chord* plus Em *minor chord*: union {F,A,C,E,G,B}; never simultaneously present among note labels in this corpus.
- These labels indicate simultaneous PITCH CLASSES, NOT independently played/recorded chord tracks. A normal six-note chord voicing can contain those classes without being two separate played chords. Likewise a sequential F -> Em harmonic change requires a DIFFERENT, transition-based audit.

## Most frequent MIDI-pitch PAIRS among existing TRAIN notes (>10ms overlap)

| Exact pitch pair | Note-instance pairs | Independent compositions |
|---|---:|---:|
| C3+C4 | 168 | 8 |
| E3+C4 | 156 | 6 |
| F3+F4 | 150 | 6 |
| E3+G3 | 136 | 6 |
| E3+E4 | 135 | 9 |
| D3+G3 | 134 | 8 |
| D4+A#4 | 128 | 9 |
| G3+C4 | 127 | 7 |
| A3+C4 | 127 | 8 |
| F3+C4 | 119 | 8 |
| F3+A3 | 118 | 11 |
| A3+F4 | 117 | 9 |
| D3+D4 | 116 | 8 |
| D4+G4 | 114 | 11 |
| G2+G3 | 112 | 7 |

## Most frequent simultaneous onset pitch sets (<=30ms) in TRAIN

These counts are **whole onset groups** by distinct pitches, NOT pair counts.
Each row is 30ms onset grouping, irrespective of continued sustain:

| Exact onset note set | Groups | Independent compositions |
|---|---:|---:|
| D4+F#4 | 26 | 3 |
| G3+B3 | 24 | 5 |
| D#3+G3 | 21 | 1 |
| F3+A#3+C#4 | 20 | 2 |
| G3+C4 | 19 | 5 |
| B3+D4 | 16 | 3 |
| C4+E4 | 16 | 4 |
| E3+G3 | 14 | 5 |
| A3+C4 | 14 | 5 |
| D4+F4 | 13 | 4 |
| D#4+G4 | 13 | 5 |
| A#2+F3+A#3+D4 | 12 | 2 |

## Diversity and repetition

| Metric | TRAIN | DEV |
|---|---:|---:|
| Analyzed original sources | 53 | 17 |
| Usable MIDI40–88 note events in first 24s | 7,361 | 2,082 |
| Exact differently pitched note-instance overlaps >=10ms | 11,888 | 2,750 |
| Distinct exact MIDI pitch pairs with observed overlap | **640** | **402** |
| Distinct whole multi-note 30ms onset MIDI pitch sets | **566** | **203** |
| Distinct simultaneously active MIDI pitch sets | **1,964** | **756** |
| Onset groups K2 | 827 | 255 |
| Onset groups K3 | 463 | 183 |
| Onset groups K4 | 205 | 44 |
| Onset groups K5 | 67 | 0 |
| Onset groups K6 | 26 | 0 |

The difference between a count of 168 C3+C4 note pairs and only 8
independent compositions is material. Repeated pairs within the same
performance are correlated examples; they are NOT 168 independent
recordings of that acoustic conflict. Some common exact K3/K4 sets
come from as few as 1–3 source compositions.

## What this DOES and DOES NOT prove

**Proved:** Exact E2+E3 overlaps are very concentrated, some adjacent
pairs are represented only once, and F major+E minor union absent
from the annotated 24s NOTE training/development split. Generic
polyphony was common but exact combinations are uneven.

**NOT proved:** These cases caused the model's actual False Negatives,
that current original audio presents an objectively inseparable
spectrum, or that generating extra E2+E3/F+Em mixtures improves
NoteOn F1, true-K or LIVE latency.

Next legitimate design gate:
1. Pair FN/FP decisions from S613 hybrid on the 17 dev recordings with
   **these exact composition-condition labels**, including E2+E3,
   adjacent semitones, differing K and successive onsets.
2. Compute per-exact-combination conditional recall/precision, confidence
   calibration, count-K and missed-onset mechanism.
3. Prioritize only combinations with significant observed errors,
   low independent-composition diversity and physically realistic
   realizable acoustic sources. Do NOT target rare patterns simply
   because they are rare.
4. Only then produce train-only additional real/stem-mixed data,
   source-group-isolated from DEV/new heldout evaluation; leave
   original NOTE S613 hybrid and S80 references intact.
5. Evaluate physically strict LIVE NoteOn and group-K1..6 separately.
