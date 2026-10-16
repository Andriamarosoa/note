# S624: real annotated NOTE polyphony coverage — measured, not guessed

Source: [successful GitHub Actions audit 38881478790](https://github.com/Andriamarosoa/note/actions/runs/38881478790). Full machine-readable per-source report in Actions artifact `s624-source-true-annotated-polyphony-census`.

Scope: Original pinned **S613 TRAIN (53)** and **DEV (17)** compositions. GuitarSet and URMP; only the **first 24 s** of each recording; only model-supported **MIDI40..88** pitches. Existing labeled note events only, NO synthesized mixtures, no model predictions, no held-out TEST accessed. Pinned GuitarSet JAMS annotation ZIP sha256 verified; URMP note annotations downloaded by S613's source-resolved parser.

## How the data are counted

1. **Simultaneous active K**: integrate duration during which at least two *different* MIDI note pitches are sustaining in overlapping time intervals. This is different from simultaneous note attacks. No need for audio samples.
2. **Onset K within 30ms**: group note starts by <=30ms difference from first onset in group; count distinct note pitches. Not a ground-truth number of vibrating strings, NOT the historical S80 Exact-K.
3. **Overlapping pairs**: each unordered pair of two note *instances* of different pitch whose durations overlap at least 10ms. A single K4 chord produces six pitch pairs, so **pair count is NOT number of audio examples/chords**.
4. **Close pitches**: among such co-active pairs, interval 1 or 2 MIDI semitones. This represents potentially difficult *note fundamentals*, not actual spectral source overlap.
5. **Harmonic-confusable intervals**: co-active pair intervals 12,19,24 semitones (2x, 3x, 4x rounded, respectively); not acoustic proof that one peak is an overtone rather than a second note.
6. **Quasi-simultaneous attacks**: distinct pitches where the second starts <=10/30/50/100ms after the earlier and the earlier is still sustaining. Different from simply being co-active.

## Real original annotations: TRAIN vs DEV

| Audit count | TRAIN | DEV |
|---|---:|---:|
| Sources | **53** | 17 |
| GuitarSet | 36 | 12 |
| URMP | 17 | 5 |
| Total source audio duration analyzed | 1,272s | 408s |
| Original note annotation rows (full composition) | 19,484 | 4,285 |
| Usable NOTE MIDI40..88 within first 24 s/track | **7,361** | 2,082 |
| Label rows excluded as past 24s / invalid duration | 12,000 | 2,170 |
| Label rows outside MIDI40..88 | 123 | 33 |
| Duration with K-active>=2 distinct pitches | **588.046s** | 184.373s |
| Percent of all audio in K-active>=2 | **46.23%** | 45.19% |
| Percent of sounding duration with K-active>=2 | **58.55%** | 58.19% |
| Source clips with at least 1 polyphonic interval | **53/53** | 17/17 |
| Distinct pitch pairs overlapping >=10ms | **11,888** | 2,750 |
| Pitch-pairs separated by 1 or 2 semitones | **202** | 48 |
| 1 semitone pairs | 47 | 14 |
| 2 semitone pairs | 155 | 34 |
| Harmonic-confusable (12/19/24) pairs | **2,412** | 428 |
| Octaves 12 semitones | 1,570 | 268 |
| 19 semitones | 541 | 135 |
| 24 semitones | 301 | 25 |
| Pairs with second onset <=30ms after first onset | **4,908** | 1,173 |
| Notes with attack while different pitch was still active | **5,550** | 1,441 |
| Source clips with any close-pitch pair | 46/53 | 13/17 |
| Source clips with any harmonic-confusable pair | 35/53 | 12/17 |

### Attack groups (distinct MIDI pitches started <=30ms apart)

| Onset group size | TRAIN | DEV |
|---|---:|---:|
| K1 | 2,971 | 834 |
| K2 | **827** | 255 |
| K3 | **463** | 183 |
| K4 | **205** | 44 |
| K5 | **67** | 0 |
| K6 | **26** | 0 |
| Total onset groups | **4,559** | 1,316 |

K2+ is 1,588/4,559=34.83% of TRAIN onset groups. K3+ is
761/4,559=16.69%; K5/6 are 93 groups (2.04%). This is a marked
imbalance despite polyphony being common by duration.

### True simultaneous ACTIVE distinct-pitch K coverage (seconds)

| K currently active | TRAIN | DEV |
|---|---:|---:|
| 0 (silence) | 267.575 | 91.125 |
| 1 | 416.379 | 132.502 |
| 2 | 236.451 | 54.391 |
| 3 | 146.349 | 72.952 |
| 4 | 133.754 | 33.905 |
| 5 | 52.094 | 23.121 |
| 6 | 19.397 | 0.004 |

**GuitarSet and URMP distribution differ considerably**:
- TRAIN GuitarSet 36 files: 5,289 usable note events, 350.719s overlapping, 9,260 co-active pairs, 104 within 1–2 semitones, 1,983 harmonic-confusable.
- TRAIN URMP 17 files: 2,072 usable note events, 237.327s overlapping, 2,628 co-active pairs, 98 within 1–2 semitones, 429 harmonic-confusable.
- GuitarSet provides all **K4–K6 near-simultaneous attack groups** on TRAIN (K4=205,K5=67,K6=26); URMP contributes K2=347,K3=71 only to this 30ms clustering.
- The two instruments/corpora differ in microphone source, pitch range and note articulation; don't treat independent URMP mixtures as identical to real guitar strings.

## Falsification of the prior ungrounded claim

**The statement 'we need more polyphonic data' was NOT established.**
Already TRAIN has 46% overlapping-audio time and 11,888 co-active
distinct-pitch pairs; source groups 53/53 have real polyphony. Simply
adding arbitrary layered audio cannot be justified by a claimed lack
of generic overlap.

What we CAN justify next:
1. Match every true false negative/false positive of frozen S613 and
   S619/S623 to its **existing annotation condition** (onset-K,
   active-K, intervals 1/2/7/12/19/24, onset offset and corpus).
2. Measure training-case count and per-condition FN/FP **rate**;
   frequency alone does not tell us whether the model learned it.
3. Check how many *independent underlying compositions* cover each
   hard condition to avoid many near-duplicates inflating the count.
4. Only if a condition has inadequate exposure and/or measured poor
   independent recall on DEV/new music, construct targeted synthetic
   TRAIN-only mixes from source-disjoint stems or new real annotations.
   Never leak source recordings across TRAIN and DEV, and never
   tune thresholds on held-out TEST.
5. Validate NoteOn F1 and K1..K6 without new future-lookahead or
   microphone-to-piano latency regressions.

## Limits (must not be omitted)

- Scope is **S613 pinned training/development**, not every historical
  NOTE dataset ever used or all audio ever recorded.
- Counts use annotation-defined MIDI pitch and note duration; a
  co-active octave is a *potential* ambiguity, not proof that one
  audible spectrum contains a clearly independent E3 and E2.
- All recordings limited to the first 24 seconds; excluded original
  annotation rows are shown above. Further training variants may have
  additional/different corpora and cannot inherit these counts.
- The 30ms onset clustering and >=10ms overlap threshold are declared
  operational definitions, not inviolable music truths.
- Repeated chords in one track share timbre, instrument and arrangement;
  pair or event counts are not independent unique training examples.
- This audit establishes **dataset content**; it does NOT establish
  whether adding any specific synthetic data improves F1/Exact-K.
