# S627 — Forensic audit, not new selectors chosen by intuition

**Protocol:** pinned NOTE S613 full native ONNX-tree+neural hybrid; 17
GuitarSet+URMP DEV source compositions; 24 seconds each; MIDI40..88;
NoteOn matches ±80ms. No held-out TEST access. Official S613, S80,
S619/S623 checkpoints and existing LIVE MIDI production are unchanged.

This follows S624 (real K/duration census), S625 (true exact
E2+E3 / F+Em chord-presence counts), S626 (which exact pairs/chords
are missed by the original model), and addresses their limitation:
the fact that a pair is frequently missed **does not establish WHY**.

## Atomic unit

One *real note event*:
- label = original S613 NoteOn within ±80ms: matched TP versus FN.
- all notes counted exactly once, unlike K3/K4's multiple possible
  pair relations.
- observed context: corpus, true independent composition, exact
  MIDI pitch/low or high register, duration, active-K, onset-K,
  co-active pitch intervals and same/other-note attack timing.
- actual *first available* frame features: log harmonic H1/H2/H3,
  harmonic strength ratios, three FFT resolutions, spectral novelty
  at exact physical 2/4/8/16-hop delays and short/long RMS.
- actual frozen head outputs onset/sustain/release, K0..6 confidence;
  original selector active/armed/high/low state BEFORE deciding.
- *post-hoc* maximum over ±80ms kept separate from online information;
  it can identify potential head/selector disagreement but CANNOT be
  used to propose zero-latency LIVE output.
- matched successful TPs as controls for FNs, first exact same
  composition/pitch/active-K, relaxing only when insufficient.
- per-split denominators, composition support and tiny-bin warnings.

One *false predicted NoteOn*:
- real 49-pitch score/selector state at emission, not a guessed source.
- relate to currently labeled active pitches and same-pitch nearby
  truth onset; distinguish late ±80ms alignment from a likely
  harmonic-interval, close-note, stale/repeated note-on or no
  labeled activity.
- note: a 12-semitone relationship alone **does not prove a physical
  overtone**, and an ON/OFF mismatch alone does not establish wrong
  labels. Requires actual spectrum/stem checks.

## Code / immutable run

- `scripts/s627_multidimensional_failure_audit.py` writes full
  per-note and per-false-event JSON plus conditional summaries.
- `test/test_s627_multidimensional_failure_audit.py` checks exact
  feature indices, K-active vs near-onset K, matched controls,
  proportions and causal pre-step state.
- [GitHub workflow run](https://github.com/Andriamarosoa/note/actions/workflows/note-s627-multidimensional-forensic.yml).

**Critical ABI correction:** from
`src/causal_note/strict_live.py CausalPitchFeatures._frame`,
feature columns **54–59 are two-hop lag** and **60–65 are four-hop lag**.
The S619/S620 research scripts earlier called columns 60–62
"two-hop novelty"; that description is wrong. The present audit
records and distinguishes both physical lags explicitly before
any new feature selection. Do not misattribute a training gain to
a 2-hop attack feature without verifying this.

## Scientifically valid sequence after S627

1. Verify original **TP=1241, FN=838, FP=734** by exactly reproducing
   frozen full hybrid S613 before inspecting condition differences.
2. Compare FN vs TP conditional rates, including their denominators,
   corpus, composition, pitch/register, K, onset lag.
3. Inspect paired within-source/within-pitch contrasts; if no good
   matched control exists, report it instead of generalizing.
4. Screen interacting factors: e.g. low pitch + K>=3 + an octave +
   delayed other-pitch onset; do not assume any alone caused failure.
5. Inspect actual spectral time series of a curated set of TP/FN/FP
   matched triples, and cases where label/algorithm disagree;
   head probabilities can expose ambiguities without proving source.
6. Train or augment **only** for conditions with demonstrated poor
   independent-composition performance AND inadequate diverse
   training coverage (S624/S625), with physically realistic audio.
7. Confirm on new source-disjoint held-out tracks, same NoteOn and
   Exact-K K1–K6 plus actual Windows iRig+FluidSynth p95 latency.
   Do not promote a dev-tuned rule because it helped one chord.

This report is an *audit design*. Please see the actual workflow
conclusion and its archived JSON before asserting measured S627
correlations, let alone a causal diagnosis.


## S627 verified result — 2026-10-16

[Successful full 17-DEV workflow](https://github.com/Andriamarosoa/note/actions/runs/38889998300).
**All unit tests and numerical audit assertions passed.**
Artifact `s627-original-S613-17dev-each-TP-FN-FP-all-dimensions`
contains per-event forensic features for **2,079** true note attacks and
**734** incorrectly emitted NoteOns. Full frozen source S613 comparator
**TP=1241, FP=734, FN=838** exactly reproduced.

### Miss rate conditioned on genuine active pitch count (one true attack counted once)

| True active distinct pitches | Missed / total expected notes | Miss % |
|---|---:|---:|
| K1 | 219/639 | **34.27%** |
| K2 | 192/500 | **38.40%** |
| K3 | 220/544 | **40.44%** |
| K4 | 117/260 | **45.00%** |
| K5+ | 90/136 | **66.18%** |

### Miss rate conditioned on coactive pitch interval

These are mutually exclusive priority categories at an expected onset.

| Existing pitch relation | Missed / expected | Miss % |
|---|---:|---:|
| No other pitch simultaneously active | 219/639 | 34.27% |
| Other coactive pitch interval | 289/753 | 38.38% |
| Perfect-fifth MIDI interval 7 | 89/236 | 37.71% |
| Semitone/whole-tone 1 or 2 | 29/59 | 49.15% |
| Octave/12th/two-octave (12/19/24) | 212/392 | **54.08%** |

This is a **pitch-interval ASSOCIATION**, not acoustic proof that a
harmonic of a single note was mistaken for a truly independent note.
Higher polyphony, pitch/register, corpus and onset timing can confound
this comparison.

### Time since latest other-pitch attack

| True recent OTHER pitch attack | FN/total | Miss % |
|---|---:|---:|
| None in last 250ms | 260/709 | 36.67% |
| 0–10ms | 198/448 | 44.20% |
| 10–30ms | 119/382 | 31.15% |
| 30–80ms | 80/151 | 52.98% |
| 80–250ms | 181/389 | 46.53% |

Not monotonic. Do not propose an arbitrary attack-window threshold
without conditioning on corpus, composition and true K.

### Source/corpus and pitch-register disparities

- GuitarSet 582/1580 FNs = 36.84%; URMP 256/499 = **51.30%**.
- Low E2–D#3: 193/358 = **53.91%** missed.
- Middle E3–D#4: 373/1020 = 36.57%.
- High E4–D#5: 225/638 = 35.27%.
- Very high E5+: 47/63 = **74.60%**.
- The pinned DEV has only **3 GuitarSet and 5 URMP independent
  compositions** represented across 17 source recordings, so 2079
  note instances are NOT 2079 independent pieces.

### Anatomy of missed notes and corrected feature ABI

- **702/838** missed attacks never showed onset head>=.60 within
  the posthoc +/-80ms diagnostic window.
- 103/838 passed onset alone but not SAME-FRAME sustain>=.50.
- 33/838 had both head thresholds in a frame but still failed
  matching/selector timing or state.
- Original first causal decision frame onset p<.15:
  669/1182 (56.60%) missed; p>=.80: 0/94 missed.
  This high correlation is unsurprising because S613 selector itself
  thresholds that SAME score; it does not explain acoustic causation.

**Matched false-negative vs successfully detected cases**:
712 of 838 missed notes could be paired 1:1 to a detected note,
at progressively looser matching levels. 314/712 were same
SOURCE + exact PITCH + active K; 126 misses had no matched control.
Pooled median FN-minus-TP original head onset = **−0.1264**.
But matched median fundamental feature H1 = **−0.0112** and
harmonic2/fundamental ratio = **−0.0107** (feature transformed
log domain); their small signs and mix should not be interpreted
as identifying a unique harmonic or volume mechanism.
Posthoc FN-vs-TP head difference can be downstream of the
exact selector gate: it is not a learned causal factor.

### False NoteOns (priority of descriptive contexts, not physical causes)

Of original 734 FP:
- 198 same pitch as an already active labeled note.
- 180 at interval 12/19/24 from active labeled pitches.
- 149 at interval 1–2 from active labeled pitches.
- 93 at other intervals while a labeled note active.
- 81 had a SAME pitch annotated onset nearby (within 250ms
  but outside matching tolerance) and are potential timing issues.
- 33 with no active annotation.

These differ from S626's **230** harmonic-*interval* FP because
S627 gives priority to possible timing mismatch (81 cases)
before intervals. Neither is a physical harmonic-source diagnosis.
Keep definitions explicit rather than treating changes in
categorization as a correction/improvement of NOTE.

### Verdict, evidence standard

S627 provides strong evidence that misses vary with **context and
crowding**, and that head onset misses account for the majority of
observed false negatives. It does **NOT** prove that the acoustic
partial itself is the cause, that close pitches require more training
examples, or that a new fusion gate would work independently.

Next: multi-factor contingency analysis (K×corpus×pitch×interval),
audio/spectrum case-control review on original stems, and
composition-disjoint perturbation/augmentation *only if validated*.


## Interaction audit — verified after full replay

[Successful source-linked interaction workflow 38890656545](https://github.com/Andriamarosoa/note/actions/runs/38890656545), using immutable audited 2,079 TP/FN events from original S627, with zero additional data downloads or changes to any model. Full report artifact `s627-context-interactions-and-conditioned-harmonic-association`.

A serious warning against interpreting one-dimensional correlations as
independent physical causes:

- Harmonic-INTERVAL relation (12/19/24 semitones) overall 212/392
  (**54.08%**) missed; this stratum contains higher K, high pitches
  and corpus-domain shifts, so comparing it to K1/no-overlap 219/639
  (**34.27%**) confounds multiple dimensions.
- On **GuitarSet**, harmonic-interval active context missed **99/238
  (41.60%)**; on **URMP**, **113/154 (73.38%)**. These are dramatically
  different despite same abstract semitone-interval label.
- When separately stratifying by corpus × true active-K × register,
  21 strata contain >=5 harmonic and >=5 comparison examples:
  harmonic contexts have *higher* missing-note rates in **12**
  strata but *lower* rates in **9**. Weighted contrast (weights =
  harmonic subset size) is **+2.83 percentage points**, and 14
  strata lack two adequate subsets and are excluded.
- Examples: GuitarSet, K3, middle E3–D#4: harmonic 15/54
  missed (27.78%) versus others 75/206 (36.41%); relative risk
  reverses sign. Conversely, URMP K3 high E4–D#5: harmonic 12/16
  (75%) versus others 20/69 (28.99%). Both observational.
- Within real K2, intervals 12/19/24 have 52/84 missed (61.90%),
  whereas K5+ harmonic intervals have 56/83 (67.47%);
  error depends on context, not simply frequency-ratio overlap.
- When a different note began 80–250ms ago and harmonic interval
  is coactive, 35/49 expected notes missed (71.43%). A near-
  simultaneous <=10ms harmonic pair yields 52/101 missed (51.49%).
  Again NOT necessarily a harmonic causal mechanism, because the
  note register, corpus and actual attack signal can differ.

**Implication:** Generic rule `harmonic interval => suppress` is
unsupported and would risk removing true played octaves.
The audio+source specific mechanism still needs spectrogram / attack
envelope / separate stem checks; no selector or data augmentation
should be justified from this cross-tab alone.

Full reproducible interaction code:
`scripts/s627_interactions_from_audit.py`.

### Limits

The contrast is **NOT** a causal adjusted effect: it is only simple
stratification with a low >=5-per-group inclusion threshold. Source
compositions, instrument attack envelopes, harmonic excitation,
absolute pitch and pitch reliability are not all controlled.
The cases within and across source clips are correlated; 21 strata
are not 21 independent experimental trials. The 2.83pp value should
not be promoted into a claim that harmonics cause 2.83pp error.
