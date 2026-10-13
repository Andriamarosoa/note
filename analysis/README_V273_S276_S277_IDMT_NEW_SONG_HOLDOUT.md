# S276/S277 — New-music prelabel cohort for S275 (IDMT-SMT-GUITAR)

**Status: selection verified; full S275 independent accuracy not yet measured.**
Source branch: codex/s276-idmt-song-disjoint-candidate-audit.
Official reference: freeze_local_combo (NOT promoted).
Source: Zenodo 7544110, pinned archive MD5 06796e08731bccffaed6ae59361486e4.

## S276 — ZIP central-directory WAV/XML FILENAMES ONLY

Successful run: https://github.com/Andriamarosoa/note/actions/runs/38318827219

| Partition | WAV | XML |
|---|---:|---:|
| Dataset 1 | 400 | 400 |
| Dataset 2 | 261 | 261 |
| Dataset 3 | 5 | 5 |

33 preliminary WAV candidates, XML member names present:

- Dataset 3 five files: pathetique_mono.wav, pathetique_poly.wav, promenade_mono.wav,
  nocturneNr2.wav, quintfall.wav. These represent AT MOST four named music
  families because Pathétique mono and poly share a composition family.
- Dataset 1 12 filenames deterministically chosen using hints of chord,
  polyphony, lick, major and minor; unique stem selection, no XML reads.
- Dataset 2 reserve 16 files from SHA filename ranks 49–261, requiring a
  uniquely named matching XML member. 213 WAV files remain after removing
  the already annotated first 48; actual composition-level novelty
  NOT independently certified.

**Caveat:** WAV filename/hash differences do not establish song/composition
disjointness, nor that two performances contain different note sequences.
Dataset 1 same-stem guitar recordings require grouping. No annotation
or WAV content was read in S276; all proposed filenames were sealed
in immutable CI artifacts.

## S277 — real native reference, source-only

Workflow: https://github.com/Andriamarosoa/note/actions/workflows/v273-s277-official-native-IDMT-dataset3.yml

S277 runs original frozen V84/V86/V87/V88 trained source pipeline and
freeze_local_combo on five Dataset 3 files. The run is explicitly limited
to the first ten seconds of each recording and excludes XML/MIDI reads.
It records true original-native V88 event start/end positions and
baseline K0–K6 predictions; it must be scored ONLY after full S275
predictions on identical native event IDs are sealed.


## Confirmed S277–S280 pre-label numerical checkpoints

Successful CI:
- S277: https://github.com/Andriamarosoa/note/actions/runs/38319155819
- S278: https://github.com/Andriamarosoa/note/actions/runs/38319390256
- S279: https://github.com/Andriamarosoa/note/actions/runs/38319535537
- S280: https://github.com/Andriamarosoa/note/actions/runs/38319717743

On first 10 seconds of each of the five Dataset3 recordings:
- **506 genuine original V88 events**; official freeze_local_combo K0..6
  prediction histogram: 457, 26, 5, 8, 10, 0, 0.
- One WAV is stereo: documented deterministic stereo->mono downmix,
  while all inputs are 44.1 kHz 16-bit PCM.
- Original S97 audio features recovered on exactly 506 baseline native IDs,
  58D summary, 42x49 sequence, 245D flow, 490D morphology, 294D physics.
- Original S5 527D input and PR16 original temporal 46D ownership
  reconstructed without model fitting.
- Four SHA-256-checked original S5 held-fold models executed: all four
  agree on 490 native events and disagree on 16, without selecting any
  original held-piece oracle.

**These counts are model predictions, not measured true Exact-K.** NO XML
contents read; no claim of composition-disjoint independent validation.


## S281 — original S8 through S18 achieved without truth access

S281 succeeded: https://github.com/Andriamarosoa/note/actions/runs/38319911630

The 133 original archived risk checkpoints were executed on the same
506 original V88 native events, with four separate folds and the
historical S8/S9/S13/S14/S15/S16/S17/S18 decoding rules. No new fit.
The original S18 folds disagree on **15 events**. This indicates
cross-fold uncertainty, not a measured Exact-K accuracy.

**Still missing for an unbiased S275 accuracy**: native Dataset3
S29/S35, all 16 real S45 producers and 193 S46 policies, apply S274
two-fold gate, original S69/S70/S71/S72/S79/S80 tail, freeze both
complete predictions BEFORE any new XML content, then verify true
composition disjointness and note-level labels. Until then:
no new independent Exact-K percentage and no official promotion.

## Strict gates before independent Exact-K and promotion

1. Complete original-source S5/S18, sixteen S45 producers, S46 quorum
   S274, and original S69 through S80 on these exact V88 events.
2. Prove the evaluated compositions are not shared with model training
   and exposed previously scored sets. The two Pathétique recordings
   must be the SAME statistical composition group.
3. Freeze the note-level annotation alignment protocol and BOTH
   prediction sets before opening new XML labels.
4. Never invent a K0 label for an unmatched onset. Complete global K0–K6
   requires exhaustive validated annotations, not merely XML onset matches.
5. Compare paired changes by K0–K6, true K2+ poly, bootstrap at composition
   level, plus regression and confidence limits against freeze_local_combo.

If any condition fails: NO independent S275 improvement claim, and NO
official promotion. The original freeze_local_combo remains in place.


## S282–S300 — Full original S80 replay and first post-frozen annotations

All remaining source-only workflows succeeded:
- [S282–S283](https://github.com/Andriamarosoa/note/actions/runs/38320164646): genuine S18 S217 >=3/4 fold consensus and original S97 audio bridge.
- [S284–S285](https://github.com/Andriamarosoa/note/actions/runs/38320247823): 38 original S29 Q models, 19 S20 parents, and all 19 original S35 branch results.
- [S286–S287](https://github.com/Andriamarosoa/note/actions/runs/38320359920): 10 original trained S45 families.
- [S288](https://github.com/Andriamarosoa/note/actions/runs/38320445824): fundamental58 validated against original 58D harmonic source.
- [S289](https://github.com/Andriamarosoa/note/actions/runs/38320553022): five historically recovered S1/S2/S3/S4/coherent families.
- [S290–S292](https://github.com/Andriamarosoa/note/actions/runs/38320668981): 16 original S45 family inputs, 304 Ridge checkpoints, 193 original S46 policies, fold neutral S219 and frozen S274 quorum2.
- [S293](https://github.com/Andriamarosoa/note/actions/runs/38320764638): 12 original trained PR16 MLP heads and 36 PR16 policies.
- [S294–S298](https://github.com/Andriamarosoa/note/actions/runs/38320939533): two COMPLETE S69/S70/S71/S72/S79/S80 source-only replays, with S291 original bank and S292 S274 bank, both locked BEFORE XML.
- [S299](https://github.com/Andriamarosoa/note/actions/runs/38321142772): post-frozen original five Dataset3 XML note matcher (locked existing S190 20ms/80ms/+30ms protocol).
- [S300](https://github.com/Andriamarosoa/note/actions/runs/38321299537): read-only per-K and grouped bootstrap audit; no re-training, recalibration or model selection.

**CRITICAL REPRODUCIBLE OBSERVATION:** On five IDMT Dataset3 WAVs,
506 original genuine V88 native events in first ten seconds of each,
the two original S80 and S274-twofold S80 final predictions are
IDENTICAL. The stricter quorum suppressed 51 of the 99 original
S46 policy–event interventions and changed six S46 events but
**zero final S80 events**, after all original learned downstream risk
and selection heads. This is unlike the small *training-overlapping*
GuitarSet S275 development gain, which cannot be transferred as a
generalization claim.

### First post-freeze observed Exact-K, annotation-MATCHED notes only

Five XML recordings yield just **47 matched real K1–K6 labels**, of
which **23 true polyphonic K2–K6**, among **506** candidate V88 events.
Exactly **459 events** remain unmatched/unknown, NOT necessarily K0.
Pathétique mono and Pathétique poly are a single composition, making
at most four named composition families in this pilot.

| TRUE note class | Events | frozen freeze_local_combo | original S80 | S274 S80 |
|---|---:|---:|---:|---:|
| K1 | 24 | 18 | 18 | 18 |
| K2 | 12 | 0 | 3 | 3 |
| K3 | 7 | 3 | 3 | 3 |
| K4 | 4 | 1 | 1 | 1 |
| K5 | 0 | n/a | n/a | n/a |
| K6 | 0 | n/a | n/a | n/a |
| **Matched K1–K6 only** | **47** | **22 (46.81%)** | **25 (53.19%)** | **25 (53.19%)** |
| **True poly K2–K6 only** | **23** | **4 (17.39%)** | **7 (30.43%)** | **7 (30.43%)** |

S80 versus freeze on these matched annotations: **4 corrections,
1 regression, net +3**. Class K2 supplies net +3 (three new correct
predictions). K3 has one correction and one regression, net zero.
K1 and K4 have no net change. K5 and K6 are absent.

The exploratory **bootstrap clustered by named composition** (four
groups, 4,096 draws, seed299) gives a 95% interval for the paired S80
gain of **[0.0, 0.1154] on matched K1–K6** and **[0.0, 0.2121]
on matched poly K2–K6**. Both include zero; four composition
groups are inadequate for a reliable promotion claim.

### Promotion verdict

- Original S80 and S274 final output are tied on this new cohort.
- The observed 7/23 is NOT comparable to the old heldfold
  GuitarSet poly 43.87% nor training-overlap S275 79.49%.
- Full Exact-K K0–K6 **was NOT measured**. Unknown/unmatched events
  cannot be assigned K0 by fiat, and song-disjoint provenance
  against all model training compositions remains unverified.
- No official reference change, no promoted S80, no PR merge.
  **freeze_local_combo remains official**.
- Next genuinely informative validation requires additional,
  provenance-audited composition-disjoint polyphonic pieces and
  a complete K0–K6 ground truth. Predetermined native alignment and
  all competing model outputs must be frozen BEFORE opening those labels.


## S301–S309 — Historical Exact-K label correction and 964 new source events

**CRITICAL LABEL CORRECTION:** Historical GuitarSet training used
`train_v90_structured_cluster_cardinality.py`:
`K = min(6, number of note onsets assigned to a 40ms V88 cluster)`.
The assignment uses the original 20ms radius, and
`yourmt3_exactk_common.assign_onsets` reproduces it from original group
start/end samples. K counts **newly attacked notes**, NOT sustained
sounding pitches.

Therefore the older S299 figures 53.19% K1–K6 and 30.43% poly
counting sounding pitches at onset+30ms under an 80ms matcher are NOT
historically equivalent Exact-K and MUST NOT be used for promotion.
S301's +1..+5 adversarial net correction bound pertains only to those
old S299 labels and is not transferable to actual attack-K.

[S303](https://github.com/Andriamarosoa/note/actions/runs/38322655543)
found 445/459 formerly unmatched events during sustained notes, but
this DOES NOT prove they are non-K0: no new attacks can occur while
an old note continues sounding. The original earlier interpretation of
S303 as a missed note target was incorrect.

[S304](https://github.com/Andriamarosoa/note/actions/runs/38323112408)
reran exact original 20ms attack grouping retrospectively on the already
exposed five Dataset3 XMLs and the original 506 V88 clusters:

| Historical attack-count metric IF XML annotations are exhaustive | Official freeze_local_combo | Reconstructed S80 |
|---|---:|---:|
| All K0–K6 native events (506) | 470/506 = 92.89% | 472/506 = 93.28% |
| Real polyphonic new attacks K2–K6 (18) | 4/18 = 22.22% | 7/18 = 38.89% |

Conditional label histogram K0=474,K1=14,K2=9,K3=8,K4=1,
K5=K6=0. Net +2 correct overall (4 fixes,2 regressions), +3 poly.
**NOT an official full global Exact-K**: the XML transcription
completeness is not independently verified, the already-open XML means
retrospective protocol, and composition disjointness is unproven.
High 92–93% overall rates are dominated by K0.

[S302](https://github.com/Andriamarosoa/note/actions/runs/38322507433)
froze TRUE original V88 `freeze_local_combo` predictions on 16 new
Dataset2 WAVs from S276 ranks 49+, disjoint from already-exposed
ranks1–48: 964 candidate native event IDs, original K0..6 histogram
[776,119,33,25,10,1,0]. Official immutable output SHA256:
`4afed533080403386ba016f409c6e8e5f43dd161fdbddd50ea31865b7585cee1`.
**No new XML opened.** The selected WAVs include the same lick
numbers performed on different guitars (Lick 1,4,7,9,12); these are
NOT necessarily distinct musical compositions.

[S305](https://github.com/Andriamarosoa/note/actions/runs/38323261561)
froze exactly the historical attack-K0..6 assignment policy BEFORE
opening any of the 16 new XML annotation files. Must not label
silence without verifying note transcription completeness.

Remaining source-only chain runs:
- [S306/307](https://github.com/Andriamarosoa/note/actions/runs/38323508991):
  all original acoustic features, 527D S5 inputs on the same 964 rows
- [S308](https://github.com/Andriamarosoa/note/actions/runs/38323592430):
  four original SHA-checked historical S5 trained fold models
- [S309](https://github.com/Andriamarosoa/note/actions/runs/38323677107):
  original 133 trained S8–S18 risk model replay

Do not infer success of S306–S309 until their CI has actually passed.
Neither complete S80 new-cohort predictions nor 16 XML labels have
yet been verified here. **No new promotion claim. Official remains
freeze_local_combo**. The next valid action is to run the complete
candidate through S80 on exactly these 964 V88 row IDs, seal all
predictions, then apply the frozen S305 historical label policy.


## S310–S329 — 964-event full historical S80 prospective test and native onset timing failure

All source-only replay stages passed:
- [S310–311](https://github.com/Andriamarosoa/note/actions/runs/38323902060): 4-fold S18 parent consensus; S97 964-row source.
- [S312–315](https://github.com/Andriamarosoa/note/actions/runs/38324201233): 19 original S29/S35 branches, ten trained S45 source families. An earlier CI assertion was corrected; no predictor fit or change.
- [S316–317](https://github.com/Andriamarosoa/note/actions/runs/38324157734): original five rare S1/S2/S3/S4/coherent trained families, authentic fundamental58 sources.
- [S318–320](https://github.com/Andriamarosoa/note/actions/runs/38324317050): **all 16** S45 families, **304** trained Ridge risks, **193** original S46 policies, source S219 and strict S274 twofold gate.
- [S321](https://github.com/Andriamarosoa/note/actions/runs/38324277092): **12** authentic pretrained PR16 neural heads / **36** original PR16 policies.
- [S322–326](https://github.com/Andriamarosoa/note/actions/runs/38324398581): two complete S69/S70/S71/S72/S79/S80 source-trained forward passes, original S46 vs S274 S46, each on EXACT SAME **964 native V88 event IDs**. **Both full S80 predictions and original freeze_local_combo SHA-sealed BEFORE accessing any reserve 16 XML**.
- [S327](https://github.com/Andriamarosoa/note/actions/runs/38324513980): first XML reads, after source SHA preflight, historical 20ms grouped note-attack K0–K6 cardinality match per S305, no fit.
- [S328](https://github.com/Andriamarosoa/note/actions/runs/38324695741): forensic XML onset coverage in audio 10-second excerpt.
- [S329](https://github.com/Andriamarosoa/note/actions/runs/38324894308): original native candidate **signed time** offset to original XML attacks, no model or threshold tuning.

### NEW independent-file, not yet composition-certified test (16 IDMT Dataset2 WAVs)

| Historical native new-attack K if annotations complete | Official freeze_local_combo | S80 reconstructed | S80 + S274 twofold |
|---|---:|---:|---:|
| All 964 events K0–K6 | 761/964 **78.94%** | 760/964 **78.84%** | 761/964 **78.94%** |
| True K2–K6 (just SIX groups) | 1/6 **16.67%** | 1/6 **16.67%** | 1/6 **16.67%** |

Attack-K histogram: **K0=924,K1=34,K2=5,K3=1,K4=K5=K6=0**.
S80 without S274: 8 corrections, 9 regressions (net −1); with
S274: 10 corrections, 10 regressions (net 0). S274 changes three
FINAL native K values relative to reconstructed S80 (two fixes, one
regression under these XML attack targets). The 95% cluster-by-lick-name
bootstrap for global improvement of original S80 versus official
includes zero, approx [−0.87,+0.84] percentage points. **NOT PROMOTED**.

Importantly XML annotations are not certified exhaustive in this
external corpus; tune-level disjointness not proven. Do not assert a
certified official global K0–K6 score from these conditional labels.

### Root-cause finding: historical 20ms native candidate vs IDMT annotations

[S328] confirms that of **300** XML note-onset events across 16 WAVs,
**50 lie outside** the first 10s portion actually fed into source
V84/V88; of the **250 that occur inside** the audio excerpt, just
**47** were assigned within the original 20ms trained target radius,
**203 missed**. Attack-onset coverage is only **47/250 = 18.8%**.
All 203 within-clip missed note onsets lie within **80ms** of at least
one original V88 candidate interval; **106 within 40ms**.

[S329] signed time audit:
- **197/203** missed XML attacks occur BEFORE their closest V88
  candidate group (**candidate time is later**, positive delay);
  just **6/203** occur after a V88 group.
- median candidate delay **+39.4104ms**;
  signed Q10=+29.33ms, Q25=+35.65ms, Q75=+44.31ms,
  Q90=+47.51ms.
- The 47 matched attacks have signed median **0ms**.

**Interpretation:** a STRONG candidate-versus-annotation **systematic
timing offset** on IDMT Dataset2, possibly original V88 source time,
the IDMT annotation onset convention, or both. The 20ms historical
training label radius thus excludes most within-clip IDMT attacks
even though a V88 proposal occurs shortly after. This is a
**fundamental domain/timing compatibility defect** that must be
understood before treating the high-K0 global score as evidence of
successful model generalization.

DO NOT fit a +/−40ms time shift to the now-exposed S327 test set,
relabel K, increase matching radius on this test after seeing XML,
or use such revised score to promote S80. The appropriate next
independent validation is to establish the timing convention from
training-only sources and/or fixed physically justified audio onset
offset, then preregister the preprocessing and rerun on separate
composition-disjoint polyphonic audio never opened.

**Final official state after S329:** `freeze_local_combo` remains
the reference; S80 and S274 cannot be promoted with these data.


### Preliminary native timestamp source check after S329

Source inspection of `scripts/train_v86_state_transition_proposals.py`:
`_predict_score_tracks` iterates raw 44.1kHz PCM chunks with
`start_sample=position`; `_candidate_groups` directly indexes
`presence[sample]` and `_records` preserves that sample index. The
original causal stream adapter `src/causal_note/v8_predictor.py`
uses `_score_chunk(outputs, start_sample, count)`, returning the
last `count` causal model outputs with exactly that contiguous input
chunk offset. **No explicit +40ms timestamp addition was found at
these adapter boundaries**.

Therefore the measured ~39.4ms mismatch is not yet proven to be
a simple indexing bug. Possible sources include causal onset
response/learned latency, different annotation onset conventions
or acoustic attack envelopes. These are competing explanations,
not validated causes. A fixed offset selected using exposed S327
XML labels is ineligible for independent S80 promotion. First
derive/verify timestamps using training-only or separate untouched
audio, then freeze and validate on NEW composition-disjoint corpus.


## S330–S356 — source debug and true frozen S80 evaluation on twelve IDMT Dataset1 chord WAVs

The previous ~39ms IDMT Dataset2 XML-to-V88 delay is NOT a pure V88
indexing offset. [S330](https://github.com/Andriamarosoa/note/actions/runs/38325866875)
compared physical RMS and FFT flux rise locations with original XML:
for 250 annotated onsets, original V88 group nearest onset had median
XML difference +37.64ms, spectral flux peak +25.40ms and RMS energy
rise +32.88ms. On 203 missed events, median V88-to-spectral-peak
time is +12.63ms and V88-to-RMS-rise is +7.10ms. Physical peaks
searched in a preregistered XML-local window are diagnostic, NOT
independent exact acoustic onsets. Source timing is causal.

[S331](https://github.com/Andriamarosoa/note/actions/runs/38326184979):
actual originally trained V84 model streaming scores match for
512/2048 stateful and 1024 fallback chunking, max onset-score absolute
difference 7.45e-9; no sample-index/chunk desynchronization.

### NEW separate prospective IDMT Dataset1 chord cohort — 477 native events

The twelve WAV filenames and paired XML **names only** were selected
in S276 before their annotation contents were opened. S332 official
reference V88 SHA and first 477 candidate groups sealed with *no XML*,
and [S333](https://github.com/Andriamarosoa/note/actions/runs/38326698570)
fixed a second model-neutral physical-onset scoring protocol with
unknown attack abstention BEFORE any XML.

The trained stack was independently reconstructed on exactly 477
real native V88 groups: S334/335 S97/527D, S336/337 four original
S5 +133 S18 heads, S338/339 4fold source, S340–343 19 S29/S35
branches and 10 S45 families, S344 true fundamental58 and five
historically trained rare S45 heads, S345–347 16 S45, 304 original
Ridge risk and 193 S46 policies and S274 twofold counterfactual,
S348 12 original pretrained neural / 36 PR16 policies. Final
S349–S353 original saved S69→S80 trained checkpoint inference
successfully sealed on
[S355](https://github.com/Andriamarosoa/note/actions/runs/38327686633)
**before first Dataset1 XML was opened**.

The two original-trained S80 branches (S274 gate versus original S46)
are bit-identical on all 477 events; 28 K changed vs official.
These are **new external fold-neutral policies**, NOT identical to the
historical held-piece oracle deployment, and neither is refit using XML.

### True pre-frozen paired comparison [S354](https://github.com/Andriamarosoa/note/actions/runs/38327717118)

The first genuine read of all 12 XMLs was AFTER SHA seals. On the
**original historical 20ms note-attack assignment**; conditional on
the XML enumerating all original attacks:

| Condition | frozen official `freeze_local_combo` | original-trained S80, 3/4 fold-neutral |
|---|---:|---:|
| K0–K6 all 477 native V88 groups | 442/477 **92.66%** | **455/477 95.39%** |
| K1+ positive attack rows (31) | 7/31 **22.58%** | **15/31 48.39%** |
| K2+ polyphonic attack rows (20) | 6/20 **30.00%** | **11/20 55.00%** |

All-row corrections **17**, regressions **4**, still-wrong changed
decisions **7**. Net **+13**. Poly correct +5 (7 corrections, 2
regressions). Baseline/K fractions are contingent on complete
transcribed XML and the historical timing convention.
The alternate frozen S333 physical peak anchor method was **NOT
evaluable on any complete WAV**: 14 of 65 onsets anchored, 51
unanchored, zero wholly certifiable native rows. Do not claim it
achieved higher S80 accuracy.

### Strength and limits of gain [S356](https://github.com/Andriamarosoa/note/actions/runs/38328133819)

- Net correct differences by the twelve WAV file hashes:
  **[+1, +2, −1, 0, +1, +4, +1, +1, 0, 0, +2, +2]**.
  8 WAV positive, 1 negative, 3 neutral.
- Exact paired event-level McNemar p **0.00720** (17 vs 4 discordants).
- Twelve-WAV signflip two-sided p **0.0234375** global; **0.0625**
  K2+, with only 20 positive poly rows.
- Cluster bootstrap by WAV for all-row improvement has 95% CI
  **[+1.05, +4.43] percentage points**; for K2+ 95% CI
  **[+11.11,+36.36]** points but K2+ sample is extremely sparse
  and WAV-by-composition independence is NOT established. These
  are NOT music-composition-clustered confidence intervals.
- **Source of uncertainty**: the 12 preselected WAVs are not yet
  proved independent compositions, and even non-exhaustive XML
  onsets create potentially spurious negative K0 truth (446 of 477
  groups are labelled K0). Original training was GuitarSet; source
  split exposure and musical-family duplicates must be documented.
- The independent external Dataset2 first attempt (16 WAV, 964
  groups) found **S80 760/964 vs official 761/964**; S274
  was 761/964. Thus improvement does NOT transfer unconditionally
  to another available IDMT subcorpus with different onset timing.

**Official promotion verdict after S356: PENDING, NOT AUTHORIZED.**
S80 is a **measured superior challenger for Dataset1 on the sealed
historical conditional 20ms attack benchmark**, not yet a replacement
global baseline. Promote only after independent, composition-disjoint
polyphonic data with verified complete annotations and relevant
metrics confirm the improvement. Do not rename `freeze_local_combo`
or erase historical reference while such evidence is missing.


## S357–S367 — NEW prospectively held-out 24 chord files and specific K3 failure located

### Prospective 24 new RAW WAVs, unchanged true original source

- [S357](https://github.com/Andriamarosoa/note/actions/runs/38330431853):
  selected exactly 24 WAV + paired XML **NAMES ONLY** using SHA order
  from 32 remaining distinct filename-family groups; excludes all
  previously exposed first 12 canonical chord families.
- [S358](https://github.com/Andriamarosoa/note/actions/runs/38330563316):
  true frozen original V84→V88 `freeze_local_combo` predictions on
  **967 authentic original native candidate event IDs** on 24 WAVs,
  no XML read. Baseline histogram [882,27,39,17,2,0,0].
  Original reference archived SHA
  `2d4244b2c13ecb96d17e6ad0ba9a86a90caa4448b59738447f0ba2c0ccfa3559`.
- [S359](https://github.com/Andriamarosoa/note/actions/runs/38330878525):
  deterministic SHA-audited source-only rebase of 20 original-trained
  S334→S353 source modules, changing only the number of native rows,
  WAV inventory and source model input pointers. This is NOT original
  historical held-piece-oracle identity. No network fit, no new audio
  label used. All 20 stage modules compile.
- [S360](https://github.com/Andriamarosoa/note/actions/runs/38330965820):
  original 58D/527D / PR16 46D audio source views on same 967 rows.
- [S361](https://github.com/Andriamarosoa/note/actions/runs/38331004943):
  four genuine pretrained original S5 and all 133 original S8–S18
  risk checkpoints applied to same original V88 rows.
- [S362](https://github.com/Andriamarosoa/note/actions/runs/38331203079):
  all six jobs executed to S80 on 967 native events; *the last run
  status was failed solely due stale CI assertion expecting 12
  WAVs even though source actually had 24 WAVs*, AFTER all predictions
  and SHA-256 were uploaded as artifacts.
- [S365](https://github.com/Andriamarosoa/note/actions/runs/38331662519):
  read-only **SUCCESSFUL** independent recovery/validation of both
  complete prelabel S80 output SHAs from S362, confirms correct 24
  WAV count and 967 unchanged native event IDs. No new model fitting
  or inferred score. S80 both variants bit-identical across all 967,
  each changes **32** incumbent predictions.

### S363 — first XML opening after actual original 967-native SHA seals

[Run S363 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38331684749).
Historical GuitarSet K0–K6 target counts NEW note attacks assigned
within 20ms to an unchanged original V88 candidate group. Results
conditioned on complete XML attack inventory and its existing
annotation onset convention:

| 24 new WAV chord group metric | Original official freeze_local_combo | Reconstructed source-trained S80 |
|---|---:|---:|
| All native original 967 event groups K0–K6 | 899/967 = 92.97% | 913/967 = **94.42%** |
| Real poly K2–K6 (44 groups) | 23/44 = 52.27% | 24/44 = 54.55% |

Twenty-one corrections, seven regressions, net **+14 global**; on
the 44 true poly groups, eight corrections and seven regressions,
net **+1**. S274 strict fold S46 produces same final S80 on all
967 rows. Original truth histogram: K0=895,K1=28,K2=28,K3=15,K4=1,
K5=K6=0. Acoustic alternate S333 was fail-closed on all 24 WAVs
(40/133 XML attacks physically anchored, 93 unanchored, ZERO
whole-WAV certifiable), so NO alternative unbiased acoustic-K score
was measured.

### S364 and S366 — original independently pre-scored two-cohort paired results

[S364 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38331705403):
24-WAV global paired 95% clustered-by-WAV bootstrap +0.32 to +2.49
percentage points and exact WAV signflip p=0.027; poly gain interval
−15.56 to +18.92 points including zero; poly signflip p=1.0.

[S366 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38332075660):
two original independently SHA prelabel-frozen *cohorts*, 12+24 = 36
distinct filename-hash WAVs and **1444** original native events:

| True note-ATTACK K, XML completeness conditional | Reference correct | S80 correct | Net |
|---|---:|---:|---:|
| **K0** (1341) | 1307 | 1323 | **+16** |
| **K1** (39) | 5 | 10 | +5 |
| **K2** (40) | 20 | **33** | **+13** |
| **K3** (18) | **8** | 2 | **−6** |
| **K4** (5) | 1 | 0 | −1 |
| K5 (1) | 0 | 0 | 0 |
| K6 (0) | n/a | n/a | n/a |
| **Full native events** 1444 | 1341/1444 = 92.87% | 1368/1444 = **94.74%** | **+27** |
| Real attacks K1–K6, 103 | 34/103 | 45/103 | +11 |
| Real poly attacks K2–K6, 64 | 29/64 = 45.31% | 35/64 = 54.69% | +6 |

The global paired event McNemar p=0.000142 and conditional
36-WAV exact signflip p=0.000811, but WAV filename-group independence
and K0 XML-negative exhaustiveness are NOT verified. Poly event
McNemar p=0.307 and conditional WAV signflip p=0.266:
**poly gain cannot be claimed robust**. Across dataset2 previously
S80 was −1 net; NOT universally superior.

### S367 — Mechanistic K3 cause located before S80 final tail

[S367 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38332268955):
- **ALL SIX** K3 regressions (reference K3 correct, final S80 K2
  incorrect) are ALREADY K2 in real original saved **S72** source.
- None were K3 at S72 nor at S79. **ZERO** K3→K2 transition
  occurred in the final S79→S80 tail. Thus the final S147 tail
  is not the cause.
- The six K3 failures are present ONLY in the 24-WAV follow-up
  (none in the previous 12); 15 real K2 fixes total across cohorts.
- Source inspection reveals `scripts/s351_S71_S72_dataset1_477.py`
  contains `veto71poly=veto71&~((parent==3)&(s70==2))`:
  **it explicitly disables veto on K3→K2 transitions** in the
  source poly guard. This is a testable design risk, not proof that
  every true K3 regression triggers the source exception.
- [S368](https://github.com/Andriamarosoa/note/actions/workflows/v273-s368-original-S71-K3-veto-exception-AUDIT.yml)
  replays actual trained S69→S72 and audits six regressed rows to
  test whether this K3→K2 veto exception is responsible.
- No postlabel threshold or routing rule should be promoted based
  on these already exposed WAVs; any fix needs a separately
  preregistered, truly untouched polyphonic test set preserving K2
  and eliminating K3 regressions.

**Promotion verdict: reference freeze_local_combo remains official.**
S80 has significant conditional gains on K0/K2 in these 36 chord WAVs
but **K3 regresses sharply** (8/18→2/18), K4 unsupported and XML K0
negative completeness is not fully established. Do not claim better
polyphonic universal Exact-K or promote the official base.


## S368–S370 — disproven S71 hypothesis; K3 collapse begins in ORIGINAL trained S5 and S8

[S368 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38332540172)
replayed real original pretrained S69→S72 stage output **bit-identical** with the S352 PRE-label frozen complete S80 archive, before any already-exposed XML truth was joined. For ALL SIX true K3→K2 regressions:
- S69 already **K2**; S70, S71, S72, S79, final S80 did not change it.
- **ZERO** were `parent=K3 -> S70=K2` so `veto71poly=veto71 & ~((parent==3)&(s70==2))` did NOT cause one single regressed case. This S71 exception was a plausible but **FALSE** source hypothesis for the six errors. Do NOT patch it based on this audit.
- S69 raw source statistics among 15 true K3 new24: **13 already K2, 2 K3**.

[S369 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38332912521)
replayed and independently matched **original** S18 fourfold parent→all 193 S46 specialists→S69. **All six K3-to-K2 regression event rows first undercount at the S18 parent**, NOT S46/S69.
On these six events, real original S18 heldfold K votes:
- **five** samples: [K2,K2,K2,K2], unanimous wrong
- **one** sample: [K2,K2,K1,K2], 3/4 majority wrong
- **zero** votes predict K3. A S46/69 repair cannot emerge from the existing S18 parent without extra independent K3-discriminative information.

[S370 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38333086532)
then used the authentic SHA-prelabel S336 saved **four S5** original trained models and S337 original **133** S8–S18 risk checkpoints:
- **4/6** erroneous K3 cases already have >=3 votes K2 at **S5**;
- the remaining **2/6** acquire >=3 votes K2 at **S8**;
- NO subsequent S9/S13/S14/S15/S16/S17/S18 stage recovers the true K3 decision.
- Across the **15 true K3** new24 events, only **2/15**
  have >=3 fourfold K3 votes at original **S5** and only 2/15 at original S8, through S18.
- Conversely, across the **28 true K2** new24 events, >=3
  trained folds correctly vote K2 only **4/28** at original S5,
  rapidly increasing to **14/28** immediately at S8.
- This is a demonstrable K2-recovery vs K3-under-count **architecture tradeoff within original S5/S8 heads**, rather than failure of late S46/S80 gating.
- Original S5 train protocol `analysis/README_V273_PROTOCOLE_CLASSES_RARES.md` used the same owned527 features and 4 heldfold independent HGB classifiers with sqrt inverse class-frequency weighting capped at 8. Earlier rare-class weighting does not guarantee calibrated K3-vs-K2 discrimination on new audio.
- It is **not justified** to overwrite frozen S8 K2 policy by a hardcoded K3 rescue fitted to these exposed XML cohorts; this would exploit labels and invalidate the independent validation.

**Next design experiment, NOT yet externally proven**:
build an EARLY K2-vs-K3 specialist using novel independent acoustic/harmonic/attack morphology features and original training-only held-piece folds. Keep the existing S8 K2 improvements; use a separately trained/validated gate to issue K3 only where independent evidence supports a third new fundamental/attack. Demonstrate audit by true K and per-composition corrections/regressions before first opening a DIFFERENT polyphonic external cohort. Reuse NONE of the S354/S363 XMLs to train, select thresholds, or claim improvement.
**No official promotion**. Frozen baseline `freeze_local_combo` remains until strict K2–K4 generalization and independent composition identity are confirmed.
