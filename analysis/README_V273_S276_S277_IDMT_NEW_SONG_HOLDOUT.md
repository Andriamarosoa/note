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
