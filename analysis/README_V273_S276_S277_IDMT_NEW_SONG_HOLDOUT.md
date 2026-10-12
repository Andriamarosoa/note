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
