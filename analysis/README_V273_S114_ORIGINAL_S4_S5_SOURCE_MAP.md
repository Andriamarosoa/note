# Original GitHub source map — S4/S5 upstream of S80 (11 October 2026)

**Status:** code/artifact provenance recovered, but no independent external S80
validation. Original reference remains `freeze_local_combo`.

## Original repositories, branches and workflow

- Original research branch:
  [`codex/v273-open-k0-k6`](https://github.com/Andriamarosoa/note/tree/codex/v273-open-k0-k6).
- Original acoustic feature extractor:
  [`scripts/extract_v273_open_cohort.py`](https://github.com/Andriamarosoa/note/blob/codex/v273-open-k0-k6/scripts/extract_v273_open_cohort.py).
  It calculates the **58D summary**, the **42×49 harmonic trajectory** and the
  **46D native event geometry**, via the original `ownership` code.
- Original workflow:
  [`v273-open-k0-k6.yml`](https://github.com/Andriamarosoa/note/blob/codex/v273-open-k0-k6/.github/workflows/v273-open-k0-k6.yml),
  verified successful historical run
  [37837235723](https://github.com/Andriamarosoa/note/actions/runs/37837235723),
  and prerelease `v273-open-k0-k6-37837235723`. The workflow covers
  initial audio extraction and 36 pretrained neural policies; it does not
  itself implement the subsequent S4/S5 HistGradientBoosting models.

## Crucial S4 `owned` → S5 contract

- [Original S4 protocol](https://github.com/Andriamarosoa/note/blob/codex/v273-open-k0-k6/analysis/README_V273_PROTOCOLE_ATTAQUES_ATTRIBUEES.md):
  S4 `control` = **58D harmonic summary + 46D geometry + 7D frozen K**
  = **111D**. `owned` augments control with labelled-independent per-pitch
  onset/flow/ownership statistics, at fixed temporal shifts 1,2,4,8
  frames and temporal ownership zones; total **527D**.
- [Original S5 rare-class protocol](https://github.com/Andriamarosoa/note/blob/codex/v273-open-k0-k6/analysis/README_V273_PROTOCOLE_CLASSES_RARES.md):
  explicitly **reuses the exact S4 `owned` 527D design** and changes
  training class weighting only. Original S5 per-fold checkpoints
  carry the same `designs_sha256` hash.
- [Evidence preservation commit](https://github.com/Andriamarosoa/note/commit/7165e1ef54510aa9effc9f597b2783c832ca907a):
  preserves 14 S4 and four S5 weight files and archived predictions.
  It does **not** add a separately versioned S4 `owned` feature-constructor script.

## GitHub archaeology results

- [S114 exhaustive old-refs workflow](https://github.com/Andriamarosoa/note/actions/runs/38183957384)
  completed SUCCESS and searched 339 branch/tag refs.
- [S114 branch-and-relevant-tags scan](https://github.com/Andriamarosoa/note/actions/runs/38184064063)
  completed SUCCESS and searched 94 selected refs. Both examined
  versioned `scripts`, workflows, tests, and analysis history.
- The exact code that *constructs all 416 extra S4 `owned` dimensions in
  their original order* was **not identified in the inspected versioned
  source**. The original protocol describes the statistical families,
  but does not uniquely specify executable numeric code.
- GitHub Actions run discovery on the S4/S5 development window
  2026-10-08 20:00–22:20 UTC found the original PR16 run and later
  S6/S7/S9 workflows, but no separately named S4/S5 training Action.
  Checkpoints may have been trained locally and committed afterward;
  this remains a *possibility*, not a verified training provenance.

## External continuation

- S97/S106: 57 GOAT audio events, 58D summaries and 42×49 real
  time-harmonic trajectories recovered without MIDI.
- [S116](https://github.com/Andriamarosoa/note/actions/runs/38184381730):
  **SUCCESS**. The original 46D ownership function was restored from the
  historical open-K0-K6 branch, combined with acoustic summary58 and
  original frozen one-hot K. All 57 GOAT events yield finite **111D**
  control features. No incompatible intervals or MIDI labels.
- [S117](https://github.com/Andriamarosoa/note/actions/runs/38184550844):
  **SUCCESS**. Eight genuine pinned S4 `control` checkpoints (4 folds × 2
  poly weights) executed their 111D forward on all 57 GOAT events. Mean
  fold posterior argmax histograms, K0..K6:
  poly1 = [47,9,1,0,0,0,0],
  poly2 = [47,8,2,0,0,0,0].
  Changed vs frozen reference: 5 and 4 events. These are not Exact-K
  accuracy measures (GOAT annotations were deliberately not read).
  The assumed column order still requires original-fold replay to prove
  numeric identity; and S4 `owned` 527D and full S80 remain blocked.
- The next required gate is an exact 527D `owned` reconstruction and
  historical checkpoint replay; only then should S45/S46/S80 undergo
  real external inference and an independent Exact-K comparison.

**Never promote S80 on a historical cached-vector replay or presumed
feature equivalence.** Validate its complete external pipeline first.
