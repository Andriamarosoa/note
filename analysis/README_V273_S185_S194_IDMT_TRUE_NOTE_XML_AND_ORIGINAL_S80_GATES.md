# S185–S194 — IDMT real audio, frozen note-level XML Exact-K, and S80 promotion gate

Evidence branch: `codex/v273-scope-rescue`. **Original S80 is NOT promoted.** The official `freeze_local_combo` reference is unchanged.

## Three original obligations and current evidence

| Required step | Verified work | Still missing for *original historical S80* |
|---|---|---|
| 1. Recover 255 original combinations, nine local audit features, trained source risks, S5 527D | Historical S1–S4 checkpoint replay (S174, zero mismatches), recovered coherent checkpoints (S175), S179 constructed a separate 255-group *adaptation* | The original S45 eight producer votes for each event, correct original 255 joint proposals, original 9-dimensional per-group audit and original logit/parent routing; S5 527D constructor still not proved |
| 2. Complete sixteen S45 and 193 S46 | Eleven original trained source families run with explicit GOAT adaptations (S173); two genuine-source S46 subsets yielded 96 policies (S180), zero independent replay mismatches (S181) | Five additional S45 families and full 16-family vote contract; therefore cannot establish 193-policy original S46 or downstream full S80 |
| 3. Run new polyphonic WAV and independent labels | S183, S185, S189, S190, S192, S193, S194 executed with audio-label separation | Original S80 plus actual `freeze_local_combo` must be run on the **same new audio events** and scored with truthful labels before any promotion |

## External dataset source

IDMT-SMT-GUITAR_V2, Zenodo 7544110, pinned manifest MD5 `06796e08731bccffaed6ae59361486e4`. HTTP 206 ZIP byte-range access avoids full 1.33 GB transfer. Individual ZIP WAV/annotation CRC and SHA are recorded; the complete 1.33 GB MD5 is **not recalculated** during range-only extraction.

S185 reused identical frozen trained S35 acoustic checkpoints and the adapted S179 255 group formula on ten second (nonoverlapping) dataset4 WAVs. The source annotations in dataset4 are chord names, not the full set of simultaneously active MIDI pitches. Therefore S187 correctly refused to assign precise K from `Am7`, `G5`, `NC`, etc.

S188 identified dataset2: 261 WAVs and 261 matched note-event XML annotation documents. On the new benchmark all model predictions, acoustic onsets, selection, and label-matching tolerance were written into **successful immutable S189/S192 CI artifacts BEFORE any XML was opened in separate later S190/S193 CI jobs**.

No pretrained checkpoint was modified or trained using the XML labels. Matching protocol: nearest note event onset no farther than 80 ms, no duplicate onset matching, note onsets within 20 ms grouped, count distinct MIDI pitches active at reference onset + 30 ms. Unmatched detections are excluded, not falsely treated as K0. This means **results apply only to events with valid matched annotation**, not to all acoustic onsets.

## Verified numeric results

First dataset2 WAV cohort, filename-hash ranks 1–16, run S189+S190: 16 WAVs, 128 audio-only candidate events, 62 matched to XML (48.44% coverage), 11 truly polyphonic. The adapted S35/S179 parent correctly predicted 43/62 = **69.35% global**, and 0/11 = **0% polyphonic**. Original pretrained H7 attack morphology produced 53/62 = **85.48% global**, 4/11 = **36.36% polyphonic**. H2 poly-only head produced 8/11 = 72.73% poly correct, but by its training support it cannot predict K0/K1 and hence is not a stand-alone global-class model.

**H7 as candidate was identified after the first cohort's true labels were exposed**; these first-cohort H7 figures must be treated as development evidence, not independent prospective confirmation.

Second dataset2 WAV cohort, different filename-hash ranks 17–32, run S192+S193: 16 additional WAVs, 128 audio-only events, 65 matched to XML (50.78% coverage), 14 truly polyphonic. The prior trained checkpoints, fixed original H7 and unchanged adapted source are all frozen before opening this cohort's XML. Adapted parent: 41/65 = **63.08% global**, 0/14 = **0% polyphonic**. H7: 45/65 = **69.23% global**, 1/14 = **7.14% polyphonic**. Thus H7 adds +4 correct overall and +1 poly correction on **these previously unused WAVs**, with no retraining. It does *not* solve poly Exact-K.

A read-only S194 audit checked **zero overlapping WAV identifiers** between the two cohorts. This is **WAV-disjoint**, NOT proven musical-piece/song-disjoint; different playing techniques or recordings of the same underlying composition can occur in the dataset. Therefore avoid claiming full song-disjoint independent validation.

Pooled for descriptive context only (not a substitute for the prospective second cohort): 127 matched audio events, 25 truly K2+; adapted parent 84/127 = **66.14% global**, 0/25 = **0% poly**. H7 98/127 = **77.17% global**, 5/25 = **20.00% poly**. The first cohort informed the H7 choice, so the pooled result is *not* an unbiased validation of that selection.

## GitHub Actions evidence

- S174 — restored S1–S4 original checkpoint historical replay: https://github.com/Andriamarosoa/note/actions/runs/38206823027
- S175 — restored four original coherent neural checkpoints: https://github.com/Andriamarosoa/note/actions/runs/38206970870
- S180 — two available S46 partial family groups and 96 source policies: https://github.com/Andriamarosoa/note/actions/runs/38208449350
- S181 — independent S46 replay, 0/96 policy deviations: https://github.com/Andriamarosoa/note/actions/runs/38208554339
- S183 — original pretrained S35 source heads on 100 unseen dataset4 events: https://github.com/Andriamarosoa/note/actions/runs/38208926118
- S185 — second ten dataset4 WAV source freeze: https://github.com/Andriamarosoa/note/actions/runs/38209386300
- S187 — actual source annotation chord-CSV format: https://github.com/Andriamarosoa/note/actions/runs/38210187927
- S188 — dataset2 XML/WAV pairing inventory, no label reads: https://github.com/Andriamarosoa/note/actions/runs/38210287227
- S189 → S190 — independent WAV audio predictions frozen **before** original XML scored: https://github.com/Andriamarosoa/note/actions/runs/38210431886 and https://github.com/Andriamarosoa/note/actions/runs/38210607635
- S191 — seven heads class-support and postfreeze individual Exact-K audit: https://github.com/Andriamarosoa/note/actions/runs/38210869411
- S192 → S193 — third WAV cohort second-stage truly frozen prior H7 comparison: https://github.com/Andriamarosoa/note/actions/runs/38211006832 and https://github.com/Andriamarosoa/note/actions/runs/38211069503
- S194 — two cohort comparison and WAV identity nonoverlap verification: https://github.com/Andriamarosoa/note/actions/runs/38211317775

## Promotion decision and strict next tasks

1. Restore/prove the **historical**, not S179-adapted, original 255 producer inputs and nine local audits, 527D S5 source and required trained logit source identities. Do not fill missing original columns with zeros or replace them with arbitrary H7 projections while claiming equivalence.
2. Once all sixteen original S45 family decisions and risk vectors exist on the **same new WAV events**, run and verify the original 193 S46 policies, plus S69/S72/S79/PR16 and actual S80 tail, without fold leakage.
3. Freeze both the fully original S80 and the actual `freeze_local_combo` source predictions **before** evaluating new valid song-disjoint exact note-event labels. Report K0–K6 confusion, poly Exact-K and corrections/regressions.
4. Only then promote S80 if a statistically defensible independent paired net win over official reference is verified; otherwise retain `freeze_local_combo`.

**Current verdict: S80 not promoted, full original steps 1 and 2 incomplete. New label-backed audio testing is operational for an explicitly separate S35/S179 experimental adaptation; its poly gate remains ineffective, despite a small independently confirmed H7 benefit.**
