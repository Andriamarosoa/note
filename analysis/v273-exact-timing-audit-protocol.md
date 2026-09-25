# Exact candidate timing: correction and audit protocol

Scope: the 50 tracks of frozen outer fold 3 only, from
`v273-native-paired-config.json`. No other fold is inferred or evaluated. No
training, calibration, threshold search, output corrector or score claim.

## Correction

New V91 and V100 caches use schema 2. They preserve the original integer group
origin, all pre-truncation candidate timestamps, and the retained timestamps in
model-row order. Full timestamps use flat arrays and offsets, without pickle.
The retained selection is shared with feature generation. Full candidate groups
drive annotation assignment; original origins drive spectral windows. Source-time
and pitch target consumers are updated, while candidate selection still indexes
the retained model rows. These metadata are derived from audio proposals only.

Schema 1 remains readable for reference replay. Exact and legacy shards cannot
be mixed. New cache production requires the exact metadata and rejects partial
or inconsistent timing rather than silently reconstructing missing origins.
Existing released caches, weights and the historical V27.3 score are unchanged.

An additional source-time bug was found while auditing affected code: collision
resolution compared a relative onset with an absolute origin. It now stores the
actual nearest-candidate distance. A regression test covers two same-string
onsets. Its occurrence in fold 3 is reported separately from synthetic evidence.

The default V101 pitch target function also receives the already established
global-row indexing fix from `run_v101_fixed_pitch_index.py`. A multi-member
regression test ensures that one track cannot overwrite another track’s rows.

## Real-data audit

1. Restore SHA-256-pinned proposal weights and previous fold-3 audit metadata.
   Validate the original GuitarSet ZIP checksums and source output manifest.
2. Run the frozen audio proposal stack on fold 3, one track at a time, in five disjoint ten-track jobs. Verify complete coverage
   before aggregation. Save both
   schema-2 caches to new locations; original artifacts are never overwritten.
3. Reload both formats. Compare complete timestamps, retained alignment, slots,
   pitch targets, onset-time distributions and spectral maps with runtime groups
   using actual raw record timestamps. Spectral equality is checked after the
   intended float16 storage conversion.
4. Independently assign all annotation onsets to the complete candidate groups
   and compare counts with the count targets created before truncation. Report
   same-string collisions and events outside the fixed spectral window.
5. Establish whether freshly mined candidate features and targets exactly match
   the prior cache before attaching old global row IDs to any changed examples.
   If they differ, report that explicitly; do not imply identical populations.
6. Publish the report and exact caches with hashes. A passed preparation audit
   establishes input/target consistency, not an improvement in Exact K. A new
   controlled training comparison remains necessary to measure that effect.

## Follow-up for issues found by regeneration

Run 36122007201 passed the five data-preparation audit jobs. Re-mining also
changed some floating proposal features, so equality of all historical cache
bytes is explicitly false. A separate audit reads the preserved caches, verifies
all 50 tracks, and checks retained time geometry, masks, group sizes and widths,
top anchors and K labels before mapping original row IDs. It quantifies floating
changes without calling them harmless or claiming an established numerical cause.

The follow-up independently reassigns the original annotations against complete
candidate groups and records every assigned onset outside the fixed spectral
window. This tests a remaining input coverage limitation without changing any
window size, target rule, trained model or score.
