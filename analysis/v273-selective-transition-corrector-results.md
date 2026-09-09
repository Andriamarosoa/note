# V27.3 selective transition corrector — final result

Source run: `34297767492`

Source commit: `dd43b4f92b234dc7c84377e18389fd34350cd1b4`

Reference: V27.1 `poly_rescue` from run `34270082405`.

Superseded run `34293193376` was cancelled and none of its outputs are used.

## Decision

**Promote V27.3 under the predeclared primary rule, but record the promotion as extremely narrow.**

Aggregate polyphonic exact-K increased by 2 correct rows, from 4,003 to 4,005 out of 9,401. This is a strict improvement and therefore satisfies the frozen promotion rule. It is not evidence of a robust or independently validated gain: the margin is only 0.0213 percentage point, one fold regressed sharply, and event F1 decreased.

| Metric | V27.1 | V27.3 | Delta |
|---|---:|---:|---:|
| Polyphonic exact-K | 42.5806% | 42.6019% | +0.0213 pp / +2 rows |
| Global exact-K | 81.4871% | 81.4897% | +0.0026 pp / +2 rows |
| Event F1 @ 50 ms | 79.7482% | 79.5625% | -0.1857 pp |

Event counts at 50 ms:

| Metric | V27.1 | V27.3 | Delta |
|---|---:|---:|---:|
| TP | 34,708 | 34,404 | -304 |
| FP | 8,116 | 7,859 | -257 |
| FN | 9,512 | 9,816 | +304 |

V27.3 is slightly more precise and materially less sensitive: reducing false positives did not compensate for the additional false negatives in the secondary event metric.

## Outer-fold polyphonic exact-K

| Fold | V27.1 | V27.3 | Delta | Changed | Corrected | Regressed | Net rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 42.4917% | 40.1705% | -2.3212 pp | 540 | 151 | 200 | -49 |
| 1 | 43.6374% | 44.8198% | +1.1824 pp | 340 | 116 | 95 | +21 |
| 2 | 42.1580% | 42.3939% | +0.2358 pp | 178 | 60 | 56 | +4 |
| 3 | 44.0816% | 44.4388% | +0.3571 pp | 43 | 19 | 12 | +7 |
| 4 | 40.4736% | 41.4962% | +1.0226 pp | 147 | 61 | 42 | +19 |
| **Total** | **42.5806%** | **42.6019%** | **+0.0213 pp** | **1,248** | **407** | **405** | **+2** |

Four folds improved, but fold 0 lost 49 exact rows and almost erased the other folds' combined gain of 51 rows. The rule also changed 436 rows that were wrong both before and after the override.

## Which transitions helped

| Directed transition | Activated | Corrected | Regressed | Net exact rows |
|---|---:|---:|---:|---:|
| `2 -> 3` | 177 | 50 | 53 | -3 |
| `3 -> 2` | 409 | 147 | 100 | +47 |
| `3 -> 4` | 165 | 46 | 64 | -18 |
| `4 -> 3` | 497 | 164 | 188 | -24 |

Only `3 -> 2` was positive in aggregate. Across true classes, V27.3 gained 94 correct K=2 rows and 50 correct K=3 rows but lost 142 correct K=4 rows. These outer-label observations are descriptive only; they cannot be used to retroactively prune V27.3 or claim a valid `3 -> 2`-only result.

## Independent artifact verification

The downloaded summary and five fold ZIPs matched the SHA-256 digests reported by GitHub. The raw NPZ files independently reproduced:

- exactly 76,768 unique outer rows with one-time coverage;
- exactly 9,401 true-polyphonic rows;
- the V27.1 baseline of 62,556 global and 4,003 polyphonic correct rows;
- the V27.3 result of 62,558 global and 4,005 polyphonic correct rows;
- per-fold net deltas `[-49, +21, +4, +7, +19]`;
- V27.1 event counts `34,708 / 8,116 / 9,512` and V27.3 event counts `34,404 / 7,859 / 9,816` for TP / FP / FN;
- exact equality between the concatenated fold arrays and the archived summary NPZ.

All five fold jobs, the source preflight, and the final summary job completed successfully. Historical validation and Locked12 remained unindexed and unevaluated.

## Conclusion

V27.3 becomes the formal development reference because exact-K polyphonic is the declared priority and the strict aggregate rule was met. The result should be treated as a two-row bookkeeping promotion, not a meaningful performance breakthrough.

The strongest next hypothesis is a predeclared `3 -> 2`-specific corrector or a new signal that protects K=4. Because that direction was identified after observing these outer results, any follow-up remains development work and requires a fresh nested protocol; it must not be presented as independent confirmation.
