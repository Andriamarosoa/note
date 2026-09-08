# V27.1 — poly rescue results

Source run: `34270082405`  
Source branch: `v271-poly-rescue`  
Source commit: `aed78122c5d85079c84b597e78331ba080f8b4b3`

The complete nested run succeeded: protocol/source preflight, all five untouched outer folds, and the final frozen outer summary.

## Frozen decision

The predeclared primary metric is polyphonic exact-K (`K >= 2`). V27.1 is promoted iff aggregate poly exact-K is strictly greater than the V27 `low_k_fusion` reference. Global exact-K and event F1 remain secondary diagnostics and are not vetoes.

| Metric | V27 `low_k_fusion` | V27.1 `poly_rescue` | Delta |
|---|---:|---:|---:|
| Polyphonic exact-K | 37.5705% | **42.5806%** | **+5.0101 pp** |
| Global exact-K | 82.4953% | 81.4871% | -1.0082 pp |
| Event F1 @ 50 ms | 80.1896% | 79.7482% | -0.4414 pp |
| TP | 33,409 | 34,708 | +1,299 |
| FP | 5,696 | 8,116 | +2,420 |
| FN | 10,811 | 9,512 | -1,299 |

Final workflow decision: **`promote: true`**.

## Per-true-K accuracy after V27.1

There are 9,401 true-polyphonic rows.

| True K | Rows | Exact-K | Under-count | Over-count |
|---:|---:|---:|---:|---:|
| 2 | 4,279 | **49.29%** | 27.32% | 23.39% |
| 3 | 2,952 | **37.43%** | 40.96% | 21.61% |
| 4 | 1,628 | **42.51%** | 51.41% | 6.08% |
| 5 | 438 | **19.86%** | 76.94% | 3.20% |
| 6 | 104 | **9.62%** | 90.38% | 0.00% |
| all K>=2 | 9,401 | **42.58%** | 38.78% | 18.64% |

V27.1 mainly improves K=2. K=4..6 remain dominated by under-counting.

## Polyphonic confusion matrix

Rows are true K; columns are V27.1 predicted K.

| true \\ pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 686 | 483 | **2,109** | 785 | 205 | 10 | 1 |
| 3 | 211 | 65 | 933 | **1,105** | 578 | 49 | 11 |
| 4 | 45 | 9 | 249 | 534 | **692** | 87 | 12 |
| 5 | 6 | 1 | 48 | 114 | 168 | **87** | 14 |
| 6 | 1 | 0 | 6 | 20 | 37 | 30 | **10** |

Largest remaining error transitions:

1. `3 -> 2`: 933 rows
2. `2 -> 3`: 785
3. `2 -> 0`: 686
4. `3 -> 4`: 578
5. `4 -> 3`: 534
6. `2 -> 1`: 483

This shows that the remaining problem is no longer only low-K rescue. A large fraction is now discrimination *inside* the polyphonic classes, especially 2 vs 3 vs 4.

## Existing-expert oracle ceiling

Using the frozen row-level predictions from V10.4, V26-uniform and V26-weighted, an oracle that is allowed to pick the correct expert independently for every row reaches only **56.0047% polyphonic exact-K**.

Among the 5,398 V27.1 polyphonic mistakes, at least one of those three existing experts is correct on only 1,262 rows (23.38%). Therefore no router, confidence threshold, or post-hoc selector over the current experts can approach 80% poly exact-K.

Numerically:

- current V27.1 correct poly rows: 4,003 / 9,401;
- oracle over all existing count experts: 5,265 / 9,401 = 56.00%;
- 80% target requires at least 7,521 correct rows;
- even after perfect routing of existing experts, at least **2,256 additional poly rows** would still require genuinely new predictive information/model capacity.

## Consequence

V27.1 is the new reference because it satisfies the frozen primary promotion rule, but it also closes the path of repeated threshold/routing variants over the same experts.

The next experiment must introduce a new count signal. The recommended direction is a hierarchical polyphonic count model that separates:

1. `K <= 1` versus `K >= 2` detection; and
2. conditional exact count `K in {2,3,4,5,6}` on true-polyphonic training rows.

The purpose is to stop asking one seven-way head dominated by K=0/1 examples to solve both presence and fine polyphonic cardinality simultaneously.