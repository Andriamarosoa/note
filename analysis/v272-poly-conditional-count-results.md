# V27.2 conditional polyphonic count — final result

Source run: `34287254337`

Source commit: `d11f73079a9eb12b2ebce3c8d1df8209b0bd6bdf`

Reference: V27.1 `poly_rescue` from run `34270082405`.

## Decision

**Do not promote V27.2. Keep V27.1 as the reference.**

The predeclared primary metric, aggregate polyphonic exact-K, regressed.

| Metric | V27.1 | V27.2 | Delta |
|---|---:|---:|---:|
| Polyphonic exact-K | 42.5806% | 41.0488% | -1.5318 pp |
| Global exact-K | 81.4871% | 81.2995% | -0.1876 pp |
| Event F1 @ 50 ms | 79.7482% | 79.0087% | -0.7395 pp |

Event counts at 50 ms:

| Metric | V27.1 | V27.2 |
|---|---:|---:|
| TP | 34,708 | 33,873 |
| FP | 8,116 | 7,652 |
| FN | 9,512 | 10,347 |

## Outer-fold polyphonic exact-K

| Fold | V27.1 | V27.2 | Delta |
|---|---:|---:|---:|
| 0 | 42.4917% | 38.7968% | -3.6949 pp |
| 1 | 43.6374% | 44.7635% | +1.1261 pp |
| 2 | 42.1580% | 40.2123% | -1.9458 pp |
| 3 | 44.0816% | 41.5816% | -2.5000 pp |
| 4 | 40.4736% | 40.2583% | -0.2153 pp |

Only fold 1 improved. Folds 0, 2, 3 and 4 regressed.

## What happened

V27.2 trained a conditional `K=2..6` count head only on truly polyphonic training rows and replaced V27.1 counts only where V27.1 already predicted `K>=2`.

The selected arm was `uniform` on all five outer folds. The rule changed many rows, but aggregate regressions exceeded corrections. Fold 0 is the clearest failure: 206 changed rows became correct, but 284 previously correct rows regressed, for a net loss of 78 exact rows. Fold 1 was the only positive fold, with 207 corrections versus 187 regressions.

The conditional head also collapsed too strongly toward lower polyphonic classes on some folds. On fold 0, for example, true `K=4`, `K=5`, and `K=6` accuracy fell sharply while `K=3` improved. This shows that simply conditioning training on `K>=2` does not solve the within-polyphony classification problem.

## Conclusion

V27.2 is a negative result. It demonstrates that a wholesale replacement of V27.1's already-polyphonic decisions with a separate conditional head is too aggressive.

The next experiment should preserve V27.1 by default and only override a polyphonic count when a new model can identify a **specific likely misclassification transition** (for example `2↔3` or `3↔4`) with inner-only calibrated evidence. A selective transition corrector is preferable to replacing all `K>=2` predictions.
