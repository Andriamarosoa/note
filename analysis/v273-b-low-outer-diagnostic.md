# Outer diagnostic — B_low corrector 32 corrections / 28 regressions

Diagnostic only. The outer fold is **not** used here to tune a threshold or select a new model.

Source:
- internal selection: run `37297010313`
- one-shot outer evaluation: run `37316567815`
- base: `freeze_local_combo + candidate_hidden1 [42,52,61,64]`
- selected corrector: `B_low | C=0.01 | class_weight=None`

## Changed rows

B_low is applied to **756** outer rows.

- unchanged prediction: **637**
- changed prediction: **119**
- exact corrections: **32**
- exact regressions: **28**
- wrong -> wrong changes: **59**
- exact net: **+4**

Among all 119 changed predictions:
- absolute K error improves in **74** cases
- worsens in **42** cases
- unchanged in **3** cases

So the corrector has useful ordinal signal, but its exact-K gain is diluted by boundary regressions.

## Exact corrections

| true K | base -> corrected | rows |
|---:|---|---:|
| 2 | 3 -> 2 | **28** |
| 3 | 4 -> 3 | 3 |
| 4 | 3 -> 4 | 1 |

The gain is overwhelmingly a **K2 repair mechanism**: 28/32 corrections are `K2: 3 -> 2`.

## Exact regressions

| true K | base -> corrected | rows |
|---:|---|---:|
| 3 | 3 -> 2 | **18** |
| 4 | 4 -> 3 | 7 |
| 4 | 4 -> 2 | 3 |

All 28 exact regressions are downward moves.

## Transition-level verdict

| prediction transition | changed | corrections | regressions | exact net | closer to true K | farther |
|---|---:|---:|---:|---:|---:|---:|
| **3 -> 2** | 85 | 28 | 18 | **+10** | 53 | 32 |
| 4 -> 3 | 15 | 3 | 7 | **-4** | 8 | 7 |
| 4 -> 2 | 17 | 0 | 3 | **-3** | 11 | 3 |
| 2 -> 3 | 1 | 0 | 0 | 0 | 1 | 0 |
| 3 -> 4 | 1 | 1 | 0 | +1 | 1 | 0 |

The principal exact-K loss is therefore not the B_low concept itself. It is the **downward correction applied when the base already predicts K4**, plus part of the `3 -> 2` ambiguity between true K2 and true K3.

## 3 -> 2 branch composition

The 85 `3 -> 2` changes contain:

- true K2: **28** — exact corrections
- true K3: **18** — exact regressions
- true K1: 17 — remains wrong but moves closer
- true K4: 14 — remains wrong and moves farther
- true K0: 8 — remains wrong but moves closer

Thus the branch itself has **+10 exact net**, versus +4 for the full B_low corrector.

However this observation comes from outer diagnostics. It must **not** be directly promoted into a new outer rule.

## Probability separability

Within the `3 -> 2` branch, the frozen base probabilities weakly distinguish true K2 from true K3:

- AUC using `P(K2)-P(K3)`: about **0.639** for true K2 vs true K3.

This is insufficient evidence for a reliable probability threshold and is not used to tune one.

## Diagnostic conclusion

B_low contains at least two mechanisms:

1. a useful `K3 predicted -> K2` correction that repairs many true K2 cases;
2. harmful downward corrections of already-correct K3/K4 cases, especially from base K4.

The next experiment must test a **predefined family of transition guards on folds 0/1/2/4 only**. A new outer evaluation is justified only if one guard is selected robustly without using fold 3 labels.
