# V27.3 outer diagnostic — robust candidate_hidden1 group [42,52,61,64]

Diagnostic only. **No outer tuning, no new selection, no promotion decision is made from this analysis.**

Sources:
- internal group selection: run `37285354716`
- one-shot outer evaluation: run `37290794473`
- reference: `freeze_local_combo` from run `37233793945`
- Cluster A membership: failure clustering run `36999901054`

## Headline

The internally selected group improves the outer fold from **83.441% to 83.533% exact global**
and from **39.614% to 39.817% poly exact**.

Paired against `freeze_local_combo`:
- corrections: **58**
- regressions: **44**
- net exact: **+14**
- changed predictions: **138**
- unchanged predictions: **15141**

Error totals:
- under-count: **1429 -> 1425** (**-4**)
- over-count: **1101 -> 1091** (**-10**)
- exact: **12749 -> 12763** (**+14**)

Therefore the gain is not a simple upward/downward cardinality shift: both under- and over-counting decrease.

## Exact gain/loss by true K

| true K | net exact |
|---:|---:|
| K0 | -9 |
| K1 | +19 |
| K2 | +2 |
| K3 | -5 |
| K4 | +6 |
| K5 | +1 |
| K6 | 0 |

Main beneficial exact transitions:
- true K1: `2 -> 1` x18, `0 -> 1` x6, `3 -> 1` x2
- true K2: `3 -> 2` x10, `0 -> 2` x2
- true K3: `2 -> 3` x4, `4 -> 3` x1
- true K4: `3 -> 4` x8
- true K5: `4 -> 5` x1
- true K0: `1 -> 0` x4, `2 -> 0` x2

Main regressions:
- true K0: `0 -> 1` x11, `0 -> 2` x4
- true K1: `1 -> 2` x5, `1 -> 0` x2
- true K2: `2 -> 1` x7, `2 -> 3` x2, `2 -> 0` x1
- true K3: `3 -> 2` x5, `3 -> 4` x5
- true K4: `4 -> 3` x2

Corrections split:
- over-count -> correct: **37**
- under-count -> correct: **21**

Regressions split:
- correct -> over-count: **27**
- correct -> under-count: **17**

This explains why K1 is the largest winner: the group strongly repairs prior over-counting around the K1/K2 boundary.
K4 also benefits from recovering K3->K4 under-counts.

## Wrong-to-wrong movements

Among the **36** changed cases that remain wrong:
- **25** move closer to the true K
- **3** keep the same absolute K error
- **8** move farther away

So even outside exact corrections, most non-exact changes reduce ordinal cardinality error.

## Probability-shift signature

Across the changed cases, the group is not acting as a uniform cardinality offset.

Observed average behavior:
- for K1/K2 cases, probability mass tends to move toward **K1** and away from K3/K4;
- for K4/K5 cases, probability mass tends to move away from **K3** toward **K4/K5**;
- K3 sits at the crossover and is therefore the most exposed class.

This is consistent with the exact transition pattern:
- many `K1: 2->1` corrections,
- many `K2: 3->2` corrections,
- many `K4: 3->4` corrections,
- but K3 loses correct cases in both directions (`3->2` and `3->4`).

The group therefore appears to **reshape local decision boundaries around K3**, rather than simply raising or lowering the predicted count.

## Why K3 drops

K3 exact changes by **-5 net**:
- corrections: 5
  - `2 -> 3` x4
  - `4 -> 3` x1
- regressions: 10
  - `3 -> 2` x5
  - `3 -> 4` x5

The K3 loss is symmetric: five correct K3 cases are pushed downward and five upward.
That argues against a simple global count bias. K3 behaves as an unstable boundary class between the two beneficial corrections.

## Why Cluster A drops

Cluster A has 499 rows.

`freeze_local_combo`:
- exact: **18.84%**
- under: 378
- over: 27

Selected group:
- exact: **17.84%**
- under: 382
- over: 28

Within Cluster A:
- corrections: **4**
- regressions: **9**
- net: **-5**
- changed predictions: **21**

Net exact by K inside Cluster A:
- K2: **-6**
- K3: **+1**
- K4: **0**

The dominant Cluster A regression is true K2 `2 -> 1` (**5 cases**), followed by true K2 `2 -> 3` (**2 cases**) and `2 -> 0` (**1 case**).
Thus Cluster A's loss is specifically a **K2 boundary-stability problem**, not a general collapse of the cluster.

## Diagnostic verdict

The +14 outer exact gain is structurally coherent:
1. it is internally robust over folds 0/1/2/4;
2. it reproduces positively on outer fold 3;
3. it reduces both total under-count and total over-count;
4. 25/36 wrong-to-wrong changed cases move closer to the target.

The residual weaknesses are now localized:
- K0 protection against upward drift;
- K3 boundary stability;
- Cluster A K2 protection, especially exact K2 being displaced to K1.

Any next model change must be developed and selected **only on internal folds 0/1/2/4**. Outer fold 3 should not be used to choose thresholds, neurons, gates, or combinations.
