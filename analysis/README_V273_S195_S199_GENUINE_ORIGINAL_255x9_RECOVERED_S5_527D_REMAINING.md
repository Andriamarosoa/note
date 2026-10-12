# S195–S199 — Restoration of genuine historical 255 group proposals and nine local audits; S5 original 527D still blocked

**Branch:** `codex/v273-scope-rescue`. **Official reference:** `freeze_local_combo` unchanged. **S80:** not promoted. This is numerical/historical provenance, not independent generalization.

## Verified outcome: original 255 × 9 historical tensor restored

The preserved historical 7,493 eligible events contain original group proposals with 255 masks each; original producer caches were separately kept. The work proceeded:

1. **S195** inventoried preserved NPZ/ZIP schemas and identified the authentic `group_proposal` in `analysis/evidence/v273-contextual-risk/probability-evidence.npz`, alongside the same actions in `analysis/evidence/v273-regression-loops/prepared/inputs.npz`. Parent risk probabilities/logits and 20D destination-aggregated audit summaries are saved; no raw 255×9 audit tensor had been preserved in current versioned snapshots.
2. **S196** independently reconciled **1,910,715 original group actions = 7,493 × 255**, **0 mismatches**, restored the authentic parent risk/logits and original saved 20D aggregate audit summaries. Replayed over **337,185 mask-membership scalar checks**, **0 errors**.
3. **S197** restored original pinned 14 specialist + 14 learned-corrector cached producer fold sets from `analysis/evidence/v273-audit-memory/source-archives/upstream-producers.zip`. It ran the original `joint_groups`, `local_audits` and `group_context` source code, preserving inner/outer fold reference exclusions. On an initial 128 held-out historical fold-0 rows, exactly reproduced original 255 actions and the 20 destination-audit aggregates with **max absolute error 0**.
4. **S198** repeated the full historical recomputation on every original eligible row from the four folds, then assembled an immutable full tensor with **7,493 × 255 × 9 = 17,196,435 audit scalars**. It independently confirmed every original event was reconstructed once, all 255 group proposals matched, and **all historic 20D aggregate audit values matched with maximum absolute difference 0**.

| Held-out fold | Historical rows | Original 255-action mismatches | 20D audit-summary max discrepancy |
|---|---:|---:|---:|
| 0 | 1,998 | 0 | 0 |
| 1 | 1,781 | 0 | 0 |
| 2 | 1,596 | 0 | 0 |
| 4 | 2,118 | 0 | 0 |
| **Total** | **7,493** | **0** | **0** |

**Important qualification:** The nine per-group original audit values were **recomputed from the actual original code and frozen cached producers**, and the archived aggregate audit summaries match exactly. There is no separately preserved original 255×9 array to compare per-group bytes directly. Thus this establishes **source-code and full aggregate parity**, not an independent byte-for-byte comparison against a nonexistent original per-group archive. Reference-fold historical labels are used only as training/reference outcome statistics; the receiver folds' labels never enter their own audit features. None of the historical input data constitutes a new unseen audio benchmark.

- S195: https://github.com/Andriamarosoa/note/actions/runs/38212022377
- S196: https://github.com/Andriamarosoa/note/actions/runs/38212219570
- S197: https://github.com/Andriamarosoa/note/actions/runs/38212536122
- S198: https://github.com/Andriamarosoa/note/actions/runs/38212710300
- Full reconstructed tensor: [S198 immutable original 255×9 artifact](https://github.com/Andriamarosoa/note/actions/runs/38212710300/artifacts/11730435822), SHA256 of internal output NPZ `375e9320e9bccd2a8c3e375232ceec15cdf85921057954eaa4ee58ed5dc955ca`.

## S5 527D: why it is NOT reconstructed

The original S4 `owned` representation used **111 known prefix columns** (58 acoustic summaries, 46 native geometry, 7 freeze-K onehot) and **416 original owned/time/activation/damping features**, total 527. S5 inherited this exact 527D design; its saved historic `designs_sha256` and original trained trees provide a checksum and tree thresholds, **not the missing 416 numeric source construction**.

The original source scan S132 searched 1,182 accessible historical commits and found no full original 416D constructor. Older attempts S118/S120/S123/S125/S126 evaluated three explicit **unverified** 416D feature layout hypotheses. Best log-layout candidate reproduced neither the original per-event probabilities (**0/15,952 within 1e-6**), nor exact original prediction output (1,882 K disagreements on historical fold0). Consequently none of those candidate functions can be renamed `original S5`.

**S199** broadened the binary matrix search to **all 45 versioned .npz and .zip files at this branch head, including nested ZIP→NPZ objects, and 669 NumPy array headers**. It found **no saved 527D or 416D original input matrix**. This is a scoped *negative provenance check*, not proof such a matrix could never exist in an external, deleted or untracked training workspace. Note the new S198 255×9 reconstruction lives in a GitHub Actions artifact, not versioned binary data, so it is outside S199's tracked-source scope.

- S199: https://github.com/Andriamarosoa/note/actions/runs/38212921360

## Remaining exact original provenance and deployment gates

1. For S5, locate **either** the original 416-column source code with preprocessing/ordering **or** a genuine saved original 527D input matrix keyed to historical events/held fold so it can be matched by exact per-event posterior replay. An original model alone cannot determine uniquely the data transform which trained it. No synthetic filler, arbitrary zero columns, column-order guesses or inverse optimization on exposed labels may be asserted to be the original implementation.
2. For **new WAV inference** of the original 255×9 grouping, recover deployable original model states behind the 14 fold-specific specialist caches and learned-corrector caches (saved caches contain historical predictions, **not guaranteed deployable estimators**), and reproduce the original event geometry, eight source vote probability channel alignment, nearest-neighbor audit training reference and fixed parent routing. Compare archived historical masks/audits before claiming new-WAV equivalence.
3. Only after S5 and all sixteen original S45 producer families are genuinely executable on **the same novel audio**, run full original S46, S80 and `freeze_local_combo` against the same pre-frozen independent Exact-K labels; otherwise keep the official reference.

**Verdict:** Historical original 255 proposals and 9 audit computations **restored and fully replayed across four folds**. **S5 original 527D input constructor not restored**. True new-WAV original full-S80 deployment still blocked and **S80 not promoted**.
