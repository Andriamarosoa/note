# S767 — Measured missed-onset candidate recall ceiling

**Verified on real GuitarSet CAL32** on October 19, 2026.
GitHub Actions: https://github.com/Andriamarosoa/note/actions/runs/39238508533

32 unit tests passed, all 32 audio recordings processed, exact six-metric S708 source parity verified. No training, no promotion, no change to S708/S707 or LIVE. FIT and CAL remain composition-disjoint; CAL is already repeatedly explored and is **not** an untouched test set.

## Fundamental finding

S708 misses 1170 of 4083 true guitar note Onsets within the evaluation tolerance of ±80ms.

The FULL ungated S765+S763 generator pool produced 6542 raw candidate instances. After deduplication by (NoteOn timestamp, pitch), 4170 unique proposals remain. Of these, 3886 were source-active blocked, 4 source-voice-budget blocked, 280 eligible, and zero source-recent blocked.

Using one-to-one matching of these fully generated candidates against S708's **baseline-unmatched** true notes:
- **97 / 1170 (8.2906%)** baseline-missed true onsets are proposed even in an oracle upper-bound matching.
- **72** recoverable missing true onsets have candidates that would pass the existing S765 source priority filter; **44** of these 72 also have a correct fully paired future release (strict ±50ms).
- **23** recoverable missing onsets are blocked by source-active pitch, 13 of these proposals' releases correspond to correct ±50ms offsets.
- **2** more are blocked by six-voice budget, one of them has a correct future release.
- **1073 / 1170 (91.7094%)** baseline-missed true onsets have **no matching candidate at all**, even BEFORE learned risk gating.

These matched counts come from CAL labels opened only AFTER the entire causal candidate proposal pool had been emitted. Label-based one-to-one matching is strictly a diagnostic oracle upper-bound of the **existing fixed proposal set**, not a LIVE capability, not an inference feature, and not a validated candidate selection score.

## Mathematical implication

Using the signed S708 TP=2913, FP=849, FN=1170 and assuming ALL 97 covered missed onsets were perfectly added with **zero new false positives and no source losses**, but NO other corrections to baseline FP/FN, maximum onset F1 would be:

  2*(2913+97)/(2*(2913+97)+849+(1170-97)) = 75.7995%

Even among the candidates admitted by the current source guard (72 potential hits), this idealized onset F1 ceiling is only 75.4074%. Thus **even a perfect S765/S766 arbiter cannot reach >80% Onset F1 using the present proposal pool alone** under these assumptions. An arbiter could improve F1 by removing existing false positives too; the calculation only bounds the add-only recovery route.

Only 23 of 1170 baseline missing attacks are recoverable from the large source-active blocked class, although 3886 proposals land there. Loosening that guard alone cannot solve the 80% target and would expose existing correct notes to regression.

## Combined with S766

S766 duration timers reduced preemptions but did not remove any prior-correct offset matching losses at any risk setting. One example lost a correct offset despite zero source preemptions. Source bytes remained unchanged; the evaluation can still match a different extra NoteOn to a true instance. S766 also failed strict S708 parity by one frame, so all S766 numbers are diagnostic only, none promoted.

## Engineering decision

Stop repetitive sweeps of S765 gates, four-expert selectors and extra-note timers on CAL32; those cannot fix the >91% missed-onset proposal coverage deficit.

Next research architecture should work on **new high-recall causal pitch attack proposals** using independent fundamental/harmonic residual evidence, short-term spectral change, inter-string overlap and voice-specific repeated attacks, trained on FIT composition folds and evaluated on an untouched held-out set after freezing. Treat these as proposals only. The separate S763 six-head architecture and original S708 MIDI remain unchanged.

Every added candidate must still maintain one causal NoteOn/NoteOff pair with stable per-instance identity and pass six-score complete MIDI validation without destroying previously correct source notes. Don't change the official scorer to manufacture an 80% result.
