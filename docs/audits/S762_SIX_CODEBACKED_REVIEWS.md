# NOTE S762 — Six code-backed second-opinion audits

Date: 19 October 2026.
Honest method: six distinct engineering review perspectives executed within ONE conversation, with actual repository and GitHub Actions inspection. These are not six separate ChatGPT sessions or independent LLMs. The S762 implementation is a research branch, not a deployed LIVE upgrade.

## Verified common evidence

Signed reference S708/S707 (GuitarSet CAL32, 32 recordings): Onset F1 74.2639%; Exact-K 62.0455%; Active Exact-K 56.2762%; Offset 77.3292%; Full Note F1 57.1319%; Frame Pitch F1 76.4626%. TP=2913, FP=849, FN=1170, correct offsets=2241/2898 eligible, 126304 time frames and 100243 non-silent frames.

Primary sources: GitHub Actions runs 39094682419 (S708), 39207493606 (S750), 39209836870 (S753), 39213327892 (S755), 39218925531 (S760), and 39221050543 (S761). S761 completed 80 FIT recordings and 32 CAL recordings, with 238 candidate rows, 49 positive new Onset rows, and 37 positive complete-note rows. Its 12 MIDI trials yielded zero safely improving candidate.

## Audit 1 — Exact-K head versus event decoder

PROVEN in src/causal_note/s704_joint_heads.py: JointSelector only activates its explicit K-based voice budget when max(K posterior) >= count_confidence. Signed S708 policy has count_confidence=1.1, while a normalized 7-class posterior cannot exceed 1. Therefore this budget condition is disabled. K still matters in GuidedSelector (guide_confidence=.5) and elsewhere; it is NOT entirely ignored. The threshold could be intentionally disabled, and blindly restoring it risks K5/K6 false negatives.

MEASURED S753: of undercounting decoder decisions, K2 had 987 cases where the K head predicted enough (48.48%); K3 had 828 (52.31%); K4 had 1233 (47.39%). For K6 the frozen head recognized 0 of 189 true K6 decision fragments; best S755 specialist recognized 3 of 189. These are associations, not proof of unique causal causes.

SOLUTION: separate rescue for head-sufficient K2-K4 from rare K5-K6 classification, and analyze the conjunctive per-pitch onset/sustain/rearm/rise gates for both categories before designing a new selector. Never force pitch purely to fill K.

## Audit 2 — Harmonic evidence and missed pitch attacks

PROVEN: src/causal_note/s731_auditory_scene.py already implements harmonic ownership up to source partial 32, independent candidate novelty and old-source explain-away. S755 features include pitch-coherent harmonic summaries. S761 only changes K probabilities before re-decoding the same per-pitch attack/sustain values: an attack not proposed by this decoding remains invisible to both learned S761 quality heads.

MEASURED S750: low E2-E3 missed 398/901 attacks, mid F3-A4 missed 701/2886, high register missed 71/296. S753 near-harmonic missed/false-positive frame indicators: 998 at 12, 750 at 19, 187 at 24 semitones (not proof of harmonic causality).

SOLUTION: train a new causal per-pitch proposal head conditioned on S731 scene novelty, parent-harmonic explanation, transient spectral changes and the K posterior as soft context. Evaluate proposal RECALL before suppression. Compare low, middle, high registers and repeated-pitch / octave cases.

## Audit 3 — Source note identity, release and Hungarian assignment

PROVEN: src/causal_note/s746_instance_guard.py preserves native S708 MIDI event metadata and order; nevertheless, inserting an earlier same-pitch extra NoteOn can change the one-to-one onset matching performed by scripts/s613_train_strict_live.py, making a previously correct source NoteOff fail to receive credit. S758-S760 show this in real CAL32 replay. A protected source event is not automatically a protected correct match.

MEASURED: S759 best Active Exact-K rose by 0.325 percentage points but lost 3 previously correct offset assignments and decreased Full Note/Offset. S760 causal 23/46/70ms delays reduced collisions but all variants still failed nonregression.

SOLUTION: test same-pitch events separated by 23/46/70ms, source-preempted supplemental notes, ties and future-free prefix invariance. Preserve official matching rather than changing metric to reward extras. New proposal labels should penalize events likely to duplicate still-unmatched signed attacks using only past source/acoustic information.

## Audit 4 — FIT-only ML sample and calibration quality

PROVEN: S761 trained logistic regression with class_weight balanced, then min(onset_score,complete_score) across only 238 proposed FIT examples (49 onset-positive, 37 complete-positive). The resulting scores are exploratory rankings, not calibrated event probabilities. Four near-related S755 experts can generate duplicate examples; their effective independent sample size must be audited, not assumed equal to 238. High thresholds admit zero notes and cannot meaningfully improve recall; permissive thresholds regress other metrics.

SOLUTION: deduplicate training proposals by composition/actual candidate timestamp/pitch; split by composition for out-of-fold learning, calibrate on FIT only and report PR/coverage and confidence intervals. Avoid tuning thresholds repeatedly on CAL. More independent valid candidates are needed before stronger risk calibration can help.

## Audit 5 — Live audio and causality

PROVEN: a S708 decision consumes four 256-sample hops at 44100 samples/s, giving a 23.22ms decision interval; one S760 confirmation block adds ~23.22ms, two ~46.44ms, three ~69.66ms to supplemental births. S707 release reads received history and may schedule causal future closure. The S761 and S762 evaluated scripts are OFFLINE replays with causal contracts, not proof of integrated microphone-to-MIDI latency.

SOLUTION: keep prefix-invariance checks (change all unseen audio, future source NoteOns and candidate future NoteOff while checking current decisions), force timestamp monotonicity and source priority, test queue overflow, processing deadline and same-time ordering. No LIVE deployment until all-six accuracy and physical latency are verified.

## Audit 6 — Mathematical evaluation and independent reproducibility

PROVEN in scripts/s704_joint_metrics.py: Onset F1 = 2*TP/(2*TP+FP+FN). Full Note F1 = 2*correct_offset/(2*TP+FP+FN). As correct_offset <= TP, Full Note F1 cannot exceed Onset F1. Even fully independent heads do NOT imply independent final evaluation metrics.

SIGNED denominator D=2*2913+849+1170=7845. Conditional illustrative scenario: if x previously missed, perfectly correct new attacks are emitted without new false positives and without source TP loss, new Onset F1=(5826+2x)/(7845+x). Achieving >80% requires at least x=376 correct newly recovered attacks; this assumes no FP reduction or other changes. If current 2241 correct offsets remain unchanged and all 376 new attacks get correct offsets, Full Note F1=5234/8221 ~=63.67%, still nowhere near 80%. In the optimistic scenario of repairing all 657 eligible incorrect source offsets, retaining FP, and giving every new note a correct eligible offset, at least 401 newly recovered correct attacks are needed for Full Note F1 >80%. These bounds are conditional and can change with false-positive removal.

MEASURED S750: 36890 undercounting active frames versus 6940 overcounting active frames. CAL32 has been examined repeatedly across many S-series, so CAL is a research calibration set, NOT an untouched independent test. Later external IDMT must be clearly separated from selection and transfer testing. HELD/TEST should remain sealed until the model is fixed.

SOLUTION: preserve the six official S704 metrics and signed baseline; group confidence intervals by composition/player, document transfer tests on separate source, do not backdate notes, do not use label or future PCM at inference, and do not promote without fresh independent evidence.

## Contradictions resolved

The Exact-K reviewer favors count guidance, while the harmonic reviewer notes rare K classes are underpredicted and count forcing is unsafe. The offset reviewer favors suppressing risky additions, while the evaluation reviewer proves suppressing almost all new notes cannot reach Full Note 80%. The live reviewer warns long confirmation delays consume an 80ms Onset tolerance. These tensions point to improving TRUE candidate generation before tightening acceptance thresholds.

## Concrete correction performed — S762

PROVEN architectural mismatch: S761 tested four specialists independently, each across an entire recording, instead of choosing a combination per fragment. Therefore it did not implement the requested fragment-adaptive multi-expert router.

IMPLEMENTED on codex/note-s762-fragment-ensemble-six-audits:
- scripts/s762_fragment_ensemble_cal32.py pools all four S755 experts at each causal fragment, selects the highest-quality proposal per (time,pitch), deduplicates overlaps and keeps all signed source events. Does not read future NoteOff fields when choosing proposals.
- test/test_s762_fragment_ensemble_cal32.py tests fragment-level mixed experts, future-offset invariance, native source preservation and future same-pitch preemption.
- .github/workflows/note-s762-fragment-ensemble-cal32.yml launches FIT80 training of quality heads and 12 real full-MIDI CAL32 score comparisons with explicit no-regression gates.
- scripts/s761_fit_dualquality_birth_routing.py is refactored only on the isolated S762 branch to reuse its common FIT80/CAL32 evaluation without claiming S761 was originally a fragment ensemble.

S762 is not a new independent-note acoustic generator and cannot by itself solve missing pitch evidence. Even if S762 passes all tests, do not claim >80 or LIVE deployment before CAL metrics prove it. Priority after S762 is a FIT80-trained new acoustic/pitch proposal generator, candidate recall ceiling for K2-K6, and grouped out-of-fold quality selection.

**Outcome of six reviews:** clear architectural misalignment found and corrected in research code; a disabled K budget condition verified; a coupled metric upper bound established; six-score 80% target not yet met.
