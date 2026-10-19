# NOTE — six-auditor protocol for EACH Sxxx iteration

S787 creates independent GitHub Actions audit jobs; these are **not** the
native ChatGPT Work agents. The six roles work on the SAME frozen artifacts,
each writing its own machine-readable evidence.

## Required roles

1. **Data & labels** — signed S708 checkpoint digest, FIT80/CAL32 held-out
   composition splits, HELD/TEST closed. Reject leaks and provenance drift.
2. **Acoustic fundamentals** — preserve all 49 pitch locations, fundamental
   and harmonic interval topology, no labels in the acoustic extractor.
3. **Architecture** — independent heads, strictly per-fragment routing,
   low-K hallucination guards, no future or oracle signal in selection.
4. **Learning & generalization** — re-derive FIT-only cross-composition
   feasible policies, objective selection and every non-regression guard.
5. **Metrics & regressions** — independently recompute 7x7 count confusion,
   per-K statistics and six *signed official* S708 reference numbers.
   Never confuse 1024-sample shadow Exact-K with final 256-hop MIDI.
6. **LIVE & reproducibility** — check changed paths against allowlist,
   causal prefix invariance tests, and no final MIDI/Onset/Offset mutation.

## Mandatory iteration

1. Freeze ONE source commit and reference checkpoint before training.
2. Run all six role audits independently against immutable common artifacts.
3. Persist six separate JSON audit reports and one aggregated verdict.
4. A missing, incomplete, contradictory or failed audit BLOCKS the experiment.
   The consensus job must fail closed, never silently skip a role.
5. If experimental candidate improves FIT, CAL is only a frozen diagnostic,
   not an opportunity to retune.
6. Repeat all six audits after EVERY Sxxx source change.
7. For an official promotion, run independent validation of Onset F1,
   Exact-K, Active Exact-K, Offset accuracy, Full Note F1, Frame Pitch F1
   on one final strict causal MIDI event stream (hop size 256).
   All six must be above 80%, with no regression to the signed baseline;
   hold HELD/TEST independent and preserve S708/S707.
8. Do not approve experimental 1024-sample shadow candidates for LIVE use.

## S787 implementation and boundary

The workflow file at .github/workflows/note-s787-six-role-audits.yml
waits for a frozen S786 run, downloads signed S708 + S786 artifacts,
and executes six PARALLEL GitHub matrix jobs. Each job independently
executes scripts/s787_six_role_audit.py for its one specialty.
The seventh consensus job in scripts/s787_audit_consensus.py checks all
six roles passed, refuses missing reports, and disallows any auto-promotion.

Its SUCCESS means six audits passed, **not** that new metrics >80%.
To apply this to S788 and all following loops, wire the audit gate into
each new workflow and adapt source-specific checks. This configuration
does not globally trigger on every arbitrary branch: subsequent workflows
MUST opt into the same six-role mandatory check. Each changed source
commit invalidates previous audit findings.

Record results and GitHub action run IDs. Scientific regressions are
negative findings even when CI exits successfully. Require a justified
hypothesis before another Sxxx trial.
