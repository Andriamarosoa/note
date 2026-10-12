# PR #17 — Deduplicated GitHub Actions research workflows

**Scope:** experimental branch `codex/v273-scope-rescue`. No changes
to the official `freeze_local_combo` reference or to historical artifacts.

## Root cause

Historically, 40 research workflows declared:

```yaml
on:
  pull_request:
    branches: [codex/v273-regression-loops]
    paths:
      - 'scripts/an_experiment.py'
```

GitHub evaluates `pull_request.paths` across the files changed by the
**whole PR**, not merely the most recent commit. This PR already changed
many scripts; therefore each new commit caused the same ~40 jobs to rerun.

## Applied repair

All 40 exact affected workflows were changed atomically in
[`771fd29`](https://github.com/Andriamarosoa/note/commit/771fd297e7fa5d08e9e88956f45a9fdce8cbf610):

- `pull_request` was replaced with a `push` trigger limited to
  `codex/v273-scope-rescue`. Its original per-file `paths` filters were
  retained; only paths in the **new commit** can trigger the workflow.
- `workflow_dispatch` was preserved for historical/manual audits.
- Per-workflow concurrency
  `group: ${{ github.workflow }}-${{ github.ref }}` plus
  `cancel-in-progress: true` prevents redundant overlapping jobs.
- No workflow history, model weights, or GitHub Actions artifacts
  were deleted.
- Exactly the affected 40 workflows and an audit script + manifest
  were modified, leaving unrelated historical workflows intact.

The self-check [S124](https://github.com/Andriamarosoa/note/actions/runs/38187307388)
subsequently returned `S124_DEDUP_AUDIT_PASSED`, `converted=40`.
Its parser supports both block and inline YAML `paths` notation.

## Additional invalid S81 workflow

S81 preregistration contained an unquoted YAML step name with a colon.
The step label was quoted in
[`d77387c`](https://github.com/Andriamarosoa/note/commit/d77387c93ce516739b6167e4069f68985ba9fdf1);
the preregistration workflow now has only `workflow_dispatch`, not an
automatic push or PR trigger. The preflight remains available but is
not a S80 independent model evaluation.

## Independent trigger isolation evidence

After the bulk repair, a code-only edit
[`7fe463f`](https://github.com/Andriamarosoa/note/commit/7fe463f5e6f687d95b4b080d3c6dbc4221e04194)
started **only the S125 run**, not the earlier ~40 research audits.
This confirms that the repeated PR-trigger fanout is fixed for current
research commits. The one-time migration commit itself necessarily
started the 40 newly edited workflow files.

**Promotion gate is unchanged:** S123 replay on fold0 reconstructed
13,053 correct vs original 13,187 (−134); original confusion is not
identical. Thus S5's historical `owned527` feature contract is still
unverified. No independent S80 external score or promotion is claimed.
