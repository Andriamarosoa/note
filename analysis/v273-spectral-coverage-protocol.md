# Native spectral coverage audit, outer fold 3 only

The exact-timing audit found 96 assigned onsets after the end of the 40 ms
spectral window (87 rows, including 42 polyphonic rows). Assignment permits an
onset within 882 samples of any candidate, and a group spans up to 1,764 samples.
Therefore assigned relative onsets can span −882 through +2,646 samples.

Extend the spectral window to the **already declared** grouping + verification
horizon: 1,764 + 1,024 = 2,788 samples, or 63.21995 ms after the original group
start. With the unchanged 1,308-sample pre-context this is 4,096 samples and
31 frames. Centers span −1,180 through +2,660. This bound comes from the fixed
protocol, not optimization against held-out labels.

This remains causal **at finalization at that horizon**, not at group start.
It does not add to V100/V102's declared maximum acoustic delay, but live
wallclock/chunk scheduling latency is not measured in this audit. File-edge
zero padding must be reported separately from observed audio.

## Fixed inputs and checks

- Restore checksum-pinned exact caches from `v273-exact-timing-36122007201`.
  Do not re-mine candidates: prior re-mining had changed two group geometries.
- Preserve every candidate feature, mask, statistic, origin, full/retained
  timestamp, K target and string target byte for byte.
- The first 23 spectral frames must equal the saved maps byte for byte.
  Preserve their normalization and model positional coordinates exactly.
- Explicit schema 3 identifies the 31-frame window. Schemas 1/2 remain
  23-frame inputs; reject mixed windows and inconsistent metadata/shapes.
- Compare cache and runtime spectra. Recheck annotation assignment, pitch,
  string presence and physical onset timestamps. Only Gaussian time-target
  support expands, from 23 to 31 frames.
- Verify coverage against the protocol bounds and all annotations on the
  50 tracks of outer fold 3; record unassigned annotations and file padding.
- Unit regressions include a latest-possible assigned impulse, invariance to
  audio after the horizon, truncated groups and historical coordinate equality.
- Synthetic inference/gradient checks cover the native full-count and poly-count
  models. Defaults stay at 23 frames. Extended V240 scaffolds are count-only;
  historical event-set training losses have not been migrated to 31 frames.

## Interpretation

No model is trained on GuitarSet in this job, no other outer fold is evaluated,
and no output corrector is added. Passing the audit proves restored input
coverage, **not** improved ExactK. Official V27.3 remains the reference.
The existing 454 unassigned annotations and the broader counting errors still
require separate investigation. Archived 31-frame fold-3 data is for audit or
held-out evaluation, never fitting or selecting model hyperparameters.
