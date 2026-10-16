# S615: specialized and shared decision-hop training (EXPERIMENTAL)

Date: 2026-10-16. Experiment branch:
`codex/note-live-s615-multihop-training`, based on
`codex/note-live-harmonic-shadow-expert`.
Native **S613 official reference and S80 historical Exact-K** untouched.

## Objective and critical distinction

Audio capture callback size `--block-size` is NOT the model decision hop.
The original S613 was trained at 44.1kHz with HOP=256 (5.805ms).
It does not acquire accuracy automatically when run at 128 (2.903ms)
or 512 (11.610ms). S615 creates **three separately learned**
native NOTE heads and **one pooled shared three-hop head**, with
hop-scaled causal temporal features and separately calibrated selectors.

The original 2048/4096/8192-sample FFT windows, 49 pitches (MIDI40–88),
K-active 0..6 and per-pitch onset/sustain/release output remain.
No future audio, no BP/V88/MIDI sidecars as neural inputs. Explicit no
post-attack wait. All actual audio must have arrived before prediction.
No new piano output engine.

## Real feature ABI correction

The actual `CausalPitchFeatures._frame` emits 126 feature columns,
not 110 as an older description claimed. CI measured shapes on
the real (not fake) GuitarSet audio at each hop and proved that
`feature_track(audio,hop=256)` remains byte-identical to the S613
default path.

S615 features are stored under `data/S613/multihop/hop{128,256,512}/`
with SHA256 of PCM, features, source ID, original annotation SHA,
provenance and composition split. The original S613 cache is unchanged.

## Source partitions, leakage protection

Reuse the **exact pinned** S613 source protocol:
- Training: 53 clips, 36 GuitarSet+17 URMP (up to 24s each);
- Development: 17 clips, 12 GuitarSet+5 URMP;
- Published historical held-out: 22 clips (12 GuitarSet+5 URMP+5 IDMT).
No S615 training, feature preparation or threshold tuning may consume
the held-out test annotations. The original 69.03% S613 F1 at 80ms
remains the official held-out baseline until new, independently locked
evaluation.

See `scripts/s615_prepare_multihop.py` and
`scripts/s615_train_multihop.py`. Candidate training:
- same native K-active MLP (49 pitches * 9 spectral count inputs);
- predicted K used to condition per-pitch onset/sustain/release MLP;
- positive attack and release rows ALWAYS preserved even if they
  exceed the nominal per-source row budget; background/sustain is
  downsampled, with sampling inverse weights and seeds frozen;
- separate per-hop thresholds chosen only on development tracks;
- `hop128`, `hop256`, `hop512` use one hop each;
- `shared` pools train rows for all three in one checkpoint, with
  per-hop dev-selected selectors. Shared candidate is not a
  guaranteed accuracy improvement or a new proven architecture.

## Correct manual training procedure

Prepare original S613 train/dev audio caches using the pinned protocol,
not the historical test labels:

```powershell
mkdir data/S615_full
Copy-Item analysis/evidence/v273-s613-strict-live/protocol.json data/S615_full/protocol.json
python -m scripts.s613_prepare_strict_live --root data/S615_full --phase development
python -m scripts.s615_prepare_multihop --root data/S615_full --hops 128 256 512
foreach ($mode in @("hop128","hop256","hop512","shared")) {
  python -m scripts.s615_train_multihop --root data/S615_full --mode $mode --out "results/S615_full/$mode"
}
```

Training will download large pinned audio archives, require Torch CPU,
SciPy and substantial disk/RAM. DO NOT mistake the small real GuitarSet
pilot for the full experiment.

To play one properly trained S615 candidate with a fixed hop:

```powershell
python -m scripts.live_strict_notes --mic --device "MME:External Microphone" --model "results/S615_full/hop128/NOTE_multihop_candidate.npz" --hop-size 128 --block-size 256 --queue-blocks auto --soundfont "PATH_TO_YDP_PIANO.sf2"
```

With the SHARED model, opt-in auto CPU selection chooses only from hops
advertised in checkpoint `trained_hops`:

```powershell
python -m scripts.live_strict_notes --mic --device "MME:External Microphone" --model "results/S615_full/shared/NOTE_multihop_candidate.npz" --hop-size auto --block-size 256 --queue-blocks auto --soundfont "PATH_TO_YDP_PIANO.sf2"
```

Original S613 auto (explicit `--hop-size auto`) is now restricted to
its actually trained hop=256. This does not imply that a shared candidate
has good F1 at all trained hops; it selects the lowest computationally
feasible among its advertised hops, currently NOT according to pitch
accuracy. No auto mid-note switches are permitted.

## Evaluation gates

`scripts/s615_exactk_audit.py` groups onsets within 30ms, matches
events within 80ms and scores **Exact-K number of NEW attacks** and
complete pitch-set equality separately from NoteOn F1. This is a declared
grouping heuristic, NOT the official historical S80 Exact-K benchmark and
NOT the K-active probability head.

The candidate must be compared on:
- train-free GuitarSet / URMP / IDMT independent test splits and fresh
  iRig recordings;
- NoteOn F1 at 50ms/80ms, missed attacks, real octave E2+E3 and adjacent
  notes; onset Exact-K K1/K2/K3/K4/K5/K6, under/over counts;
- time of the first correct attack, NoteOff lifetime/duration and native
  causal lookahead, output piano underruns;
- CPU p95/p99 relative to each hop and total real microphone-to-SF2
  latency on the target Windows PC.

## Current scientific status

Software smoke GitHub Actions run **38862219229**: **17 tests pass**
including mock-source training of all 4 variants and positive-preserving
Exact-K regression audit. **Mock-source F1 scores are not meaningful.**

Pilot script `scripts/s615_seed_guitarset_pilot.py` creates a
strict train/dev-only **6 real GuitarSet train / 3 real GuitarSet dev** subset
using pinned source audio and annotation SHA256. Real pilot workflow
`.github/workflows/note-s615-real-pilot.yml` runs 4 count/6 pitch
epochs per candidate, publishing development-only reports/artifacts.
This is intentionally too small for model promotion. Refer to its
actual GitHub Actions conclusion and artifact before quoting any score.

## Completed real GuitarSet pilot (measured, NOT held-out)

Workflow: https://github.com/Andriamarosoa/note/actions/runs/38862219355
Result **SUCCESS**. Pinned pilot uses 6 train GuitarSet MIC recordings
from composition groups Funk2 and BN1, 3 development GuitarSet MIC recordings
from BN3. Train/dev songs are composition-disjoint. Only 4 K/count epochs
and 6 pitch epochs, so this is a low-cost architectural validation, not
a fair comparison to fully trained S613. All 349 dev reference NoteOn
events were evaluated with the original pitch-identical +/-80ms metric.

| Mode | Hop | TP | FP | FN | Dev NoteOn F1 | Onset Exact-K groups / 181 |
|---|---:|---:|---:|---:|---:|---:|
| Trained specialized | 128 | 80 | 311 | 269 | 21.62% | 24 / 181 = 13.26% |
| Trained specialized | 256 | 49 | 92 | 300 | 20.00% | 26 / 181 = 14.36% |
| Trained specialized | 512 | 0 | 0 | 349 | 0.00% | 0 / 181 = 0.00% |
| Trained shared | 128 | 115 | 263 | 234 | 31.64% | 26 / 181 = 14.36% |
| Trained shared | 256 | 111 | 230 | 238 | 32.17% | 25 / 181 = 13.81% |
| Trained shared | 512 | 117 | 271 | 232 | 31.75% | 34 / 181 = 18.78% |

True simultaneous-onset groups in this small dev cohort (30ms grouping):
**K1=100, K2=15, K3=45, K4=21**. No true K5/K6 onset groups here.
On the **shared** candidate:
- hop128: K1 15/100, K2 2/15, K3 9/45, K4 **0/21**;
- hop256: K1 18/100, K2 1/15, K3 6/45, K4 **0/21**;
- hop512: K1 26/100, K2 2/15, K3 6/45, K4 **0/21**.
Exactly matched pitch sets are even less frequent (12/181, 13/181,
14/181 for shared 128/256/512 respectively). In other words, the
multi-hop concept **passes software/training tests but does not yet
solve genuine polyphonic K3/K4 or note identification**.

Do not mistake the 18.78% grouped-onset Exact-K diagnostic for the
official historical S80 poly K2-K6 Exact-K benchmark: different data,
onset grouping, thresholds, and evaluation task.

Full replay with 53 train / 17 dev tracks and then a genuinely
independent new test is needed before comparing to 69.03% frozen S613.
The original test split has previously been studied by the project;
a new test would be preferable for an independent promotion decision.
A trained model must also be tested under actual iRig/FluidSynth
hardware latency and on true E2+E3/close-note pairs. This pilot does
NOT provide sufficient evidence to change the official default hop.

For per-K/NoteOn source-of-truth audit, see completed extraction of
the artifact reports at:
https://github.com/Andriamarosoa/note/actions/runs/38862872386
