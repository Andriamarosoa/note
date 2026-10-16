# S621 — real 128-hop-trained fast NOTE head + frozen S613 256-hop hybrid

Research branch: `codex/note-live-s621-trained-dual-hop` (base S620).
Original S613 baseline, official live reference and S80 are untouched.

## Why S621?

S620 runs native fast128 and frozen S613 hybrid reference256 in parallel,
but the 128 path was using a checkpoint trained ONLY for hop256.
This is a diagnostic, not a scientifically correct multirate model.

S621 trains a distinct native `hop128` neural NOTE model on the **full**
pinned 53-source GuitarSet/URMP train split (with 17 source-disjoint dev):
K0–K6 count head and per-pitch onset, sustain, release. It retains every
real onset/release example, uses 50% sustain + 8% silence/background and
a 15,000-row budget for negatives, reproducing the S617 ablation.
The original S613 frozen neural+ONNX hybrid remains untouched at hop256.

Dual path:
- original reference S613 256: full frozen ONNX+neural consensus;
- new S621 fast 128: separate trained native NOTE count/pitch head;
- fast offers a new pitch only where existing causal 126-feature spectrum
  has positive pitch novelty and the native onset+sustain agree;
- per-note two-source MIDI ownership prevents double NOTE-on or orphan
  NOTE-off, with the reference 256 path prioritized at equal times.
- no future note samples or post-onset waiting; original 2048/4096/8192
  FFT sizes unchanged; independent of microphone block-size.

## Windows testing (experimental, no default change)

After the workflow completed successfully and its candidate was retrieved
from the GitHub Actions artifact:
```powershell
git fetch origin
git switch codex/note-live-s621-trained-dual-hop
git pull --ff-only

# Shadow: records full dual CPU work, emits only original NOTE MIDI.
python -m scripts.live_strict_notes --mic --device "MME:External Microphone" --block-size 256 --dual-hop-mode shadow --dual-hop-fast-model ".\results\S621_full\hop128\NOTE_multihop_candidate.npz" --soundfont "PATH_TO_YDP_PIANO.sf2"

# Guard: truly fuse live fast128 candidate notes with original slow256 MIDI
# DO NOT claim 80%, and do not use as official reference.
python -m scripts.live_strict_notes --mic --device "MME:External Microphone" --block-size 256 --dual-hop-mode guard --dual-hop-fast-model ".\results\S621_full\hop128\NOTE_multihop_candidate.npz" --soundfont "PATH_TO_YDP_PIANO.sf2"
```

`--dual-hop-mode off` remains 100% original S613. `--hop-size 256`
remains the baseline; the fast 128 cadence is internally determined.
Merging changes event times and note lengths in `guard`, requiring
independent validation (shadow output remains unchanged).

## Training + evaluation (same frozen protocol)

GitHub Actions workflow `note-s621-full-trained-dual-hop.yml`:
1. Seed exactly 53 TRAIN / 17 DEV pinned GuitarSet+URMP sources;
   leave all TEST annotations and audio untouched.
2. Compute original causal 126-col/49-pitch features at hop128 for
   all TRAIN/DEV.
3. Train real NOTE fast128 24 K epochs and 28 pitch epochs, preserving
   onset/release samples.
4. Compute hop256 features ONLY for 17 DEV tracks for apples-to-apples
   comparison with original S613.
5. Reproduce the true 256 original frozen S613 hybrid (TP1241 FP734
   FN838 NoteOn ±80ms) and evaluate 24 simultaneous 128+256 policies.
6. Report new recovered truth identities, lost old truth, added FP,
   real grouped-onset Exact-K K1..K6 and separate GuitarSet/URMP breakdown.

Run initiated:
https://github.com/Andriamarosoa/note/actions/runs/38874449408
Do not quote success, F1 or K accuracy before final workflow verdict.

## Full scientific caveats

- Fast hop128 training *alone* does not mean increased acoustic accuracy.
- CPU approximately tripled in S620 synthetic CI; true Windows/iRig
  audio+FluidSynth end-to-end latency not measured.
- Development-tuned thresholds on the SAME 17 original compositions
  are not an independent held-out evaluation; promotion requires new
  independently labeled songs/strums with true close-pitch and E2/E3
  octave attacks, K2–K6 and error-free output.
- Target 80% LIVE NoteOn F1 is unachieved and cannot be inferred from
  offline noncausal Basic Pitch 80% results.
