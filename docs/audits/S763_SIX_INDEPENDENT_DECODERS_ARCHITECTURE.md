# NOTE S763 — Six independent causal decoders

Date: 19 October 2026. Research-only architectural prototype, not validated audio-to-MIDI performance.

## Why separate decoders?

S708/S762 combined specialist outputs in one mutable MIDI state. JointSelector uses active/armed/last_on; GuidedSelector consumes K to admit weak acoustic attacks. Improving the K head can change actual MIDI notes and thereby move Onset/Offset/Full Note/Frame Pitch scores. Distinct training heads never implied independent output state.

The six official scores are also necessarily coupled because scripts/s704_joint_metrics.py scores ONE final MIDI stream. The design objective is to isolate decisions and diagnose regressions, not to make final metrics mathematically independent.

## Implemented in src/causal_note/s763_independent_decoders.py

Each 1024-sample (~23.22ms at 44100Hz) causal fragment has an immutable 49x3 Onset/Sustain/Release posterior and normalized K0..6 posterior. Each axis receives a distinct immutable copied snapshot. Six modules have separate state and output own axis-specific hypotheses:

1. Exact-K: K0..6 posterior -> integer K, stateless.
2. Active Exact-K: private sustain hysteresis memory -> active note count (independent of Exact-K decoder output).
3. Onset: its own onset threshold, armed and refractory state -> NoteOn proposals.
4. Offset: its own release/sustain threshold and armed state -> release proposals. These are NOT valid final MIDI NoteOff without reconciling note identity.
5. Frame Pitch: its own sustain threshold and sorted top-six binary pitch state -> framewise pitch hypothesis.
6. Full Note: private attack, sustain, release and note-instance registry -> an independent stream of paired NoteOn/NoteOff hypotheses. It does NOT read OnsetDecoder or OffsetDecoder outputs.

Nuance: the modules still share upstream acoustic model observations (copied), which is appropriate for feature sharing, but no state or decisions. A model retrain that modifies common posteriors could still move multiple axes. The testable isolation guarantee is for decoder architecture/threshold/intervention changes, not arbitrary changes to their shared upstream source.

IsolatedDecoderBank accepts exactly six distinct object instances; it refuses missing channels, incompatible result axes and timestamps. The execution order is arbitrary and testably invariant. Each decoder is causal; private state never enters another decoder.

## Fusion arbiter intentionally read-only

ShadowArbiter summarizes conflicting proposals but does not inject or suppress any MIDI note. It preserves the signed S708/S707 source stream unchanged. This guarantees no regression of official six baseline scores during this shadow-only phase, but it cannot improve them either.

Separate diagnostic head scores MUST NOT be reported as official six MIDI metrics: every future candidate is evaluated on the single final serialized MIDI stream.

## Tests

test/test_s763_independent_decoders.py covers:
- six distinct decoder objects, no state reuse;
- immutable input arrays and separately copied fragment inputs;
- deliberately malicious Exact-K decoder that attempts to mutate input; all five other outputs remain byte-for-byte equivalent;
- independent Onset and Offset threshold changes leaving all other outputs unchanged;
- permutation of decoder execution order;
- prefix invariance under future fragment change;
- separate Full Note On/Off pairing, EOF panic closure;
- clock causality and refusal of future-stamped events;
- original source MIDI unchanged under shadow arbiter.

CI: .github/workflows/note-s763-isolated-decoders.yml

## Next validation gates

S763: pass isolation tests. Do not change original S708, S707, metrics or microphone LIVE.

S764: shadow replay six decoders on GuitarSet. Measure their own independent predictions and compare to official S708 MIDI. Attribute failure to the individual head, its decoder or future arbiter. No promotion.

S765: fit a separate instance-aware causal arbiter using FIT only, train/calibrate by composition, stable note identity, explicit non-regression. Do not use CAL to continuously tune.

S766: full output MIDI six-axis validation, per-K0..K6, register, repeated pitch and independent IDMT transfer. CAL32 has been repeatedly seen by many experiments and cannot be presented as an untouched final test. HELD/TEST remain sealed until a legitimate model freeze.

LIVE: only after a validated improvement, measure actual queue, CPU and latency before guarded rollout.

## Decision

S763 is a true architecture-isolation prototype, not an 80% score achievement or real LIVE integration. Its independent decoders make it possible to pinpoint the module that causes a regression; six final metrics remain coupled by the MIDI arbiter and matching rules.
