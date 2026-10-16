# NOTE strict LIVE — missed attacks and broken FluidSynth audio: causal/offline separation

Date 16 October 2026. Experimental branch `codex/note-live-harmonic-shadow-expert`.
**Native S613 weights, thresholds and original reference branch unchanged.**

## Findings established by source-code audit

1. A missing audible note can mean (A) acoustic attack without a native
   `note_on`, or (B) valid `note_on` emitted but silent/short/glitchy SF2 output.
   Do NOT infer which from listening alone.
2. Native selection `CausalSelection.step` needs onset>=0.60 AND
   sustain>=0.50 in the same frame; with onset falling below 0.20 for rearm,
   a 10-frame (58.05ms) refractory for each pitch and 5 release frames
   (29.02ms) of sustain<0.25 or release>0.75. Default `count_confidence=1.1`
   disables active-K veto. Therefore an onset model may respond yet output no
   `note_on`. S613's active-K head is not historical Exact-K of new onsets.
3. Evidence `analysis/evidence/v273-s613-strict-live/runtime_real_audio_events.jsonl`:
   140 native events = 70 note_on+70 note_off, 9 native notes lasting
   29.0249ms, 10 notes <=35ms (all causal release). These are short MIDI
   gates, not proof of a faulty synth.
4. `PianoOutput` drains all queued `note_on`/`note_off` commands BEFORE
   `synth.get_samples(frames)` in a 128-frame (2.902ms) output callback.
   **Test reproduced a silent same-callback MIDI note** when note_on and
   note_off are queued before the callback. This is a genuine rendering
   weakness, but its frequency on the user's PC is still unmeasured.
   Output PortAudio callback statuses previously weren't counted. Passive
   counters were added **only on the experimental branch**.
5. `--block-size 1024`: native decisions still every 256 samples, but
   `available_sample` and send-to-FluidSynth happen at the **end of 1024
   sample chunks**: grouping/jitter up to 17.414ms additional versus
   `--block-size 256`, plus normal callback/input latency. Using 1024 is
   less granular for a strict LIVE instrument.
6. A `--wav` source uses **the identical causal trained model and selector**
   as `--mic`. Thus a mistake already in the model's decision from identical
   sample input is reproduced offline. Device overflows, output jitter,
   MIDI-command coalescence and callback underruns differ.
7. **Do not use `--wav --soundfont` as an offline-audio quality test**:
   the CLI loops over the WAV as fast as the CPU permits, sends control
   messages to the asynchronous piano output and then closes the stream.
   The WAV timing is NOT paced. Use `--wav --out events.jsonl` followed
   by `render_live_midi_offline` instead. This renderer uses MIDI
   `decision_sample` timing and saves a standalone stereo WAV without
   PortAudio output callbacks.
8. Four user WAVs attached privately to the chat, not committed in GitHub,
   were checked independently for PCM clipping: peak -10.53/-17.25/-20.66/
   -17.75 dBFS (E2-only/E3-only/simultaneous/sequential); no clipped samples.
   The WAVs by themselves do **not** prove how the S613 detector behaves
   until replayed on a machine with the actual checkpoint.

## Reproducible Windows user test

```powershell
git fetch origin
git switch codex/note-live-harmonic-shadow-expert
git pull --ff-only

python -m scripts.audit_live_vs_offline --wav test_01_E2_seule.wav --expected 40@1.045 --json-out audit01.json > $null
python -m scripts.audit_live_vs_offline --wav test_02_E3_seule.wav --expected 52@1.503 --json-out audit02.json > $null
python -m scripts.audit_live_vs_offline --wav test_03_E2_E3_ensemble.wav --expected 40@1.666 --expected 52@1.666 --json-out audit03.json > $null
python -m scripts.audit_live_vs_offline --wav test_04_E2_puis_E3.wav --expected 40@1.533 --expected 52@2.525 --json-out audit04.json > $null

foreach ($file in @("audit01.json","audit02.json","audit03.json","audit04.json")) {
    $a = Get-Content $file -Raw | ConvertFrom-Json
    Write-Host "=== $file ==="
    $a.expected_events | Format-Table
    $a.pitch_head_gating | Select-Object pitch,marked_onset_s,max_onset_nearby,max_sustain_nearby,frames_with_simultaneous_onset_and_sustain | Format-Table
    $a.notes.short_note_counts | Format-List
    Write-Host ("Same decisions 256/1024: " + $a.identical_decision_events_256_vs_1024)
}

python -m scripts.live_strict_notes --wav test_04_E2_puis_E3.wav --out test04_s613.jsonl
python -m scripts.render_live_midi_offline --events test04_s613.jsonl --soundfont "PATH_TO_EXTRACTED_PIANO.sf2" --out test04_rendered_offline.wav
```

NOTE: Markers ~1.045,1.503,1.666,1.533,2.525 seconds are
approximate from manual audio annotation. Evaluate note timing ±80ms around
markers. A user's actual chronological E3->E2 guitar recording was NOT
supplied among the four files; the supplied sequential example is E2->E3.

For live piano diagnostics, always log output counters on exit via Ctrl+C
(`piano_output_diagnostics`). The counters include callback status
and `same_callback_on_off_pairs`. If `note_on` appears in JSONL but
not in the offline render, investigate silent gates or wrong program/SF2;
if offline render is clean but mic->piano glitches, investigate output callback
underflow, output device and CPU scheduling (without modifying model).

## CI verification

CI workflow `note-live-harmonic-shadow-ci.yml` run 38857026415:
15/15 unit tests passing. The pure-python FakeSynth test reproduced
the note-on/off-before-first-audio rendering pathology. Synthesized E2/E3
WAV was also replayed through the REAL S613 hybrid checkpoint with
256 and 1024 capture-sized blocks; native decisions identical.
This synthetic test emitted **zero note events**, illustrating a
head/threshold silence failure for that limited synthetic stimulus.
It is not a GuitarSet performance estimate and not a validation on user WAVs.

## Decision

- **Do not** relax onset threshold or release threshold based on one
  recording. That can degrade polyphonic K2/K3/K4 and duplicate attacks.
- First separate model missing `note_on` vs rendering failure using the
  two diagnostic scripts. Then audit any changes behind an experimental
  opt-in branch, with latency and real-octave/no-regression gates.
- A different NONCAUSAL offline model may produce different predictions,
  but its scores cannot be inferred from this causal WAV replay.

## Auto queue sizing and unrestricted PortAudio capture block sizes (experimental update)

The earlier CLI accepted only block sizes 128/256/512/1024 and used a
fixed 32-block microphone queue. **That was a CLI/transport policy, not a
limitation inherent to the native NOTE S613 decision hop.**

Now:
- `--block-size N` accepts every **positive integer** supported by the
  underlying PortAudio driver. `--block-size 0` delegates callback buffer
  sizing to PortAudio. No artificial whitelist or maximum is imposed in CLI.
- Internally, packets larger than 4096 samples are split into bounded chunks
  for `CausalPitchFeatures.push`. Existing 256-sample decisions, 8192-sample
  spectral history and 32-frame feature history stay unchanged.
- `--queue-blocks auto` is **the default**. At start, allocate approximately
  75 ms worth of microphone queue slots, at least two packets, e.g.
  256->13, 512->7, 1024->4, 4096->2. `--queue-blocks 16` retains a manual
  override. For PortAudio `--block-size 0`, initial planning uses the 256-sample
  NOTE hop because callback length is driver-dependent; its true queue
  coverage can vary.
- The queue is a maximum pending-input budget, **not an extra required
  wait**. No dynamic unbounded growth, no hidden lookahead, no silent
  dropped frames: callback status/overflow still fails closed.
- A large capture block may materially increase event-delivery latency
  despite a constant 256-sample *model decision* hop. E.g. 8192 samples
  contain 185.76ms audio and cannot be a 5ms responsive microphone callback.
  Large values may also be rejected by the audio driver. These are physical
  latency/driver constraints, not software-chosen size limits.
- `--capture-deadline-blocks` remains opt-in, default 0 (disabled).
- Automatic queue capacity is chosen at startup from audio duration, not
  dynamically stretched to hide a model that runs slower than real-time.

Example:

```powershell
git pull --ff-only
python -m scripts.live_strict_notes --mic --device "MME:External Microphone" --harmonic-mode shadow --block-size 0
# Or choose a specific size; queue remains automatic
python -m scripts.live_strict_notes --mic --device "MME:External Microphone" --harmonic-mode shadow --block-size 512
# Manual override, if needed for diagnosis:
python -m scripts.live_strict_notes --mic --device "MME:External Microphone" --harmonic-mode shadow --block-size 512 --queue-blocks 16
```

This does not solve missed acoustic NoteOns or change FluidSynth output
callbacks by itself. Strict real-time musical latency is still best tested
with small packet sizes and measured device round-trip.
