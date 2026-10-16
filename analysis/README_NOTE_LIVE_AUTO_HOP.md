# NOTE S613 — adaptive decision hop experiment (NO PROMOTION)

Branch: `codex/note-live-harmonic-shadow-expert`. Base/reference:
`codex/note-live-audio-monitor-wasapi` (unchanged). Date: 2026-10-16.

## Three DIFFERENT time controls

- `--block-size`: PortAudio capture packet length (any positive integer, or
  `0` to let the driver choose callback frames). Large blocks increase
  **actual event delivery** latency even if a model computes frequent decisions
  inside the packet.
- `--hop-size`: the NOTE model's internal acoustic decision interval,
  **128 = 2.9025ms**, **256 = 5.8050ms**, **512 = 11.610ms** at 44.1kHz.
- `--queue-blocks`: input FIFO MAXIMUM capacity, default `auto` estimated
  from ~75ms raw PCM; capacity itself is not mandatory buffering delay.

S613 weights were trained/validated **only for a 256-sample hop**. Manual
128/512 and `auto` are experimental and can alter predictions, including
Exact-K, F1, and attack/offset timing. **256 is the default**, preserving
historical NOTE behavior.

## What is implemented

`src/causal_note/strict_live.py` accepts hop=128/256/512 in
`StrictLiveStream`, `CausalPitchFeatures`, `CausalSelection`. The preexisting
8192/4096/2048 causal FFT lengths, model heads, learned weights and six-note
limit remain unchanged. Temporal-history lags (2,4,8,16 at 256),
8- and 32-frame aggregations, and release/confirmation frame counts are
mapped to approximate the **same time in milliseconds** at other hops. The
spectral flux interval itself still differs by hop; this is **not equivalent**
to training/validating another hop model. Refractory period keeps its
existing absolute sample duration.

`scripts/live_hop_tuning.py` implements preflight CPU benchmarking of
128/256/512 on a short fixed, synthetic harmonic signal using the real loaded
S613 heads and optional expert `shadow`. It picks the **smallest hop** whose
measured p95 compute time is at most 55% of its nominal audio period, else
falls back to 512. Selection is performed **once before microphone capture**
to avoid corrupting state mid-note. A fallback 512 is not guaranteed to
eliminate overload. No online audio samples or future frames enter this
selection; no post-onset waiting, no new model architecture.

**Auto optimizes CPU headroom, NOT musical accuracy.** No new model F1,
Exact-K score or mic-to-piano roundtrip has been obtained for any other hop.

## Commands in Windows PowerShell

Stay in experimental branch:
```powershell
git fetch origin
git switch codex/note-live-harmonic-shadow-expert
git pull --ff-only
python -m unittest -v test.test_live_adaptive_hop
```

Original validated reference, now default:
```powershell
python -m scripts.live_strict_notes --mic --device "MME:External Microphone" --block-size 256 --soundfont "PATH_TO_PIANO.sf2"
```

Explicit opt-in CPU automatic decision hop (block size independent):
```powershell
python -m scripts.live_strict_notes --mic --device "MME:External Microphone" --block-size 256 --hop-size auto --harmonic-mode shadow --soundfont "PATH_TO_PIANO.sf2"
```

Manual ablation:
```powershell
python -m scripts.live_strict_notes --wav "test_04_E2_puis_E3.wav" --hop-size 128 --out "events_128.jsonl"
python -m scripts.live_strict_notes --wav "test_04_E2_puis_E3.wav" --hop-size 256 --out "events_256.jsonl"
python -m scripts.live_strict_notes --wav "test_04_E2_puis_E3.wav" --hop-size 512 --out "events_512.jsonl"
```

The CLI reports `hop_selection` JSON on stderr (preflight p95 values or
manual mode), and mic startup reports `decision_hop_samples` and
`decision_hop_ms`. Fresh `--out` filenames are required (no overwrite).

## Measured tests

GitHub Actions run 38859186111 (commit 642ad7ba):
**27 safety/unit tests passed** and the real S613 hybrid checkpoint
executed synthetic E2/E3 at hops 128, 256, 512 and auto. The locked/default
256 and explicit `--hop-size 256` JSONL files were byte-identical.
Future-audio mutation invariant passed for 128/256/512.

With a limited synthetic control audio:
- 128 produced **2 note events**;
- 256 produced **0**;
- 512 produced **0**.
These differences are NEITHER demonstrated corrections NOR measured errors
because the synthetic stimulus has no independently scored ground-truth
guitar NoteOn annotations. They are direct evidence that changing hop alters
model output even when weights and architecture remain fixed.

On Linux CI, explicit `--hop-size auto` selected 128 with measured
per-inference p95 approx. 1.48/1.44/1.43ms at 128/256/512 respectively.
Your Windows hardware may choose differently. Measured p95 is not real
microphone or FluidSynth output latency. Tested with CI, not yet proven
accurate on GuitarSet, URMP, IDMT or target iRig.

## Promotion gate

Before making 128/512/auto a new official default:
- Replay independent labeled GuitarSet/URMP/IDMT clips and four iRig controls
  with the **same trained checkpoint** at 128,256,512.
- Track NoteOn F1 at ±50/80ms, Exact-K onset cardinality, K2–K6, short
  `note_off`, onset/offset error, and missed genuine E2+E3 notes.
- Evaluate microphone+FluidSynth p95 compute and round-trip on Windows.
- Promote only after quantified gain with no unacceptable regression.
