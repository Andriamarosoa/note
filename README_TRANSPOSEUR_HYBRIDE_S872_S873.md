# NOTE — S872-S873 hybrid down-octave: standard DSP corrected by neural GRU

Date: 2026-10-24. Status: EXPERIMENTAL, NOT PROMOTED. Six official NOTE heads and MIDI untouched.

User goal: convert highs to lows by exactly -12 semitones. S870 at the **end** of S871 audio (14.8–18.3 s) fails strongly, drifting to 24–50 Hz with mean DC offset +0.0213 on its original 3.5-second recording. We now keep conventional octave-down DSP as the authoritative audio source. The AI only learns a continuously smoothed mixture of two independently pitch-shifted standard streams; it does not produce pitches from scratch. Exact DSP fallback available.

## Audited results on real S847 GuitarSet recordings

39 real captured note excerpts: TRAIN 27, CAL 6, AUDITED 6. The AUDITED group was already seen in previous experiment cycles, therefore DO NOT describe these percentages as new independent generalization.

| DSP grain samples | Calibration correctly pitched ±0.5 semitone | Six previously examined notes |
|---|---:|---:|
| 1024 | 1/6 | 0/6 |
| 2048 | 2/6 | 2/6 |
| 4096 | 5/6 | 6/6 |
| 8192 | 6/6 | 6/6 |

The offline librosa -12 reference is used only for training and scoring, NEVER during streaming.

| Variant | CAL correct baseline→IA | Previously examined correct baseline→IA | CAL median log-STFT magnitude distance baseline→IA (lower better) | Previously examined log-STFT distance baseline→IA |
|---|---|---|---|---|
| S872: 2048+4096 DSP + neural | 5/6→5/6 | 6/6→6/6 | 0.13045→0.12627 (-3.2%) | 0.15104→0.13822 |
| S873: 4096+8192 DSP + neural | 6/6→6/6 | 6/6→6/6 | 0.16548→0.14258 (-13.8%) | 0.20692→0.17657 |

The improvement percentages are percentage reductions in spectral DISTANCE, NOT percentage success at audio transposition. Neither variant has passed a proper unseen polyphonic musical evaluation. No promotion to NOTE.

## Timing and strict hop contract

Exactly 256 NEW PCM samples per decision at 44100 Hz (~5.805 ms), past samples only, no future input, no extra hold. A GRANULAR pitch shifter introduces NONZERO and phase-varying acoustic delay, even though it emits each 256-output hop on time. Estimated mean delay from delay trajectories: S872 quality grain 4096 ~24.7 ms, S873 quality grain 8192 ~47.9 ms; impulsive probes show phase-dependent differences. These delay numbers are NOT measured on Windows/iRig.

Local CPU-only 220-hop p99: S872 neural ~1.48 ms; S873 neural ~2.13 ms; compare 5.805 ms available per hop. Hardware timing, audio roundtrip latency, actual physical acoustic latency, clicks, timbre preservation, and performance on unseen chords remain unvalidated.

## Reproduction and explicit fallback

Source: research/transpose256/s872_hybrid_standard_ai256.py
Tests: research/transpose256/test_s872_hybrid_standard_ai256.py
The chat file NOTE_S872_S873_HYBRID_STANDARD_AI_MINUS12_PCM256.zip contains authentic trained checkpoints, full local training scripts, WAVs and JSON audits. Github contains engine and tests only; structural CI does not re-train models or validate scores.

Use: python s872_hybrid_standard_ai256.py --checkpoint s873_neural_correction.pt --input guitar.wav --output corrected.wav
Disable AI: add --no-ai for exactly the same standard grain 8192 waveform. The fallback is *user-controlled* in the engine, not a magical per-hop correctness predictor. The CAL-based guard was evaluated offline; potential new types of regression still require tests.

Research context: Signalsmith Stretch (https://github.com/Signalsmith-Audio/signalsmith-stretch) offers well-defined streaming transpose and inputLatency/outputLatency APIs; Rubber Band realtime (https://www.breakfastquay.com/rubberband/integration.html) also documents nonzero algorithmic delay. Both are future baseline candidates if their delay meets practical requirements.

See series/S872/resume.md, series/S872/resultat.md, series/S873/resume.md, series/S873/resultat.md.
