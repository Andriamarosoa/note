# S633 — Bass + High-Pitch Rescue in strictly causal NOTE LIVE

Status: **EXPERIMENTAL, NOT PROMOTED**. The frozen S614/S613 model,
native pitch heads, K0..6 output, and S630 inference batching are unchanged.

## Audit evidence

S632 real DEV audit: 17 historically viewed compositions (12 GuitarSet,
5 URMP), same pinned frozen S614 checkpoint and original native frontend;
22 predetermined causal candidate policies were compared. No held-out TEST
labels were read for this experiment. Therefore no independent generalization
claim.

| Corpus | Reference F1 | S632 bass_k50_high_moderate | TP reference/new | FP reference/new |
| --- | --- | --- | --- | --- |
| GuitarSet | 0.631046 | 0.633948 | 998/1014 | 585/605 |
| URMP | 0.545455 | 0.548458 | 243/249 | 149/160 |

HIGH-MODERATE alone: URMP E5+ matched 16/62 -> 21/62,
with high-register FP 8 -> 9, and overlapping E5+ 14/57 -> 19/57.
On GuitarSet DEV, only one true E5+ event is present, so external evidence
there is very weak.

Bass rescue remains more false-positive prone than high rescue:
with both selected experts GuitarSet +16 TP / +20 FP, URMP +6 TP / +11 FP.
This small F1 gain does not justify an automatic official promotion.

S632 evidence workflow:
https://github.com/Andriamarosoa/note/actions/runs/38905283363

## Native/microphone LIVE integration

S633 supports both `StrictLiveStream` (batch 1, 0ms) and
`BatchedLiveStream` (batch4, 0ms; shared lookahead allowed with **shadow** only).
LIVE flags are opt-in:

- `--register-rescue-mode off` (default; no change to NOTE);
- `--register-rescue-mode shadow` (evaluate additional candidates,
  **without changing audible NOTE-on/off**);
- `--register-rescue-mode guard` (EXPERIMENT: proposed extra NOTE-on,
  can create false notes).
- `--register-rescue-policy high_moderate` (default when a mode is selected)
  or `bass_k50_high_moderate` (both register experts).

Guard **requires** `--lookahead-ms 0`. No dual-hop, harmonic sidecar, or
untrained hop128. The extra score checks consume CPU, so microphone queue
stability must be separately profiled on real Windows hardware.

### Commands: first shadow, then optional guard

Run from the S633 branch. First record shadow candidates (SoundFont outputs
still match the original):
```powershell
python -m scripts.live_strict_notes --mic --soundfont ".\SalamanderGrandPiano-V3+20200602.sf2" --hop-size 256 --block-size 256 --lookahead-ms 0 --inference-batch-hops 4 --register-rescue-mode shadow --register-rescue-policy bass_k50_high_moderate
```

Then, if evaluating the experimental additional notes:
```powershell
python -m scripts.live_strict_notes --mic --soundfont ".\SalamanderGrandPiano-V3+20200602.sf2" --hop-size 256 --block-size 256 --lookahead-ms 0 --inference-batch-hops 4 --register-rescue-mode guard --register-rescue-policy bass_k50_high_moderate
```

Live run emits `s633_register_rescue` settings on startup and candidate
counts in shutdown diagnostics. These counts are **proposals**, not
validated true positive counts, without microphone annotations.

S633 real CLI/native/batch integration unit tests:
https://github.com/Andriamarosoa/note/actions/runs/38906711206
Both Windows and Linux succeeded. The hardware microphone/PortAudio and
actual user's external SF2 are not exercised by CI.

Next validation gate: novel independent multi-source recordings, blind
annotations and measured real microphone CPU + end-to-end latency. Do not
overwrite S614 or promote because DEV F1 moves by fractions of a percent.
