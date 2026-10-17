# S657 — Selective causal Tone octave gate, controlled source validation

Successful CI: https://github.com/Andriamarosoa/note/actions/runs/39008335174

S657 tests an improvement to S656: leave all original NOTE
predictions unchanged unless a **native upper note** has onset
confidence now or within the previous 4 or 8 hops. Then allow
a lower-octave candidate from the digitally filtered Tone view
when native lower onset has not already fired.

No pitch transposition, no added future audio, no changes to
original NOTE S614/Exact-K. This test contains 8 pairs of
original isolated GuitarSet TRAIN E3 and E4 WAV; the duo is a
digitally generated juxtaposition of separately performed
real notes, **not a naturally recorded polyphonic guitar chord**.

## Actual 8-pair NoteOn counts

| Condition | Native S614 | Gate 4-hop, Tone minimum .50 |
|---|---|---|
| E3 alone — target E3 TP | 8/8 | 8/8 |
| E4 alone — target E4 TP | 8/8 | 8/8 |
| E3+E4 simultaneous — E3 correct | 6/8 | **8/8** |
| E3+E4 simultaneous — E4 correct | 8/8 | 8/8 |
| E3 then E4 120ms later — E4 correct | 5/8 | 5/8 |
| E4 alone — *false emitted E3* | 0 | **0** |

On E3+E4 simultaneous, full event scores:
- S614: TP 14 / FP 61 / FN 2;
- S657 4-hop: **TP 16 / FP 61 / FN 0**.
The remaining 61 false note attacks are significant.
This is **not 100% NoteOn F1**, only 16/16 recall
for the paired two annotated target pitches. Other pitch
attacks are emitted and judged FP against the paired test.

On delayed E3 followed by E4, S657 retains original
TP 13 / FP 56 / FN 3 (same as S614), whereas full dark Tone
in S656 lost three upper E4 detections.

The five predeclared gates:
- same-hop high, filter threshold 0.50;
- last 4 hops high, threshold 0.50;
- last 8 hops high, threshold 0.50;
- last 4 hops high, threshold 0.70;
- last 8 hops high, threshold 0.70.
The 4- and 8-hop 0.50 rules produced 16/16 target TP;
the high-threshold variants recovered only 15/16.

**No promotion based on these controlled examples.**
The next valid experiment is **S658: natural whole-song transfer**,
testing the same fixed rules on 36 TRAIN and 12 DEV real GuitarSet
recordings, where other notes, real attacks and polyphony can
produce new false lower octaves.
