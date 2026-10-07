# Verdict — YourMT3+ representation ablation for Exact-K

Run: [37649776590](https://github.com/Andriamarosoa/note/actions/runs/37649776590)

Source full-decoder comparison: run [37605163312](https://github.com/Andriamarosoa/note/actions/runs/37605163312).

## Protocol

The same 59,309 native Exact-K groups and folds 0,1,2,4 were used. Fold 3 and player05
remain excluded. The pinned official YourMT3+ checkpoint is unchanged.

Two frozen intermediate representations were extracted without decoding notes and without
using labels during extraction. A fixed linear probe was trained on three folds and evaluated
on the fourth, rotating over all four folds. No held-out hyperparameter selection was used.

Measured checkpoint shapes:

- spectrogram: 110 x 1024
- convolutional pre-encoder: 110 x 128 x 128
- Perceiver-TF/MoE encoder: 110 x 26 x 128
- 13-channel pre-decoder projection: 13 x 110 x 512

The local probe summary uses +/-2 model frames around each native group, approximately
matching the native ~93 ms group scale while the frozen encoder still sees the full ~2.048 s
YourMT3+ segment.

## Aggregate result

| Stage | Global Exact-K | Poly K2-K6 | Poly under | Poly over |
|---|---:|---:|---:|---:|
| freeze_local_combo | 81.6976% | 34.2586% | 3630 | 1225 |
| encoder linear probe | 78.4316% | 24.4008% | 3303 | 2280 |
| pre-decoder linear probe | 78.4198% | 21.8009% | 3710 | 2065 |
| full YourMT3+ decoder | 86.5434% | 54.5430% | 2205 | 1152 |

Per-K:

| K | freeze | encoder probe | pre-decoder probe | full decoder |
|---:|---:|---:|---:|---:|
| 0 | 95.922% | 91.461% | 93.138% | 95.420% |
| 1 | 64.285% | 68.848% | 64.936% | 77.119% |
| 2 | 34.340% | 22.903% | 17.097% | 52.772% |
| 3 | 39.974% | 21.669% | 24.596% | 54.696% |
| 4 | 34.548% | 30.986% | 26.761% | 60.066% |
| 5 | 4.225% | 30.704% | 30.141% | 53.239% |
| 6 | 0.000% | 38.202% | 31.461% | 49.438% |

## Interpretation

The large Exact-K gain does not appear as a linearly readable count signal in the local frozen
encoder or in the 13-channel projection. In this diagnostic, the major jump appears only after
the autoregressive Multi-T5 event decoder.

This does **not** prove that the encoder is unimportant. The encoder probe compresses the
intermediate tensors and is deliberately linear. In particular, K5/K6 improve strongly already
at the encoder level, showing that high-cardinality information is present. The dominant K2/K3
classes, however, remain poor until full event decoding.

The next decisive ablation should therefore separate **nonlinearity** from **autoregression**:

1. frozen encoder + small nonlinear MLP count head;
2. frozen encoder + anonymous autoregressive count/event decoder, with no pitch or instrument
   identity;
3. full YourMT3+ decoder as the upper diagnostic reference.

If the MLP remains near the linear probe while the anonymous autoregressive decoder rises
substantially, sequence/event decoding is the transferable mechanism we should bring into
`note` before note identification.

No model is promoted by this experiment. The YourMT3+ comparison remains exploratory because
pretraining overlap with GuitarSet has not been excluded.
