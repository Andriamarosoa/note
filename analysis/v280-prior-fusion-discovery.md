# V28.0-H — discovery on frozen G outputs

Exploratory results on G's already-selected internal checkpoints, not independent confirmation.

| Method | Poly Exact-K | Global Exact-K | False polyphonic rows | Poly NLL |
|---|---:|---:|---:|---:|
| control | 38.4009% | 77.8587% | 839 | 1.61843 |
| balanced | 52.1959% | 68.5165% | 2877 | 1.19616 |
| inverse_weights | 37.4437% | 76.7374% | 881 | 1.58262 |
| soft_hierarchical | 39.1329% | 77.8373% | 848 | 1.60362 |
| mixture_half | 39.5833% | 78.0659% | 799 | 1.57179 |
| prior_alpha_0.25 | 50.7320% | 71.6306% | 2271 | 1.25850 |
| prior_alpha_0.5 | 48.1982% | 74.0376% | 1725 | 1.34060 |
| prior_alpha_0.75 | 43.9189% | 75.7017% | 1269 | 1.44708 |
| hard_routed | 39.1892% | 77.9587% | 839 | n/a |

Frozen candidate: mixture_half; equal fusion after full inverse-weight correction.
Poly delta +1.1824 pp; global delta +0.2071 pp.
Descriptive paired track interval: [-0.5510, 2.9889] pp.
Nine candidates were examined; the interval is not corrected for selection.
Fusion uses two 110402-parameter networks (220804 total), with two forward passes.
H freezes this method and epoch 2 before evaluating a second internal split. V27.3 remains the reference.
