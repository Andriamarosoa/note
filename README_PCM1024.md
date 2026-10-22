# Frame Pitch experimental PCM1024
One fresh, strictly causal 44.1 kHz mono PCM acoustic and temporal inference per 1024 samples (23.21995465 ms). This fork does not modify the reference PCM256 branch or MIDI production output.
The original frozen S708 neural weights were trained using a historical 1024 hop; the native acoustic frontend has its own frozen training provenance and is NOT retrained in this change.
FIT80 signed 80-track measurements use the 1024 decision grid; no equivalent 256 frame denominator is assumed. Historic S834 PCM256 F1 is 84.998850% (467540 TP, 71501 FP, 93528 FN), retained as descriptive, non-comparable metadata. A completed run produces a separate PCM1024 metric; no promotion or performance claim until logs are reviewed.
