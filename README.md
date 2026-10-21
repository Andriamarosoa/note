# Active Exact-K ONLY, S828 true native PCM256

This Git branch contains only Active Exact-K source/readout, including its own exact S827 guided K state. It has NO pipelines/decoders for Exact-K, Onset, Offset, Frame Pitch, Full Note. Original frozen S708 is used as an acoustic backbone (originally pretrained 1024), but the S828 private HGB readout is trained and CAL-selected on each FIT80 fold at true native PCM256, not at 1024.

Signed S828 benchmark to reproduce exactly: 184143 / 249411 active frames = 73.831146% (80 GuitarSet FIT80 tracks), TRAIN 48 / CAL 16 / OUTER 16 tracks times five composition splits. This score is SHADOW ActiveK only; original MIDI remains unchanged. If the exact expected count fails in the trimmed replayer, declare revalidation failed, not promoted.
