# Strict native PCM256 Active-K compatibility audit

This branch is an **experimental** copy of codex/note-best-active-exact-k-s828-256.
Its decoder accepts only exact 256-sample decision hops and rejects 1024-sample
or skipped decision timestamps. The reference branch is preserved unchanged.

- 44,100 Hz PCM input, decisions every 256 samples (5.804988662 ms cadence).
- No extra hold/lookahead and no production MIDI modification.
- The frozen S708 temporal model was historically trained with 1024-sample
  decision hops. Running it each 256 samples does **not** remove learned
  1024-rate temporal dependencies. Fully removing those requires training and
  validating a fresh model at 256 on composition-disjoint folds.
- The preserved S828 FIT80 baseline is 184143 correct / 249411 active frames.
- The signed 80-track scientific score must be measured again; smoke tests
  alone cannot justify promotion. No generalization to external GuitarSet/IDMT
  subsets is inferred from explored FIT80.
- The historical S708/H0 reference branches and all other axis heads are
  untouched.
