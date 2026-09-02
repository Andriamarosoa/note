"""V11.1 strict-OOF pilot with a genuinely hierarchical decoder.

V11 trained separate birth and positive-multiplicity heads, but decoded them by
reconstructing P(K=0..6) and taking one flat argmax. That gives K0 an unfair
structural advantage because positive birth mass is split across K1..K6.

V11.1 keeps the exact same training/model family and changes only the decision
rule to match the hierarchy:

    birth < 0.5  -> K = 0
    birth >= 0.5 -> K = 1 + argmax Q(K | birth)

The 0.5 binary decision is fixed a priori and is not tuned on any held fold,
historical validation, or locked12.
"""
from __future__ import annotations

import json
import numpy as np

from causal_note.guitarset import SLOT_COUNT
from scripts import train_v11_hierarchical_birth_multiplicity as v11


def _hierarchical_probabilities(model, features, anchor, p102):
    """Return a normalized score matrix whose argmax is the hierarchical decision."""
    out = model.predict(v11._inputs(features, anchor, p102), batch_size=256, verbose=0)
    birth = np.asarray(out["birth"], dtype=np.float64).reshape(-1)
    q = np.asarray(out["multiplicity"], dtype=np.float64)
    if q.shape != (len(birth), SLOT_COUNT):
        raise RuntimeError(f"unexpected multiplicity shape {q.shape}")

    positive = birth >= 0.5
    p = np.zeros((len(birth), SLOT_COUNT + 1), dtype=np.float64)
    p[~positive, 0] = 1.0
    p[positive, 1:] = q[positive]
    # Softmax already normalizes q, but normalize defensively for report hygiene.
    p /= np.maximum(np.sum(p, axis=1, keepdims=True), 1e-12)
    return p


def main(argv=None):
    args = v11.parser().parse_args(argv)
    original = v11._probabilities
    v11._probabilities = _hierarchical_probabilities
    try:
        result = v11.run(args)
    finally:
        v11._probabilities = original

    result["schema_version"] = 2
    result["protocol"]["decoder_threshold_tuned"] = False
    result["protocol"]["birth_decision_threshold_fixed"] = 0.5
    result["architecture"]["training_factorization"] = result["architecture"].pop("factorization")
    result["architecture"]["decoder"] = "if P_birth < 0.5: K=0; else K=1+argmax Q(K|birth)"
    result["architecture"]["decoder_name"] = "strict_hierarchical_050"
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
