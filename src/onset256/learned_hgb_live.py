"""Optional research-only Live wrapper for learned HGB arbitration.

Do not attach to official MIDI or any other task head. The underlying direct
acoustic/reference model and all their parameters remain immutable.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from .acoustic_live import load_acoustic_stream
from .learned_hgb_arbiter import DIM, SCHEMA, GatePolicy, LearnedHGBArbiter
from .live_event_stream import OnsetDecision
from .runtime import Intent


class ArbitratedAcousticOnsetStream:
    hop_samples = 256
    added_hold_samples = 0
    future_audio_samples = 0

    def __init__(self, acoustic, gate_model, gate_policy):
        self.acoustic = acoustic
        self.gate = LearnedHGBArbiter(gate_model, gate_policy)
        self.calls = 0
        self.last_reference = None

    def push(self, pcm, *, start_sample):
        direct = self.acoustic.push(pcm, start_sample=start_sample)
        self.last_reference = self.acoustic.last_reference
        if self.acoustic.model is not None:
            evidence = self.acoustic.reference.risk.acoustic_evidence
            q = np.asarray(self.acoustic.model.predict_proba(evidence),
                           np.float32)[:, 1]
        else:
            q = np.zeros(49, np.float32)
            evidence = self.acoustic.reference.risk.acoustic_evidence
        selected, added = self.gate.step(
            q, evidence, (e.pitch for e in self.last_reference.intents),
            (e.pitch for e in direct.intents), end_sample=direct.sample)
        self.calls += 1
        existing = {e.pitch: e for e in direct.intents}
        # Preserve every direct-acoustic Intent exactly, including source.
        intents = tuple(existing[p] if p in existing else Intent(
            "note_on", p, direct.sample, float(q[p-40]),
            "learned_hgb_gate_pcm256") for p in selected)
        return OnsetDecision(direct.sample, intents)

    def finish(self):
        return self.acoustic.finish()


def load_arbitrated_stream(experiment, gate_dir, *, fold,
                           base_bundle, recovery_bundle, native_path):
    gate_dir = Path(gate_dir)
    meta = json.loads((gate_dir / "frozen_gate.json").read_text())
    if (meta["schema"] != SCHEMA or meta["feature_dim"] != DIM or
            meta["trained_hop_samples"] != 256 or
            meta["future_audio_samples"] != 0 or
            meta["added_hold_samples"] != 0 or
            meta["outer_composition"] != fold or
            fold in meta["train_compositions"] or
            fold == meta["cal_composition"] or
            not meta["frozen_before_outer"]):
        raise ValueError("learned gate lineage/causal contract mismatch")
    checkpoint = gate_dir / "hgb_arbiter.joblib"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != meta["checkpoint_sha256"]:
        raise ValueError("learned gate checkpoint SHA mismatch")
    gate_model = joblib.load(checkpoint) if meta["selected"]["enabled"] else None
    gate_policy = (GatePolicy(**meta["selected"]["policy"])
                   if meta["selected"]["enabled"] else GatePolicy())
    acoustic = load_acoustic_stream(
        Path(experiment) / "trained" / fold, fold=fold,
        base_bundle=base_bundle, recovery_bundle=recovery_bundle,
        native_path=native_path)
    return ArbitratedAcousticOnsetStream(acoustic, gate_model, gate_policy)
