"""Onset-only compiled runtime: consume exactly one current block of 256."""
from dataclasses import dataclass
import numpy as np

from .acoustics import HOP
from .compiled_inference import accelerated, SharedEvidenceRisk, SharedEvidenceRecovery
from .event_confidence import EventConfidenceHead
from .runtime import Stream


@dataclass(frozen=True)
class OnsetDecision:
    sample: int
    intents: tuple


class LowLatencyOnsetStream:
    """No pending attack queue, confirmation hop or timestamp compensation.

    All compilation happens during construction, before audio ingestion.
    This output is private Onset; it does not implement NoteOff or MIDI.
    """
    added_hold_samples = 0
    hop_samples = 256

    def __init__(self, native, s701, risk, recovery, event_head):
        self.risk = SharedEvidenceRisk(risk)
        self.recovery = SharedEvidenceRecovery(recovery, self.risk)
        self.stream = Stream(native, accelerated(s701), risk_model=self.risk,
                             recovery_model=self.recovery)
        self.head = EventConfidenceHead(accelerated(event_head.model),
                                         cut=event_head.cut,
                                         selected_model=event_head.selected_model)

    def push(self, samples, *, start_sample):
        x = np.asarray(samples)
        if (x.shape != (HOP,) or type(start_sample) is not int
                or start_sample % HOP or x.dtype.kind != "f"):
            raise ValueError("exactly 256 new float samples on the PCM256 grid required")
        decisions = self.stream.push(x, start_sample=start_sample)
        if len(decisions) != 1:
            raise AssertionError("missing current-hop decision")
        d = decisions[0]
        if d.sample != start_sample + HOP:
            raise AssertionError("delayed or backdated output")
        q = self.head.predict(self.risk.evidence, sample=d.sample)
        kept = self.head.select(d.predecision_intents, q, sample=d.sample)
        return OnsetDecision(d.sample, kept + d.fn_recovery_intents)

    def finish(self):
        self.stream.finish()
        return ()


def load_research_stream(bundle, *, fold, native_path):
    """Load a preserved research fold with signed project-owned checkpoints.

    The fold is explicit for research reproduction. These five out-of-fold
    models are not a single deployed production model.
    """
    import hashlib
    import json
    from pathlib import Path
    from .error_evidence69 import load_hard_model
    from .event_confidence import load_event_head
    from .fn_recovery import load_fn_recovery
    from .model import load

    root = Path(bundle)
    manifest = json.loads((root / "manifest.json").read_text())
    if fold not in manifest["folds"] or manifest["hop_samples"] != HOP:
        raise ValueError("unknown research fold or hop")
    for name, expected in manifest["files_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("research bundle SHA mismatch: " + name)

    def signed(prefix, loader):
        path = root / f"{prefix}_{fold}.joblib"
        meta = json.loads(path.with_suffix(".json").read_text())
        if (meta["outer_composition"] != fold or fold in meta["train_compositions"]
                or fold == meta["cal_composition"]):
            raise ValueError("research fold contamination")
        return loader(path, path.with_suffix(".json"),
                      expected_sha256=manifest["files_sha256"][path.name])

    native, s701 = load(native_path, root / "s701.joblib")
    risk = signed("risk", load_hard_model)
    recovery = signed("recovery", load_fn_recovery)
    head = load_event_head(root / f"selected_{fold}.joblib",
                           root / f"selected_{fold}.json", outer=fold)
    return LowLatencyOnsetStream(native, s701, risk, recovery, head)
