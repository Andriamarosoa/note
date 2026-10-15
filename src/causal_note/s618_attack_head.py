"""S618 native causal attack-evidence head — strictly an auxiliary NOTE head.

The original S614 model and its Exact-K ancestor S80 stay byte-identical.
The new head sees only the old acoustic/NN probabilities at the current or
past frame. No S601, midi repo, labels, current/future annotations at inference.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from causal_note.strict_live import CausalSelection, Selection, HOP

# Explicit native S613 schema column IDs. Changing them changes the ABI.
PICK = np.asarray([
    0, 1, 2, 8, 9, 10, 16, 17, 18,
    24, 32, 40, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65,
    78, 79, 80, 81, 82, 83,
    84, 85, 86, 87, 88, 89,
    90, 91, 92, 93, 94, 95,
    102, 103, 104, 105, 106, 107,
    114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125
], dtype=int)
DIM = 3 + 3 + len(PICK)


def input_features(x, probs):
    """(T,49,126)+(T,49,3) -> (T,49,DIM), strictly right-aligned."""
    x = np.asarray(x, np.float32)
    probs = np.asarray(probs, np.float32)
    if x.ndim != 3 or x.shape[1:] != (49,126) or probs.shape != (*x.shape[:2],3):
        raise ValueError("Only the fixed original causal S614 49x126/49x3 interface")
    if not (np.isfinite(x).all() and np.isfinite(probs).all()):
        raise ValueError("Non-finite acoustic / attack probabilities")
    previous = np.concatenate((np.zeros_like(probs[:1]), probs[:-1]), axis=0)
    change = probs - previous
    return np.concatenate((probs, change, x[:,:,PICK]), axis=-1).astype(np.float32)


def candidate_mask(x, probs):
    """A *causal*, rather permissive candidate generator; evaluated on DEV."""
    return ((probs[:,:,0] >= .08) |
            (probs[:,:,1] >= .20) |
            (x[:,:, [57,58,59]].max(axis=-1) >= .12))


@dataclass(frozen=True)
class TrainedPolicy:
    veto_below: float = -1.
    rescue_above: float = 2.

    def __post_init__(self):
        if not (-1. <= self.veto_below <= 1.) or not (0. <= self.rescue_above <= 2.):
            raise ValueError("Invalid trained policy probability thresholds")


class NativeLearnedSelection:
    def __init__(self, config:Selection, policy:TrainedPolicy):
        self.selector = CausalSelection(config)
        self.policy = policy
        self.changed = 0

    def step(self,p, k, q, acoustics, end):
        p = np.asarray(p, np.float32)
        q = np.asarray(q, np.float32)
        x = np.asarray(acoustics, np.float32)
        if p.shape != (49,3) or q.shape!=(49,) or x.shape!=(49,126):
            raise ValueError("Native head input shape mismatch")
        if not (np.isfinite(q).all() and np.all((q>=0) & (q<=1))):
            raise ValueError("Invalid trained head probabilities")
        edited=p.copy()
        currently_off = ~self.selector.active
        if self.policy.veto_below >= 0:
            block = currently_off & (p[:,0] >= self.selector.config.onset) & (
                q < self.policy.veto_below)
            edited[block,0]=0.
            self.changed+=int(block.sum())
        if self.policy.rescue_above <= 1:
            candidate=currently_off & (q >= self.policy.rescue_above) & (
                (p[:,0]>=.08) | (x[:,[57,58,59]].max(axis=-1)>=.12))
            # Distinct from S617's unconditional onset bridge: require
            # learned acoustically classified event.
            edited[candidate,0]=np.maximum(edited[candidate,0],self.selector.config.onset)
            edited[candidate,1]=np.maximum(edited[candidate,1],self.selector.config.active_on)
            self.changed+=int(candidate.sum())
        return self.selector.step(edited,k,int(end),int(end))

    def finish(self,sample):
        return self.selector.panic(int(sample))


def variants():
    return [
        ("S614_reference", TrainedPolicy()),
        ("veto_020",TrainedPolicy(veto_below=.20)),
        ("veto_040",TrainedPolicy(veto_below=.40)),
        ("veto_060",TrainedPolicy(veto_below=.60)),
        ("rescue_065",TrainedPolicy(rescue_above=.65)),
        ("rescue_080",TrainedPolicy(rescue_above=.80)),
        ("rescue_090",TrainedPolicy(rescue_above=.90)),
        ("rescue_095",TrainedPolicy(rescue_above=.95)),
        ("joint_020_080",TrainedPolicy(veto_below=.20,rescue_above=.80)),
        ("joint_040_090",TrainedPolicy(veto_below=.40,rescue_above=.90)),
        ("joint_060_095",TrainedPolicy(veto_below=.60,rescue_above=.95)),
    ]
