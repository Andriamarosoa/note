"""Pure V8 target helpers."""
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence


MAX_EXACT_COUNT = 3


@dataclass(frozen=True)
class HierarchicalCountTarget:
    presence: int
    multiplicity_class: int

    def __post_init__(self) -> None:
        if self.presence not in (0, 1):
            raise ValueError("presence must be 0 or 1")
        if self.multiplicity_class not in (0, 1, 2):
            raise ValueError("multiplicity_class must be 0, 1 or 2")


def exact_count_to_hierarchical(count: int) -> HierarchicalCountTarget:
    """Map exact count 0/1/2/3+ to presence plus conditional multiplicity."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be an integer >= 0")
    if count == 0:
        # Multiplicity is ignored by zero sample weight when presence == 0.
        return HierarchicalCountTarget(0, 0)
    return HierarchicalCountTarget(1, min(count, MAX_EXACT_COUNT) - 1)


def slot_targets_to_exact_counts(
    slot_targets: Sequence[Sequence[float]],
) -> List[int]:
    """Collapse historical six-slot binary rows to anonymous exact counts."""

    return [
        sum(1 for value in row if float(value) > 0.0)
        for row in slot_targets
    ]


def collapse_boundary_targets(
    targets: Mapping[str, Sequence[Sequence[float]]],
) -> Dict[str, List[int]]:
    """Convert historical slot targets to exact anonymous count sequences."""

    if "onset" not in targets or "offset" not in targets:
        raise ValueError("targets must contain onset and offset")
    if len(targets["onset"]) != len(targets["offset"]):
        raise ValueError("onset and offset temporal lengths must match")
    return {
        "onset_count": slot_targets_to_exact_counts(targets["onset"]),
        "offset_count": slot_targets_to_exact_counts(targets["offset"]),
    }


__all__ = [
    "HierarchicalCountTarget",
    "MAX_EXACT_COUNT",
    "collapse_boundary_targets",
    "exact_count_to_hierarchical",
    "slot_targets_to_exact_counts",
]
