"""Pure V8 target conversion helpers."""
from typing import Dict, List, Mapping, Sequence

COUNT_CLASSES = 4


def slot_targets_to_count_classes(
    slot_targets: Sequence[Sequence[float]],
) -> List[int]:
    """Collapse per-slot binary labels to anonymous count classes 0/1/2/3+."""

    result = []
    for row in slot_targets:
        count = sum(1 for value in row if float(value) > 0.0)
        result.append(min(count, COUNT_CLASSES - 1))
    return result


def collapse_boundary_targets(
    targets: Mapping[str, Sequence[Sequence[float]]],
) -> Dict[str, List[int]]:
    """Convert historical six-slot targets to V8 anonymous cardinalities."""

    if "onset" not in targets or "offset" not in targets:
        raise ValueError("targets must contain onset and offset")
    if len(targets["onset"]) != len(targets["offset"]):
        raise ValueError("onset and offset temporal lengths must match")
    return {
        "onset_count": slot_targets_to_count_classes(targets["onset"]),
        "offset_count": slot_targets_to_count_classes(targets["offset"]),
    }


__all__ = [
    "COUNT_CLASSES",
    "collapse_boundary_targets",
    "slot_targets_to_count_classes",
]
