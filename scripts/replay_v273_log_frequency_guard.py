"""Replay the two frozen linear/log-frequency guard arms without audio or fitting."""
import argparse
import json
from pathlib import Path

from scripts.audit_v273_log_frequency_guard import NAMES
from scripts.replay_v273_harmonic_decay_guard import verify
from scripts.v273_residual_audit import write_json


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = verify(a.input, NAMES)
    if a.output:
        write_json(a.output, result)
    print(json.dumps(result, sort_keys=True))
