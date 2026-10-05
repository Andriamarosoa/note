"""Verify a residual export without TensorFlow, audio, or a new model fit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from scripts.v273_residual_audit import verify_export


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(verify_export(a.input), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
