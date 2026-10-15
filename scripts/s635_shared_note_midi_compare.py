"""S635 compare NOTE and MIDI only after the same source audio is certified.

Usage:
  PYTHONPATH=.:src python -m scripts.s635_shared_note_midi_compare \
       --manifest comparison_frozen.json --out comparison_report.json

No resampling, model inference, web downloading or ground-truth guessing.
Input JSON must contain both emitted-event streams, labels and identical
SHA256 for every clip. This gate is deliberately independent of both models.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from causal_note.s635_shared_midi_metric import load_and_compare


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("Do not overwrite frozen comparison")
    results=load_and_compare(args.manifest)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    print("S635 COMMON CAUSAL COMPARISON",json.dumps(dict(
        recordings=results["recordings"],common_range=results["common_pitch_range"],
        note=results["overall"]["note"]["80"],
        midi=results["overall"]["midi"]["80"]),indent=2))


if __name__=="__main__":
    main()
