"""Recover the V13 five-fold summary without retraining.

The original summarizer concatenated the one-element ``schema_version`` array
with row-wise arrays and then applied the cluster sort order to it.  This shim
copies only row-wise arrays into temporary shards and delegates all metric logic
to the original summarizer.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np

from scripts import summarize_v130_causal_event_set_decoder as summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    with tempfile.TemporaryDirectory(prefix="v130-summary-fixed-") as td:
        root = Path(td)
        for fold in range(5):
            reports = sorted(args.input_dir.glob(f"**/report-fold-{fold}.json"))
            shards = sorted(args.input_dir.glob(f"**/predictions-fold-{fold}.npz"))
            if len(reports) != 1 or len(shards) != 1:
                raise RuntimeError(f"fold {fold}: expected one report and one prediction shard")
            out = root / f"fold-{fold}"
            out.mkdir(parents=True)
            shutil.copy2(reports[0], out / reports[0].name)
            with np.load(shards[0], allow_pickle=False) as z:
                n = int(np.asarray(z["global_index"]).shape[0])
                rowwise = {
                    key: np.asarray(z[key])
                    for key in z.files
                    if np.asarray(z[key]).ndim >= 1 and np.asarray(z[key]).shape[0] == n
                }
            if "schema_version" in rowwise:
                raise RuntimeError("schema_version unexpectedly classified as row-wise")
            np.savez_compressed(out / shards[0].name, **rowwise)

        ns = argparse.Namespace(input_dir=root, output_dir=args.output_dir)
        summary.summarize(ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
