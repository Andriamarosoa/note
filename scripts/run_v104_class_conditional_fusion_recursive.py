"""Recovery wrapper for V10.4 fusion artifact layout.

GitHub's merged artifact download preserves nested paths from each OOF artifact,
while the original fusion loader expects the five NPZ shards directly under
--oof-dir.  Flatten only the five already-produced strict OOF NPZ files into a
temporary sibling directory, then run the unchanged V10.4 fusion script.

No OOF expert is retrained and locked12 remains untouched until the underlying
fusion script reaches its final frozen evaluation stage.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

from scripts import train_v104_class_conditional_fusion as v104


def _flatten_oof_dir(argv: list[str]) -> list[str]:
    args = list(argv)
    try:
        pos = args.index("--oof-dir")
        source = Path(args[pos + 1])
    except (ValueError, IndexError) as exc:
        raise SystemExit("--oof-dir is required") from exc

    shards = sorted(source.rglob("v104-oof-fold-*.npz"))
    if len(shards) != v104.FOLD_COUNT:
        raise SystemExit(f"expected {v104.FOLD_COUNT} recursive OOF shards, found {len(shards)}")

    flat = source.parent / f"{source.name}-flat"
    if flat.exists():
        shutil.rmtree(flat)
    flat.mkdir(parents=True)

    names = set()
    for shard in shards:
        if shard.name in names:
            raise SystemExit(f"duplicate OOF shard filename: {shard.name}")
        names.add(shard.name)
        shutil.copy2(shard, flat / shard.name)

    expected = {f"v104-oof-fold-{i}.npz" for i in range(v104.FOLD_COUNT)}
    if names != expected:
        raise SystemExit(f"unexpected OOF shard set: {sorted(names)}")

    args[pos + 1] = str(flat)
    print("V10.4 recovery flattened strict OOF shards:", sorted(names))
    return args


def main() -> int:
    argv = _flatten_oof_dir(sys.argv[1:])
    return int(v104.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
