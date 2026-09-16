"""Restore our checksum-pinned source backup after original Actions expiry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import tempfile
import zipfile


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_original_inventory(root):
    expected = json.loads((root/'file-sha256.json').read_text())
    files = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if files != set(expected) | {'file-sha256.json'}:
        raise RuntimeError('preserved source file inventory mismatch')
    for name, digest in expected.items():
        if sha256(root/name) != digest:
            raise RuntimeError(f'preserved source file changed: {name}')


def extract_verified(archive, output_dir, expected_sha256, validate=validate_original_inventory):
    archive, output_dir = Path(archive), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if sha256(archive) != expected_sha256:
        raise RuntimeError('preserved source archive checksum mismatch')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
        root = Path(temporary) / 'verified'
        root.mkdir()
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            if len(set(names)) != len(names):
                raise RuntimeError('duplicate backup entries')
            for info in z.infolist():
                p = Path(info.filename)
                if p.is_absolute() or '..' in p.parts or stat.S_ISLNK(info.external_attr >> 16):
                    raise RuntimeError('unsafe backup path')
            z.extractall(root)
        validate(root)
        root.rename(output_dir)


def restore(config_path, output_dir):
    cfg = json.loads(Path(config_path).read_text())
    source = cfg['preserved_originals_backup']
    if source['repository'] != 'Andriamarosoa/note':
        raise RuntimeError('unexpected recovery repository')
    asset = source['asset_name']
    if Path(asset).name != asset or not asset.endswith('.zip'):
        raise RuntimeError('invalid recovery asset name')
    with tempfile.TemporaryDirectory() as temporary:
        subprocess.run(['gh', 'release', 'download', source['tag'],
                        '--repo', source['repository'], '--pattern', asset,
                        '--dir', temporary], check=True)
        extract_verified(Path(temporary)/asset, output_dir, source['sha256'])
    print('Restored and verified the preserved original source files.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    restore(args.config, args.output_dir)
