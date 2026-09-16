"""Restore the pinned rebuilt input release for the paired native experiment."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from scripts.restore_v273_original_backup import extract_verified
from scripts.rebuild_v273_sources import write_json
from scripts.v273_native_protocol import load_config, verify_native_cache


def prepare(config, output):
    cfg = load_config(config)
    if output.exists():
        raise FileExistsError(output)
    if cfg['repository'] != 'Andriamarosoa/note':
        raise RuntimeError('unexpected source repository')
    for i, source in enumerate(sorted(cfg['archives'], key=lambda x: x['name'])):
        if source['name'] != f'v273-rebuilt-native-{i}.zip':
            raise RuntimeError('unexpected source archive')
        with tempfile.TemporaryDirectory() as temp:
            subprocess.run(['gh', 'release', 'download', cfg['source_release'],
                            '--repo', cfg['repository'], '--pattern', source['name'],
                            '--dir', temp], check=True)
            archive = Path(temp)/source['name']
            # The archive digest pins the complete inventory. Per-cache file
            # digests and producer identity are additionally checked below.
            extract_verified(archive, output/f'shard-{i}', source['sha256'],
                             validate=lambda root: None)
    verified = verify_native_cache(output, config)
    for name, values in cfg['expert_configuration'].items():
        write_json(output/'expert-config'/f'{name}-report.json', {'configuration': values})
    write_json(output/'verified-inputs.json', verified)
    print(json.dumps(verified, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    prepare(args.config, args.output_dir)
