"""Resume the failed source recovery from its completed, pinned V8.1 weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from scripts import rebuild_v273_sources as rebuild
from scripts.restore_v273_original_backup import extract_verified


def prepare_resume(root, source):
    """Keep the failed audit as history; only completed V8.1 becomes reusable."""
    state_path = root / 'rebuild-state.json'
    state = rebuild.validate_sources(root, ('v81',))
    if state['source_code_sha'] != source['producer_commit']:
        raise RuntimeError('checkpoint producer mismatch')
    if state.get('dataset_md5') != rebuild.DATA_MD5:
        raise RuntimeError('checkpoint dataset mismatch')
    if set(state['stages']) != {'v81', 'audit'} or state['stages']['audit']['status'] != 'failed':
        raise RuntimeError('expected completed V8.1 followed by a failed audit')
    expected = [{'path': 'v81/stream.epoch-03.keras', 'sha256': source['checkpoint_sha256']}]
    if state['stages']['v81']['outputs'] != expected:
        raise RuntimeError('unexpected V8.1 checkpoint identity')
    prior_state_sha = rebuild.digest(state_path)
    history = root / 'history' / ('run-' + str(int(source['run_id'])))
    history.mkdir(parents=True)
    shutil.copy2(state_path, history / 'rebuild-state.json')
    if (root / 'audit').exists():
        (root / 'audit').rename(history / 'audit')
    state['stages'] = {'v81': state['stages']['v81']}
    state['stages']['v81']['source_code_sha'] = source['producer_commit']
    state['stages']['v81']['reused_from_run_id'] = source['run_id']
    state['resumed_from'] = {**source, 'state_sha256': prior_state_sha,
                             'preserved_history': history.relative_to(root).as_posix()}
    state['source_code_sha'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=rebuild.ROOT, text=True).strip()
    state['resumed_at_utc'] = rebuild.now()
    state['native_decay_comparison_started'] = False
    state['historical_v273_reproduced'] = False
    rebuild.write_json(state_path, state)


def restore_archive(archive, output_dir, source):
    extract_verified(archive, output_dir, source['sha256'],
                     validate=lambda root: prepare_resume(root, source))


def restore(config_path, output_dir):
    source = json.loads(Path(config_path).read_text())['rebuilt_v81_backup']
    if source['repository'] != 'Andriamarosoa/note':
        raise RuntimeError('unexpected recovery repository')
    asset = source['asset_name']
    if Path(asset).name != asset or not asset.endswith('.zip'):
        raise RuntimeError('invalid recovery asset name')
    with tempfile.TemporaryDirectory() as temporary:
        subprocess.run(['gh', 'release', 'download', source['tag'],
                        '--repo', source['repository'], '--pattern', asset,
                        '--dir', temporary], check=True)
        restore_archive(Path(temporary) / asset, output_dir, source)
    print(json.dumps({'stage': 'v81', 'status': 'verified_checkpoint_reused',
                      'source_run_id': source['run_id'],
                      'sha256': source['checkpoint_sha256'],
                      'native_decay_comparison_started': False}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    restore(args.config, args.output_dir)
