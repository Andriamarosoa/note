"""Download checksum-pinned inputs for the coherent/stable development audit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.request import urlopen

from scripts.rebuild_v273_sources import digest, verify_dataset
from scripts.restore_v273_original_backup import extract_verified, validate_original_inventory


def restore(root):
    manifest = json.loads(Path('analysis/v273-stable-prediction-sources.json').read_text())
    def get(item):
        name = item['name']
        if Path(name).name != name or not item['url'].startswith('https://'):
            raise ValueError('invalid source path or URL')
        directory = root/('download' if name.startswith('ownership-') else 'GuitarSet')
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/name
        if not path.exists():
            temporary = path.with_suffix(path.suffix+'.part')
            with urlopen(item['url'], timeout=90) as source, temporary.open('wb') as out:
                while chunk := source.read(1024*1024):
                    out.write(chunk)
            if digest(temporary) != item['sha256']:
                raise RuntimeError('download checksum mismatch: '+name)
            temporary.replace(path)
        if digest(path) != item['sha256']:
            raise RuntimeError('existing source changed: '+name)
        if name.startswith('ownership-'):
            target = root/name[:-4]
            if target.exists():
                validate_original_inventory(target)
            else:
                extract_verified(path, target, item['sha256'])
        print('Verified '+name, flush=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(get, manifest['files']))
    verify_dataset(root/'GuitarSet')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    restore(p.parse_args().root)
