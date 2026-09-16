"""Back up one completed or partial native training stage without expiry."""
import argparse
import os
from pathlib import Path
import subprocess
import zipfile

from scripts.rebuild_v273_sources import digest, write_json


def archive(root, name, tag):
    if not root.exists() or not any(root.rglob('*')):
        print('No stage output to back up.')
        return
    if Path(name).name != name:
        raise ValueError('invalid archive name')
    files = {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob('*'))
             if p.is_file() and p.name != 'file-sha256.json'}
    write_json(root/'file-sha256.json', files)
    path = root.parent/(name+'.zip')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(root))
    check = path.with_suffix('.zip.sha256')
    check.write_text(digest(path)+'  '+path.name+'\n')
    subprocess.run(['gh', 'release', 'upload', tag, str(path), str(check),
                    '--repo', os.environ['GITHUB_REPOSITORY']], check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--tag', required=True)
    a = p.parse_args()
    archive(a.root, a.name, a.tag)
