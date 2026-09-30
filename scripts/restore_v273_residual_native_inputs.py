"""Restore checksum-pinned inputs for native residual comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.request import urlopen

from scripts.rebuild_v273_sources import digest,verify_dataset
from scripts.restore_v273_original_backup import extract_verified,validate_original_inventory


def restore(root, with_audio, launch_path=Path('analysis/v273-residual-native-launch.json')):
    launch=json.loads(launch_path.read_text())
    sources=json.loads(Path('analysis/v273-stable-prediction-sources.json').read_text())['files']
    selected=[r for r in sources if with_audio and r['name'] in ('annotation.zip','audio_mono-pickup_mix.zip')]
    selected.append(dict(name=launch['geometry_asset'],sha256=launch['geometry_archive_sha256'],
        url='https://github.com/Andriamarosoa/note/releases/download/'+launch['geometry_release']+'/'+launch['geometry_asset']))
    selected.append(dict(name=launch['bundle_asset'],sha256=launch['bundle_archive_sha256'],
        url='https://github.com/Andriamarosoa/note/releases/download/'+launch['source_release']+'/'+launch['bundle_asset']))
    def get(item):
        name=item['name'];directory=root/('GuitarSet' if name in ('annotation.zip','audio_mono-pickup_mix.zip') else 'download')
        directory.mkdir(parents=True,exist_ok=True)
        path=directory/name
        if not path.exists():
            temporary=path.with_suffix('.part')
            with urlopen(item['url'],timeout=90) as source,temporary.open('wb') as target:
                while chunk:=source.read(4*1024*1024):
                    target.write(chunk)
            if digest(temporary)!=item['sha256']:
                raise RuntimeError('download checksum mismatch: '+name)
            temporary.replace(path)
        if digest(path)!=item['sha256']:
            raise RuntimeError('existing archive differs: '+name)
        if directory.name=='download':
            target=root/name[:-4]
            if target.exists():validate_original_inventory(target)
            else:extract_verified(path,target,item['sha256'])
        print('Verified '+name,flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(get,selected))
    if with_audio:verify_dataset(root/'GuitarSet')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--with-audio',action='store_true')
    p.add_argument('--launch',type=Path,default=Path('analysis/v273-residual-native-launch.json'))
    args=p.parse_args();restore(args.root,args.with_audio,args.launch)
