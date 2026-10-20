"""Preserve exact teacher decisions and checkpoints, without a large DSP cache."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from scripts.s813_prepare_native256 import digest_file, save_json


def package(args):
    report = json.loads((args.experiment/'report.json').read_text())
    if not report['complete'] or len(report['tracks']) != 80:
        raise ValueError('Completed genuine 80-track experiment required')
    if args.out.exists():
        raise FileExistsError('Evidence is immutable')
    args.out.mkdir(parents=True)
    shutil.copy2(args.experiment/'report.json',args.out/'report.json')
    shutil.copy2(args.cache/'report.json',args.out/'extraction.json')
    (args.out/'models').mkdir()
    for fold,entry in report['folds'].items():
        source = args.experiment/entry['checkpoint']
        if digest_file(source) != entry['checkpoint_sha256']:
            raise ValueError('Model SHA mismatch')
        shutil.copy2(source,args.out/entry['checkpoint'])
    masks = {}
    for row in report['tracks']:
        source = args.experiment/row['predictions']
        if digest_file(source) != row['predictions_sha256']:
            raise ValueError('Original prediction SHA mismatch')
        with np.load(source,allow_pickle=False) as z:
            mask = z['predicted']
            if mask.dtype != bool or mask.shape != (row['frames'],49):
                raise ValueError('Invalid teacher decision dimensions')
            np.testing.assert_array_equal(z['decision_ends'],np.arange(256,(len(mask)+1)*256,256))
            masks[f"fit_{row['index']:03d}"] = mask
    np.savez_compressed(args.out/'decisions.npz',**masks)
    with zipfile.ZipFile(args.out/'source_metadata.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for row in sorted(report['tracks'],key=lambda r:r['index']):
            name = f"fit_{row['index']:03d}.json"
            z.write(args.traces/name,name)
    paths = sorted(p for p in args.out.rglob('*') if p.is_file())
    manifest = dict(schema='S813_EXACT_TEACHER_EVIDENCE_BUNDLE_V1',
        full_probability_files_retained_in_research_workspace=True,
        decisions_are_exact_not_reconstructed=True,
        source_code_sha256={name:digest_file(name) for name in (
            'src/causal_note/strict_live.py','src/causal_note/s706_sharp_spectra.py',
            'src/causal_note/s813_framepitch_teacher.py','scripts/s813_prepare_native256.py',
            'scripts/s813_train_framepitch256.py')},
        files={p.relative_to(args.out).as_posix():digest_file(p) for p in paths})
    save_json(args.out/'bundle.json',manifest)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for arg in ('experiment','cache','traces','out'):
        p.add_argument('--'+arg,type=Path,required=True)
    package(p.parse_args())
