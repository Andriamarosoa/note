"""Repack completed, regenerable float32 caches with verified bit parity.

This storage operation never touches an event, model or calibrated policy.
It can free space in an interrupted reproduction before collection resumes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from onset256.feature_cache import encode_features, decode_features
from scripts.onset_event_integrity256 import digest, dump


def repack(directory):
    for metadata in sorted(directory.glob('track_*.json')):
        row=json.loads(metadata.read_text())
        path=directory/f"features_{row['index']:03d}.npz"
        if not path.exists():continue
        with np.load(path,allow_pickle=False) as stored:
            original=decode_features(stored)
            encoded=encode_features(original) if 'features' in stored else None
        if encoded is not None:
            restored=decode_features(encoded)
            np.testing.assert_array_equal(original.view(np.uint32),restored.view(np.uint32))
            previous=path.stat().st_size
            # This is an explicitly regenerable scratch cache. Holding both
            # verified representations in memory permits repacking a full disk.
            with path.open('wb') as fd:np.savez(fd,**encoded)
            with np.load(path,allow_pickle=False) as stored:
                np.testing.assert_array_equal(original.view(np.uint32),decode_features(stored).view(np.uint32))
            print('REPACK',row['index'],previous,path.stat().st_size,flush=True)
        row['feature_sha256']=digest(path)
        row['cache_storage']='zstd_byteplanes_float32_v1'
        dump(metadata,row)
        duplicate=path.with_suffix('.npy')
        if duplicate.exists():
            # The metadata names the verified NPZ, not a redundant/incomplete
            # NPY left by an interrupted attempt. Never train from that NPY.
            duplicate.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    repack(p.parse_args().directory)
