"""Lossless research-cache storage; never used by the Live decision path."""
from pathlib import Path

import numpy as np


def encode_features(value):
    import zstandard
    value = np.asarray(value)
    if value.dtype != np.float32:
        raise ValueError('research feature cache requires float32')
    reverse = tuple(reversed(range(value.ndim)))
    planes = np.ascontiguousarray(value.transpose(reverse)).view(np.uint8).reshape(-1,4).T.copy()
    payload = zstandard.ZstdCompressor(level=3).compress(planes.tobytes())
    return dict(shape=np.asarray(value.shape,np.int64),
        layout=np.asarray('zstd_byteplanes_float32_v1'),
        payload=np.frombuffer(payload,np.uint8))


def decode_features(stored):
    if 'features' in stored:
        return stored['features']
    import zstandard
    if str(stored['layout']) != 'zstd_byteplanes_float32_v1':
        raise ValueError('unsupported lossless cache format')
    shape=tuple(int(n) for n in stored['shape'])
    raw=zstandard.ZstdDecompressor().decompress(stored['payload'].tobytes(),
        max_output_size=int(np.prod(shape))*4)
    value=np.frombuffer(raw,np.uint8).reshape(4,-1).T.copy().view(np.float32).reshape(shape[::-1])
    return np.ascontiguousarray(value.transpose(tuple(reversed(range(len(shape))))))


def load_features(directory, index):
    directory = Path(directory)
    path = directory / f"features_{index:03d}.npy"
    if path.exists():
        return np.load(path, mmap_mode="r", allow_pickle=False)
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as stored:
        return decode_features(stored)


def compress_features(path):
    """Check bit-for-bit roundtrip before removing the uncompressed cache."""
    path = Path(path)
    target = path.with_suffix(".npz")
    if target.exists():
        raise FileExistsError(target)
    original = np.load(path, mmap_mode="r", allow_pickle=False)
    np.savez(target, **encode_features(original))
    with np.load(target, allow_pickle=False) as stored:
        restored = decode_features(stored)
        if (restored.dtype != original.dtype or restored.shape != original.shape
                or not np.array_equal(restored.view(np.uint32), original.view(np.uint32))):
            raise AssertionError("lossless cache changed acoustic features")
    del original
    path.unlink()
    return target
