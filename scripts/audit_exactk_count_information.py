"""Count-only diagnostic probes, frozen composition folds and local audio budget.

No pitch, note identity, YourMT3 output, annotation timing, or reference prediction
is an input to any probe. Labels and reference predictions are evaluation-only,
except K on the three training folds for supervised fitting. This is exploratory
validation on the existing development folds, not a new untouched benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np

FOLDS = (0, 1, 2, 4)
ARMS = ('candidate_only', 'native_channel_norm', 'native_raw', 'native_raw_multires')
SEED = 7102026
PRE, POST, SR = 1308, 2788, 44100
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    'purpose': 'find information useful for direct Exact-K without note identification',
    'folds': list(FOLDS), 'excluded_fold': 3, 'excluded_player': '05',
    'window': {'pre_samples': PRE, 'post_samples': POST, 'sample_rate': SR},
    'arms': list(ARMS),
    'classifier': {'type': 'HistGradientBoostingClassifier', 'max_iter': 120,
                   'learning_rate': .08, 'max_leaf_nodes': 15, 'max_depth': 6,
                   'min_samples_leaf': 30, 'l2_regularization': 5.,
                   'max_bins': 64, 'early_stopping': False, 'random_state': SEED},
    'weighting': 'inverse-square-root class frequency, clipped [0.35,4], fit folds only',
    'training': 'leave one of four frozen composition folds out; no hyperparameter selection',
    'yourmt3_predictions_used_for_training': False,
    'freeze_predictions_used_for_training': False,
    'note_or_pitch_features': False,
    'automatic_promotion': False,
    'interpretation': 'diagnostic information probes on existing development folds; upstream cached inputs shared with reference',
}


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def cohort(summary):
    with zipfile.ZipFile(summary) as z, z.open('predictions.npz') as f, np.load(f) as d:
        result = {k: np.asarray(d[k]) for k in d.files}
    assert len(result['k']) == 59309 and set(result['fold']) == set(FOLDS)
    assert sum(result['k'] >= 2) == 7385
    assert all(str(m).startswith(('00_', '01_', '02_', '03_', '04_')) for m in result['member'])
    assert len(np.unique(result['global_index'])) == len(result['k'])
    return result


def composition(member):
    s = Path(str(member)).stem.split('_', 1)[1]
    return s.rsplit('_', 1)[0]


def candidate_features(sequence, mask, stats):
    s = sequence.astype(np.float32)
    m = mask.astype(bool)
    count = np.maximum(m.sum(1), 1).astype(np.float32)
    mean = (s * m[:, :, None]).sum(1) / count[:, None]
    variance = ((s - mean[:, None, :]) ** 2 * m[:, :, None]).sum(1) / count[:, None]
    maximum = np.where(m[:, :, None], s, -np.inf).max(1)
    return np.concatenate([stats, np.log1p(count[:, None]), mean, np.sqrt(variance), maximum], 1)


def grid_features(spectral, normalize=False):
    x = np.asarray(spectral, np.float32)
    assert x.ndim == 4 and x.shape[1:] == (31, 64, 3)
    if normalize:
        # Keras LayerNormalization defaults: axis=-1, epsilon=0.001.
        # Initial gamma/beta are unnecessary for measuring information availability.
        x = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + .001)
    output = []
    for start, end in [(0, 9), (9, 17), (17, 24), (24, 31)]:
        v = x[:, start:end].reshape(len(x), end-start, 16, 4, 3)
        output.extend([v.mean((1, 3)).reshape(len(x), -1), v.max((1, 3)).reshape(len(x), -1)])
    return np.concatenate(output, 1)


def prepare_cache(args):
    d = cohort(args.summary); root = args.output; root.mkdir(parents=True, exist_ok=True)
    write_json(root/'protocol.json', PROTOCOL)
    manifest = json.loads((args.bundle/'bundle.json').read_text())
    cfg_path = args.config or ROOT/'analysis/v273-native-paired-config.json'
    assert manifest['config_sha256'] == sha(cfg_path)
    cfg = json.loads(cfg_path.read_text())
    idx = d['global_index']
    arrays = {}
    for name in ['members', 'exact', 'cluster_start_samples', 'sequence', 'mask', 'stats', 'spectral']:
        p = args.bundle/(name+'.npy')
        assert sha(p) == manifest['fields'][name]['sha256'], name
        arrays[name] = np.load(p, mmap_mode='r', allow_pickle=False)
    np.testing.assert_array_equal(arrays['members'][idx], d['member'])
    np.testing.assert_array_equal(np.minimum(arrays['exact'][idx], 6), d['k'])
    np.testing.assert_array_equal(arrays['cluster_start_samples'][idx], d['starts'])
    np.testing.assert_array_equal([cfg['member_folds'][str(m)] for m in d['member']], d['fold'])
    for m, fold in zip(d['member'], d['fold']):
        assert cfg['composition_folds'][composition(m)] == fold
    out = {
        'candidate': np.empty((len(idx), 408), np.float32),
        'native_norm': np.empty((len(idx), 384), np.float32),
        'native_raw': np.empty((len(idx), 384), np.float32),
    }
    raw_summary = np.empty((len(idx), 3), np.float32)
    for begin in range(0, len(idx), 128):
        local = slice(begin, begin+128); g = idx[local]
        out['candidate'][local] = candidate_features(arrays['sequence'][g], arrays['mask'][g], arrays['stats'][g])
        sp = arrays['spectral'][g].astype(np.float32)
        out['native_norm'][local] = grid_features(sp, True)
        out['native_raw'][local] = grid_features(sp)
        raw_summary[local] = np.mean(sp[:, 10:], axis=(1, 2))
    assert all(np.isfinite(x).all() for x in out.values())
    np.savez_compressed(root/'cache_features.npz', **out, raw_channel_mean=raw_summary)
    np.savez_compressed(root/'cohort.npz', **d)
    write_json(root/'cache_provenance.json', {'bundle_manifest_sha256': sha(args.bundle/'bundle.json'),
        'summary_sha256': sha(args.summary), 'feature_sha256': sha(root/'cache_features.npz'),
        'shapes': {k:list(v.shape) for k,v in out.items()}, 'rows':len(idx),
        'feature_code_sha256': sha(__file__)})
    print('cached features verified', len(idx), flush=True)


def local_windows(samples, starts):
    positions = starts[:, None] - PRE + np.arange(PRE+POST)[None, :]
    valid = (positions >= 0) & (positions < len(samples))
    return samples[np.clip(positions, 0, len(samples)-1)] * valid


def multires_features(windows):
    result = []
    targets = np.geomspace(55., 6000., 64)
    fft_freq = np.fft.rfftfreq(4096, d=1/SR)
    right = np.searchsorted(fft_freq, targets); left = right-1
    frac = (targets-fft_freq[left])/(fft_freq[right]-fft_freq[left])
    pre_power = np.mean(windows[:, :PRE] ** 2, axis=1) + 1e-10
    for width in [1024, 2048]:
        starts = np.arange(0, windows.shape[1]-width+1, 256)
        taper = np.hanning(width).astype(np.float32)
        frames = windows[:, starts[:, None]+np.arange(width)[None, :]] * taper
        p = np.abs(np.fft.rfft(frames, n=4096, axis=-1)) ** 2 / np.sum(taper**2)
        p = p[:, :, left]*(1-frac) + p[:, :, right]*frac
        x = np.clip(np.log1p(p/pre_power[:, None, None]), 0., 12.)
        for sub in np.array_split(np.arange(len(starts)), 4):
            v = x[:, sub].reshape(len(x), len(sub), 16, 4)
            result.extend([v.mean((1,3)), v.max((1,3))])
    return np.concatenate(result, 1).astype(np.float32)


def native_spectral(windows):
    starts = np.arange(0, 4096-256+1, 128)
    frame = windows[:, starts[:,None]+np.arange(256)[None,:]] * np.hanning(256)
    p = np.abs(np.fft.rfft(frame,n=2048,axis=-1))**2
    f = np.fft.rfftfreq(2048,1/SR);t=np.geomspace(55.,6000.,64)
    right=np.searchsorted(f,t);left=right-1;frac=(t-f[left])/(f[right]-f[left])
    p=p[:,:,left]*(1-frac)+p[:,:,right]*frac
    pre=p[:,:9];scalar=np.median(pre.mean(2),axis=1)+1e-10;band=pre.mean(1)+1e-10
    log=np.clip(np.log1p(p/scalar[:,None,None]),0.,12.)
    birth=np.clip(np.log1p(np.maximum(p-band[:,None,:],0.)/band[:,None,:]),0.,12.)
    flux=np.zeros_like(log);flux[:,1:]=np.maximum(log[:,1:]-log[:,:-1],0.)
    return np.stack([log,birth,flux],-1)


def prepare_audio(args):
    from scipy.io import wavfile
    d=cohort(args.summary); out=np.empty((len(d['k']),256),np.float32)
    native=np.load(args.bundle/'spectral.npy',mmap_mode='r')
    checks=[]; begin=time.monotonic()
    with args.audio.open('rb') as f:
        assert hashlib.file_digest(f,'md5').hexdigest()=='aecce79f425a44e2055e46f680e10f6a'
    with zipfile.ZipFile(args.audio) as z:
        lookup={Path(n).name:n for n in z.namelist() if n.endswith('.wav')}
        for j,member in enumerate(sorted(set(d['member']))):
            rows=np.flatnonzero(d['member']==member)
            sample_rate,x=wavfile.read(io.BytesIO(z.read(lookup[Path(member).stem+'_mix.wav'])))
            assert sample_rate==SR and x.ndim==1
            if x.dtype.kind=='i':x=x.astype(np.float32)/float(2**(np.iinfo(x.dtype).bits-1))
            else:x=x.astype(np.float32)
            for k in range(0,len(rows),64):
                local=rows[k:k+64]
                windows=local_windows(x,d['starts'][local])
                out[local]=multires_features(windows)
                if k==0:
                    ref=np.asarray(native[d['global_index'][local[:1]]],np.float32)
                    rebuilt=native_spectral(windows[:1]).astype(np.float32)
                    delta=float(np.max(np.abs(ref-rebuilt)))
                    assert np.allclose(ref,rebuilt,atol=.005,rtol=.002), (member,delta)
                    checks.append(delta)
            if j%20==0: print('audio features',j+1,'/190',round(time.monotonic()-begin,1),'s',flush=True)
    assert np.isfinite(out).all()
    np.save(args.output/'multires_features.npy',out)
    write_json(args.output/'audio_provenance.json',{'rows':len(out),'tracks':len(checks),
        'same_pcm_window_as_reference':True,'native_replay_max_abs_error':max(checks),
        'feature_sha256':sha(args.output/'multires_features.npy'),
        'feature_shape':list(out.shape),'no_future_samples_after_group_start_plus_2788':True})


def metric(y,p):
    y=np.asarray(y);p=np.asarray(p)
    def sub(m):
        n=int(m.sum());correct=int((m&(y==p)).sum())
        return {'rows':n,'correct':correct,'exact':correct/n if n else None,
                'under':int((m&(p<y)).sum()),'over':int((m&(p>y)).sum())}
    return {**sub(np.ones(len(y),bool)),'poly':sub(y>=2),
            'by_k':{str(k):sub(y==k) for k in range(7)}}


def train(args):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    import sklearn
    root=args.output
    with np.load(root/'cohort.npz') as d:
        y=d['k'];fold=d['fold'];base=d['baseline'];ymt=d['predicted'];members=d['member']
    with np.load(root/'cache_features.npz') as d:
        c=d['candidate'];norm=d['native_norm'];raw=d['native_raw']
    parts={'candidate_only':[c],'native_channel_norm':[c,norm],
           'native_raw':[c,raw]}
    if args.arm=='native_raw_multires':
        parts[args.arm]=[c,raw,np.load(root/'multires_features.npy')]
    x=np.concatenate(parts[args.arm],1)
    pred=np.full(len(y),-1,np.int32);prob=np.full((len(y),7),np.nan,np.float32)
    report={'arm':args.arm,'features':x.shape[1],'sklearn':sklearn.__version__,
            'protocol':PROTOCOL,'by_fold':{}}
    comp=np.asarray([composition(m) for m in members])
    for f in FOLDS:
        tr=fold!=f;va=fold==f
        assert not set(comp[tr])&set(comp[va])
        freq=np.bincount(y[tr],minlength=7)
        weights=np.clip(np.sqrt(tr.sum()/(7*freq)),.35,4.)
        weights/=weights[y[tr]].mean()
        config={k:v for k,v in PROTOCOL['classifier'].items() if k!='type'}
        model=HistGradientBoostingClassifier(**config)
        begin=time.monotonic()
        with threadpool_limits(limits=2):
            model.fit(x[tr],y[tr],sample_weight=weights[y[tr]])
            prob[va]=model.predict_proba(x[va]);pred[va]=model.classes_[np.argmax(prob[va],1)]
        report['by_fold'][str(f)]={'train_rows':int(tr.sum()),'test_rows':int(va.sum()),
            'fit_weights':weights.tolist(),'metrics':metric(y[va],pred[va]),
            'seconds':time.monotonic()-begin}
        write_json(root/(args.arm+'-progress.json'),report)
        print(json.dumps({'arm':args.arm,'fold':f,'seconds':round(time.monotonic()-begin,1),
             'global':float(np.mean(pred[va]==y[va])),
             'poly':float(np.mean(pred[va&(y>=2)]==y[va&(y>=2)]))}),flush=True)
    assert (pred>=0).all() and np.isfinite(prob).all()
    report['metrics']=metric(y,pred)
    report['reference']=metric(y,base)
    report['paired_vs_freeze']={key:{'fixed':int((m&(base!=y)&(pred==y)).sum()),
        'regressed':int((m&(base==y)&(pred!=y)).sum()),
        'net':int(((pred==y).astype(int)-(base==y).astype(int))[m].sum())}
        for key,m in [('global',np.ones(len(y),bool)),('poly',y>=2)]}
    write_json(root/(args.arm+'-report.json'),report)
    np.savez_compressed(root/(args.arm+'-predictions.npz'),predicted=pred,probability=prob)
    print('COMPLETE',args.arm,json.dumps(report['metrics']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['cache','audio','train'])
    p.add_argument('--summary',type=Path)
    p.add_argument('--bundle',type=Path)
    p.add_argument('--audio',type=Path)
    p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=ARMS)
    a=p.parse_args()
    {'cache':prepare_cache,'audio':prepare_audio,'train':train}[a.stage](a)
