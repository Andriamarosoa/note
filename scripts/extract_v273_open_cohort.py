"""Label-independent acoustic summaries, full trajectories and group geometry."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from scripts.yourmt3_exactk_common import FOLDS, RADIUS_SAMPLES, SAMPLE_RATE, digest, require


def ownership(starts, ends, i, query):
    """Whether the current native interval wins, including earliest-interval ties."""
    starts, ends, query = map(np.asarray,(starts,ends,query))
    require(np.all(starts[1:] > ends[:-1]), 'overlapping or unordered intervals')
    distance = np.maximum(np.maximum(starts[i]-query,query-ends[i]),0)
    own = distance <= RADIUS_SAMPLES
    if i:
        previous = np.maximum(np.maximum(starts[i-1]-query,query-ends[i-1]),0)
        own &= distance < previous
    if i+1 < len(starts):
        following = np.maximum(np.maximum(starts[i+1]-query,query-ends[i+1]),0)
        own &= distance <= following
    return own


def main(a):
    from causal_note.guitarset import index_guitarset
    from scripts.train_boundaries import decode_pcm16_mono_wav
    from scripts.extract_v273_harmonic_trajectory import _frames, features_from_spectrogram
    require(a.fold in FOLDS and not a.output.exists(),'fold/output')
    cohort_archive = a.root/'analysis/evidence/v273-yourmt3-target/upstream/yourmt3-exactk-summary.zip'
    require(digest(cohort_archive) == 'd47cbe59f66139dbbbb0c38b1dc2a1eb050d7ea25062f1dbba096971cb56eb89',
            'cohort source checksum')
    with zipfile.ZipFile(cohort_archive) as z:
        with np.load(io.BytesIO(z.read('predictions.npz')),allow_pickle=False) as m:
            sel=m['fold']==a.fold
            # Deliberately never read the competitor's predictions during extraction.
            d={k:m[k][sel] for k in ['global_index','member','starts','ends','k','baseline','fold']}
    order=np.argsort(d['global_index']);d={k:v[order] for k,v in d.items()}
    ids, members, starts, ends=d['global_index'],d['member'],d['starts'],d['ends']
    require(len(set(ids))==len(ids),'duplicate cohort IDs')
    require(all(Path(m).name[:2] in {'00','01','02','03','04'} for m in members),'excluded player')
    require(np.all((ends>=starts)&(ends-starts<=1764)), 'native interval width')
    wanted=set(members.tolist())
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,'audio membership mismatch')
    summaries=np.empty((len(ids),58),np.float32)
    sequence=np.empty((len(ids),42,49),np.float32)
    geometry=np.empty((len(ids),46),np.float32)
    feature_names=None;t0=time.monotonic()
    a.output.mkdir(parents=True)
    for track_number,name in enumerate(sorted(wanted),1):
        track=tracks[name];wav=decode_pcm16_mono_wav(track.audio_zip,track.audio_member)
        require(wav.sample_rate==SAMPLE_RATE,'sample rate')
        samples=np.asarray(wav.samples,np.float64)/32768.
        rows=np.flatnonzero(members==name);rows=rows[np.argsort(starts[rows])]
        track_starts,track_ends=starts[rows],ends[rows]
        for j,row in enumerate(rows):
            spectrum,times=_frames(samples,int(starts[row]))
            features,activation=features_from_spectrogram(spectrum,times,return_states=True)
            names=sorted(features)
            if feature_names is None:feature_names=names
            require(names==feature_names and activation.shape==(42,49),'feature schema')
            summaries[row]=[features[k] for k in names]
            sequence[row]=np.log1p(activation)
            query=starts[row]+np.rint(times*SAMPLE_RATE/1000).astype(np.int64)
            mask=ownership(track_starts,track_ends,j,query)
            before=(starts[row]-track_ends[j-1])/10584 if j else 1.
            after=(track_starts[j+1]-starts[row])/10584 if j+1<len(rows) else 1.
            geometry[row]=np.r_[mask,(ends[row]-starts[row])/1764,
                                np.clip(before,0,1),np.clip(after,0,1),mask.mean()]
        print(json.dumps(dict(stage='extract',fold=a.fold,track=track_number,tracks=len(wanted),
                              rows=int(len(rows)),elapsed_seconds=round(time.monotonic()-t0,2))),flush=True)
    require(all(np.isfinite(x).all() for x in (summaries,sequence,geometry)),'nonfinite features')
    # Original 58 features are an independent overlap check, not training input.
    old=a.root/'analysis/evidence/v273-regression-loops/prepared/inputs.npz'
    with np.load(old,allow_pickle=False) as z:
        check=z['fold']==a.fold
        positions=np.searchsorted(ids,z['global_index'][check])
        require(np.array_equal(ids[positions],z['global_index'][check]),'old cohort coverage')
        require(feature_names==z['feature_names'].tolist(),'old feature name order')
        expected=z['features'][check]
        require(np.allclose(summaries[positions],expected,rtol=3e-5,atol=3e-5),
                'old feature overlap drift')
        difference=float(np.max(np.abs(summaries[positions]-expected)))
    output=a.output/f'features-fold-{a.fold}.npz'
    np.savez_compressed(output,**d,summary=summaries,sequence=sequence,geometry=geometry,
                        feature_names=np.asarray(feature_names))
    report=dict(status='completed',fold=a.fold,rows=len(ids),recordings=len(wanted),
        feature_file=output.name,feature_sha256=digest(output),summary_shape=list(summaries.shape),
        sequence_shape=list(sequence.shape),geometry_shape=list(geometry.shape),
        old_feature_overlap_max_abs=difference,labels_used_in_features=False,
        baseline_used_in_features=False,yourmt3_used_in_features=False,
        fold3_used=False,player05_used=False,audio_lookahead_ms=160,
        elapsed_seconds=time.monotonic()-t0)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--fold',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
