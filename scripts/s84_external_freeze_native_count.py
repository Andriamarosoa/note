"""S84: frozen original V8->V27.3 count predictor on unseen GOAT WAVs.

No MIDI or annotation is ever opened. This is an external deployment
component test, NOT an evaluation of the S80 proposal selector.
"""
from __future__ import annotations
import argparse, hashlib, io, json, wave, zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scripts.evaluate_v273_fresh_player05 import (
    proposal_args, runtime_records, runtime_stack
)
from scripts.train_v91_ordinal_cardinality import _load_frozen_stack
from scripts.train_v86_state_transition_proposals import _predict_score_tracks
from scripts.train_v90_structured_cluster_cardinality import (
    _extended_candidate_features, _cluster_arrays, MAX_CANDIDATES,
    CLUSTER_WINDOW_MS
)
from scripts.evaluate_v90_cardinality_realization import _clusters_window
from scripts.train_v100_spectral_string_slots import _spectral_maps_for_runtime
from scripts.spectral_window import COVERED
from scripts.candidate_timing import capture_timing
from scripts.audit_v273_internal_b_like_boundary_corrector import predict,SEED
from scripts.train_v273_group_gate_ab import build_model
SR=44100
def require(c,why):
    if not c:raise RuntimeError(why)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()
def external_track_zip(audio,out,seconds):
    require(not out.exists(),'unseen audio cache already exists')
    tracks=[]
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_STORED) as z:
        for no,name in enumerate(('example1_di.wav','example2_di.wav'),1):
            path=audio/name
            with wave.open(str(path),'rb') as w:
                require(w.getframerate()==SR and w.getsampwidth()==2 and
                        w.getnchannels()==1 and w.getcomptype()=='NONE','GOAT WAV format drift')
                pcm=w.readframes(min(int(seconds*SR),w.getnframes()))
            stream=io.BytesIO()
            with wave.open(stream,'wb') as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(SR)
                w.writeframes(pcm)
            namebase='00_GOATexample'+str(no)+'_comp'
            wav_member=namebase+'_mix.wav'
            z.writestr(wav_member,stream.getvalue())
            tracks.append(SimpleNamespace(
                annotation_member=namebase+'.jams',
                audio_zip=out,audio_member=wav_member))
    return tracks
def run(source,audio,weights,output,seconds):
    require(not output.exists(),'do not overwrite frozen independent test')
    output.mkdir(parents=True)
    tracks=external_track_zip(audio,output/'external-unlabeled-audio.zip',seconds)
    args=proposal_args(source)
    floor,threshold,enc86,enc87,model88=_load_frozen_stack(args)
    score=_predict_score_tracks(args.base_model,tracks)
    records=runtime_records(tracks,score,floor)
    require(len(records)>0,'no native original V84-V88 candidate records')
    x88,out88=runtime_stack(records,enc86,enc87,model88)
    feats,fused=_extended_candidate_features(x88,out88)
    clusters=_clusters_window(records,CLUSTER_WINDOW_MS)
    sequence,mask,stats,_,_,truncated=_cluster_arrays(
        clusters,{},records,feats,out88,fused)
    geometry=capture_timing(clusters,records,fused,MAX_CANDIDATES)
    spectra=_spectral_maps_for_runtime(tracks,clusters,records,window=COVERED)
    require(np.isfinite(sequence).all() and np.isfinite(stats).all() and
            np.isfinite(spectra).all(),'infinite/NaN native input')
    cache={'sequence':np.asarray(sequence,np.float32),
           'mask':np.asarray(mask,np.float32),
           'stats':np.asarray(stats,np.float32),
           'spectral':np.asarray(spectra,np.float32),
           'members':np.asarray([c['member'] for c in clusters]),
           'cluster_start_samples':np.asarray(geometry['cluster_start_samples'],np.int64)}
    require(len(cache['members'])>0,'no original native clusters')
    mdl=build_model('learned_gate',SEED)
    mdl.load_weights(weights)
    pr,counts=predict(mdl,cache,np.arange(len(clusters)))
    require(pr.shape==(len(counts),7),'invalid frozen count logits')
    starts=cache['cluster_start_samples']
    require(np.all(np.isfinite(pr)) and np.isin(counts,range(7)).all(),
            'invalid original freeze count')
    np.savez_compressed(output/'external_freeze_local_combo_predictions.npz',
        member=cache['members'],start_samples=starts,
        incumbent_K=counts,incumbent_probabilities=pr,
        frozen_source='freeze_local_combo')
    info={'status':'EXTERNAL_ORIGINAL_V8_STACK_AND_FROZEN_COUNT_PREDICTIONS',
          'full_S80_executed':False,'midi_labels_read':False,
          'audio_seconds_per_file':seconds,
          'files':[{'member':t.annotation_member,
                    'wav_sha256':digest(audio/('example'+str(i)+'_di.wav'))}
                   for i,t in enumerate(tracks,1)],
          'candidate_count':len(records),'native_group_count':len(clusters),
          'original_v86_floor':float(floor),'original_v88_retain_threshold':float(threshold),
          'count_model':'freeze_local_combo_no_neuron_transplant',
          'weights_sha256':digest(weights),
          'outputs_sha256':digest(output/'external_freeze_local_combo_predictions.npz'),
          'no_promotion':True,'no_exact_K_reported':True}
    (output/'report.json').write_text(json.dumps(info,indent=2,sort_keys=True)+'\n')
    print(json.dumps(info,indent=2),flush=True)
    return info
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--audio',type=Path,required=True)
    p.add_argument('--freeze-weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=6.)
    a=p.parse_args()
    run(a.sources,a.audio,a.freeze_weights,a.output,a.seconds)
