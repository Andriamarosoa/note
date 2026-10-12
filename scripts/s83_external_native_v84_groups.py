"""S83: freeze original V8.4 onset candidate groups from unseen GOAT WAVs.

This is a label-blind external INFRASTRUCTURE test, NOT S80 validation.
It replaces the S82 temporary spectral-flux group generator with the
original frozen V8.4 neural proposal stream and native 40ms grouping.
"""
from __future__ import annotations
import argparse, hashlib, json, wave, zipfile
from pathlib import Path
import numpy as np
from causal_note.v84_onset_predictor import V84OnsetOnlyKerasPredictor
from scripts.train_v86_state_transition_proposals import (
    _candidate_groups, RECEPTIVE_FIELD, CHUNK_SIZE
)
from scripts.evaluate_v90_cardinality_realization import _clusters_window
from scripts.train_v90_structured_cluster_cardinality import CLUSTER_WINDOW_MS

SR=44100
def require(c,msg):
    if not c:raise RuntimeError(msg)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for q in iter(lambda:f.read(1<<20),b''):h.update(q)
    return h.hexdigest()
def audio_pcm(path,seconds):
    with wave.open(str(path),'rb') as wf:
        require(wf.getframerate()==SR and wf.getnchannels()==1 and
                wf.getsampwidth()==2 and wf.getcomptype()=='NONE','invalid source WAV')
        n=min(wf.getnframes(),int(seconds*SR))
        raw=wf.readframes(n)
    samples=np.frombuffer(raw,'<i2').copy()
    require(len(samples)>2*SR,'short audio')
    return samples
def scores_from_frozen_v84(predictor,samples):
    predictor.reset()
    presence=[];multiplicity=[]
    for lo in range(0,len(samples),CHUNK_SIZE):
        x=tuple(float(v)/32768. for v in samples[lo:lo+CHUNK_SIZE])
        out=predictor.predict_chunk(x,start_sample=lo)
        presence.extend(out.onset_presence);multiplicity.extend(out.onset_multiplicity)
    require(len(presence)==len(samples),'frozen V84 output sample coverage')
    require(len(multiplicity)==len(samples),'V84 multiplicity coverage')
    return {'presence':np.asarray(presence,np.float32),
            'multiplicity':np.asarray(multiplicity,np.float32)}
def run(model,audio,output,seconds,max_events):
    require(not output.exists(),'output already exists; immutable evidence')
    cfg=json.loads((model/'v86'/'report.json').read_text())
    floor=float(cfg['configuration']['candidate_floor'])
    require(cfg['configuration'].get('candidate_floor_selected_on_locked_validation') is False,
            'frozen V86 threshold provenance failed')
    require(.01<floor<.4,'unexpected V86 floor')
    mdl=model/'v84'/'control.epoch-01.keras'
    require(mdl.is_file(),'missing original V84 checkpoint')
    predictor=V84OnsetOnlyKerasPredictor.from_path(str(mdl),receptive_field=RECEPTIVE_FIELD)
    predictor.warm_up(CHUNK_SIZE)
    records=[];members=[];starts=[];ends=[];detail=[]
    for name in ('example1_di.wav','example2_di.wav'):
        path=audio/name
        s=audio_pcm(path,seconds)
        stream=scores_from_frozen_v84(predictor,s)
        proposals=_candidate_groups(stream,floor)
        origin=len(records)
        # Clusters only need sample/member/arrangement/indices.
        local=[{'member':name,'arrangement':'comp','sample':int(p['sample'])}
               for p in proposals]
        clusters=_clusters_window(local,CLUSTER_WINDOW_MS)
        require(0<len(clusters),'native V84 returned no candidate groups: '+name)
        group_events=[]
        for cl in clusters:
            p=[int(local[i]['sample']) for i in cl['indices']]
            start,end=min(p),max(p)
            if start<=int(.15*SR) or end>=len(s)-int(.2*SR):continue
            group_events.append((start,end))
        require(len(group_events)>0,'no usable native groups')
        selected=group_events[:max_events]
        for x,y in selected:
            members.append(name);starts.append(x);ends.append(y)
        detail.append({'filename':name,'raw_wav_sha256':digest(path),
             'processed_seconds':len(s)/SR,'original_candidates':len(proposals),
             'native_clusters':len(clusters),'usable_clusters':len(group_events),
             'saved_clusters':len(selected),
             'floor':floor})
        print(json.dumps({'track':name,'native_clusters':len(clusters),
             'saved_clusters':len(selected)},sort_keys=True),flush=True)
    output.mkdir(parents=True)
    np.savez_compressed(output/'native_groups.npz',
        member=np.asarray(members),starts=np.asarray(starts,np.int64),
        ends=np.asarray(ends,np.int64))
    report={'status':'NATIVE_V84_GROUPS_FROM_EXTERNAL_AUDIO_LABEL_BLIND',
            'not_full_s80_inference':True,'not_model_validation':True,
            'used_midi_labels':False,'model':'ORIGINAL_FROZEN_V84',
            'source_checkpoint_sha256':digest(mdl),
            'group_generator':'ORIGINAL_V86_PEAKS_AND_V90_40MS_CLUSTERING',
            'v86_floor':floor,'processed_seconds_each':seconds,
            'truncated_audio_for_smoke':True,'groups_count':len(starts),
            'files':detail,'groups_sha256':digest(output/'native_groups.npz')}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    require(len(starts)==len(ends)==len(members)>0,'native groups missing')
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--audio',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--max-events',type=int,default=100)
    a=p.parse_args()
    run(a.sources,a.audio,a.output,a.seconds,a.max_events)
