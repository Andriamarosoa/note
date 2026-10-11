"""Run ORIGINAL PR16 neural weights on independent guitar audio, WITHOUT ANY labels.

This is an external inference *component* smoke test, NOT S80 validation.
The audio-only onset proposals are NOT the original V8/native event generator;
thus no K accuracy from them may be presented as an official model score.
"""
from __future__ import annotations
import argparse, hashlib, json, wave
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks, stft
from scripts.extract_v273_harmonic_trajectory import _frames, features_from_spectrogram
from scipy.special import erf, softmax
def numpy_forward(x,weights,batch=256):
    out=[]
    for start in range(0,len(x),batch):
        a=np.asarray(x[start:start+batch],np.float64)
        for layer in range(3):
            a=a@weights[2*layer]+weights[2*layer+1]
            if layer<2:a=.5*a*(1+erf(a/np.sqrt(2)))
        out.append(softmax(a,axis=1))
    return np.concatenate(out)
def ownership(starts,ends,i,query):
    starts,ends,query=map(np.asarray,(starts,ends,query))
    distance=np.maximum(np.maximum(starts[i]-query,query-ends[i]),0)
    own=distance<=882
    if i:
        prev=np.maximum(np.maximum(starts[i-1]-query,query-ends[i-1]),0)
        own &= distance<prev
    if i+1<len(starts):
        nxt=np.maximum(np.maximum(starts[i+1]-query,query-ends[i+1]),0)
        own &= distance<=nxt
    return own

SR=44100
def require(x,m):
    if not x: raise ValueError(m)
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_wav(path):
    with wave.open(str(path),'rb') as r:
        require(r.getframerate()==SR and r.getsampwidth()==2 and r.getnchannels()==1,'WAV must be mono 44.1kHz PCM16')
        wave_data=r.readframes(r.getnframes())
    return np.frombuffer(wave_data,'<i2').astype(np.float64)/32768.
def candidate_starts(samples, max_events=60):
    # Fully label-free, fixed acoustic energy-rise candidates, explicitly only
    # a temporary source for PR16 deployment smoke (not native V8).
    _,t,Z=stft(samples,fs=SR,nperseg=2048,noverlap=1792,
               boundary='zeros',padded=True)
    energy=np.log1p(150*np.abs(Z))
    strength=np.maximum(0,np.diff(energy,axis=1)).mean(axis=0)
    median=np.median(strength)
    scale=np.median(np.abs(strength-median))+1e-8
    peak,_=find_peaks(strength,distance=11,prominence=1.2*scale)
    # forbid synthetic events near segment boundaries
    starts=(np.round(t[peak+1]*SR).astype(np.int64))
    starts=starts[(starts>int(.15*SR))&(starts<len(samples)-int(.2*SR))]
    # stable deterministic sample stride, NOT class-dependent
    if len(starts)>max_events:
        keep=np.linspace(0,len(starts)-1,max_events,dtype=int)
        starts=starts[keep]
    return starts

def infer(model_dir, audio_root, output, max_events):
    require(not output.exists(),'do not overwrite experiment')
    records=[]
    for filename in ('example1_di.wav','example2_di.wav'):
        path=audio_root/filename
        samples=load_wav(path);starts=candidate_starts(samples,max_events)
        require(15<=len(starts)<=max_events,'audio onset candidate failure '+filename)
        ends=starts.copy()
        summary=[];seq=[];geom=[];names=None
        for i,s in enumerate(starts):
            spec,t=_frames(samples,int(s))
            d,activation=features_from_spectrogram(spec,t,return_states=True)
            keys=sorted(d)
            if names is None:names=keys
            require(keys==names and activation.shape==(42,49),'PR16 feature schema')
            summary.append([d[k] for k in keys]);seq.append(np.log1p(activation))
            query=s+np.rint(t*SR/1000).astype(np.int64)
            mask=ownership(starts,ends,i,query)
            before=(s-ends[i-1])/10584 if i else 1.
            after=(starts[i+1]-s)/10584 if i+1<len(starts) else 1.
            geom.append(np.r_[mask,0,np.clip(before,0,1),np.clip(after,0,1),mask.mean()])
        require(len(names)==58,'wrong acoustic features')
        rec=dict(name=filename,sha256=sha(path),sample_rate=SR,
                 audio_seconds=len(samples)/SR,starts=starts,
                 summary=np.asarray(summary,np.float32),
                 sequence=np.asarray(seq,np.float32),
                 geometry=np.asarray(geom,np.float32))
        records.append(rec)
    raw58=np.concatenate([r['summary'] for r in records])
    rawtraj=np.concatenate([np.column_stack((r['summary'],r['sequence'].reshape(len(r['summary']),-1),
                                             r['geometry'])) for r in records])
    require(raw58.shape[1]==58 and rawtraj.shape[1]==2162,'PR16 input dimensions')
    predlist=[];pnames=[];normalizer_shas={};weight_shas={}
    for rep,raw in (('summary58',raw58),('trajectory',rawtraj)):
        norm=model_dir/(rep+'-normalizer.npz')
        with np.load(norm,allow_pickle=False) as z:
            mean,scale=z['mean'],z['scale']
        require(len(mean)==raw.shape[1] and np.all(scale>0),'trained normalizer mismatch')
        x=raw.astype(np.float32)
        x-=mean.astype(np.float32);x/=scale.astype(np.float32);np.clip(x,-6,6,out=x)
        normalizer_shas[rep]=sha(norm)
        for weight in (1,1.5):
            for epoch in (4,8,12):
                key=f'{rep}__poly{weight:g}__epoch{epoch}'
                source=model_dir/(key+'.weights.npz')
                with np.load(source,allow_pickle=False) as z:
                    w=[z['w'+str(i)] for i in range(len(z.files))]
                require(len(w)==6,'invalid original frozen PR16 MLP')
                p=numpy_forward(x,w)
                require(p.shape==(len(raw),7) and np.isfinite(p).all() and
                        np.max(np.abs(p.sum(1)-1))<1e-5,'invalid prediction')
                predlist.append(p);pnames.append(key);weight_shas[key]=sha(source)
    probabilities=np.stack(predlist,axis=1)
    require(probabilities.shape[1:]==(12,7),'not twelve original PR16 weights')
    # Externally inferred outputs contain no true_K and no annotation fields.
    src=[];names=[]
    for r in records:
        src.extend(r['starts'].tolist());names.extend([r['name']]*len(r['starts']))
    output.mkdir(parents=True)
    np.savez_compressed(output/'label_blind_pr16_predictions.npz',
        member=np.asarray(names),start_samples=np.asarray(src,np.int64),
        probabilities=probabilities,producer_names=np.asarray(pnames))
    manifest={'status':'EXTERNAL_PR16_NEURAL_PRODUCER_INFERENCE_COMPLETED_NOT_S80',
       'label_fields_present':False,'midi_accessed':False,'s80_scored':False,
       'candidate_source':'TEMPORARY_LABEL_BLIND_SPECTRAL_FLUX_NOT_NATIVE_V8',
       'producer_model_fold':0,'producer_count':12,'n_external_events':len(names),
       'files':[{k:v for k,v in r.items() if k in ('name','sha256','sample_rate','audio_seconds')}|
                {'n_events':len(r['starts'])} for r in records],
       'normalizer_sha256':normalizer_shas,'model_weight_sha256':weight_shas,
       'prediction_sha256':sha(output/'label_blind_pr16_predictions.npz'),
       'model_validation_status':'NOT_TESTED_NEEDS_S46_S72_S80_FULL_RUNTIME'}
    (output/'report.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:manifest[k] for k in ('status','n_external_events','files','model_validation_status')},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--audio',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-events',type=int,default=60)
    a=p.parse_args()
    infer(a.model,a.audio,a.output,a.max_events)
