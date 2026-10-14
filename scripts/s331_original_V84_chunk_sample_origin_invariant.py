"""S331 verify trained V84 causal stream sample/time invariance independently of XML.

The exact same deterministic 44.1kHz synthetic guitar-like waveform is
scored using different chunk sizes, stateful vs fallback recomputation.
Any systematic translation of sample indices with chunk size, warmup,
or model runner is a BUG in actual inference plumbing; do not invent
a +40ms shift from the external XML audit.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from causal_note.v84_onset_predictor import V84OnsetOnlyKerasPredictor
from scripts.evaluate_v273_fresh_player05 import proposal_args
RECEPTIVE=4093
SR=44100
def need(x,msg):
    if not x:raise ValueError(msg)
def sha(x):return hashlib.sha256(Path(x).read_bytes()).hexdigest()
def waveform(n=44100):
    t=np.arange(n,dtype=np.float64)/SR
    x=np.zeros(n,np.float64)
    for at in [.10,.40,.71]:
        a=np.maximum(t-at,0)
        pluck=np.sin(2*np.pi*220*a)+.45*np.sin(2*np.pi*440*a)
        envelope=(t>=at)*np.exp(-a*25)*(1-np.exp(-a*450))
        x+=pluck*envelope
    return np.asarray(np.clip(x*.3,-1,1),np.float32)
def run_one(model,audio,chunk,fast):
    predictor=V84OnsetOnlyKerasPredictor.from_path(
         str(model),receptive_field=RECEPTIVE,use_stateful=fast)
    predictor.warm_up(chunk)
    predictor.reset()
    values=[];multiplicity=[]
    for start in range(0,len(audio),chunk):
        slice_=audio[start:start+chunk]
        score=predictor.predict_chunk(tuple(map(float,slice_)),start_sample=start)
        need(len(score.onset_presence)==len(slice_) and
             len(score.onset_multiplicity)==len(slice_),
             'V84 does not preserve one score per 44.1kHz input sample')
        values.extend(score.onset_presence)
        multiplicity.extend(score.onset_multiplicity)
    return np.asarray(values,dtype=np.float64),np.asarray(multiplicity,dtype=np.float64),predictor.stateful_enabled
def run(sources,out):
    need(not out.exists(),'S331 source-only audit immutable')
    path=proposal_args(sources).base_model
    need(Path(path).is_file(),'original V84 model absent, cannot test source stack')
    sig=waveform()
    records=[]
    for chunk,fast in [(512,True),(2048,True),(1024,False)]:
        presence,mult,actual=run_one(path,sig,chunk,fast)
        need(presence.shape==(len(sig),) and mult.shape==(len(sig),3)
             and np.isfinite(presence).all() and np.isfinite(mult).all(),
             'wrong V84 model output temporal alignment')
        records.append((presence,mult))
        print(json.dumps(dict(chunk=chunk,asked_stateful=fast,
          actual_stateful=actual,peak_sample=int(np.argmax(presence)),
          peak_time_ms=float(np.argmax(presence)*1000/SR))),flush=True)
    p0,m0=records[0]
    differences=[]
    for p,m in records[1:]:
        err=float(np.max(np.abs(p-p0)))
        merr=float(np.max(np.abs(m-m0)))
        differences.append(dict(onset_presence_max_abs=err,
                                multiplicity_max_abs=merr))
    report=dict(status='S331_ORIGINAL_TRAINED_V84_CHUNK_STABILITY_SAMPLE_ORIGIN',
        source_V84_model_SHA256=sha(path),
        duration_seconds=len(sig)/SR,
        onsets_generated_at_samples=[int(round(i*SR)) for i in [.10,.40,.71]],
        compares='512 stateful vs 2048 stateful vs 1024 fallback',
        tested_chunk_sizes=[512,2048,1024],
        max_abs_diff_vs_512=differences,
        all_three_predictors_exact_sample_count=True,
        no_implicit_40ms_timestamp_addition=True,
        no_XML_read=True,no_retraining=True,no_promotion=True)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    need(all(d['onset_presence_max_abs']<2e-4 and
             d['multiplicity_max_abs']<2e-4 for d in differences),
         'ORIGINAL V84 chunk execution mismatch: potential systemic temporal inference bug')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.sources,a.out)
