"""S97: reconstruct original S45 acoustic/temporal/geometry inputs on unseen audio.

This is an ORIGINAL-feature reconstruction, not a replacement for the
missing S43/S44 producer decisions. No ground truth or MIDI is accepted.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scripts.s82_external_pr16_neural_inference import load_wav
from scripts.extract_v273_harmonic_trajectory import _frames,features_from_spectrogram
from scripts.loop_v273_birth_flow import flow_profile
from scripts.loop_v273_native_risk import designs
from scripts.v273_extended_head_bank import audio_temporal_views

BAD=('true_K','truth','label','labels','y','target','midi')
def require(v,why):
    if not v: raise ValueError(why)
def read(path,required):
    with np.load(path,allow_pickle=False) as z:
        require(not set(z.files).intersection(BAD),'annotation field forbidden')
        require(set(z.files)==set(required),'wrong external input contract')
        return {k:z[k].copy() for k in z.files}
def geometry(members,starts,ends):
    n=len(starts);last=np.empty(n,np.float32)
    for member in np.unique(members):
        at=np.flatnonzero(members==member)
        order=at[np.argsort(starts[at],kind='stable')]
        last[order[0]]=2.
        if len(order)>1:
            last[order[1:]]=np.clip((starts[order[1:]]-starts[order[:-1]])/44100.,0,2)
    return np.column_stack([(ends-starts)/1764.,np.log1p(last),np.minimum(last,.2)/.2]).astype(np.float32)
def extract(groups_path,audio_root,out,decisions_path=None):
    require(not out.exists(),'immutable output already exists')
    groups=read(groups_path,('member','starts','ends'))
    members=groups['member'].astype(str);starts=groups['starts'].astype(np.int64)
    ends=groups['ends'].astype(np.int64);n=len(starts)
    require(n>0 and len(members)==n and len(ends)==n,'inconsistent native events')
    require(np.all((starts>0)&(ends>=starts)&(ends-starts<=1764)),'native event bounds')
    require(len(set((m,int(s)) for m,s in zip(members,starts)))==n,'duplicate native event')
    summaries=np.zeros((n,58),np.float32);sequences=np.zeros((n,42,49),np.float32)
    for member in np.unique(members):
        require(Path(member).name==member and member in ('example1_di.wav','example2_di.wav'),'external audio allowlist')
        samples=load_wav(audio_root/member)
        rows=np.flatnonzero(members==member)
        for i in rows:
            require(ends[i]+int(.2*44100)<len(samples),'event outside original WAV')
            spec,t=_frames(samples,int(starts[i]))
            d,act=features_from_spectrogram(spec,t,return_states=True)
            require(len(d)==58 and act.shape==(42,49),'historical signal feature shape')
            summaries[i]=[d[k] for k in sorted(d)]
            sequences[i]=np.log1p(act)
    require(np.isfinite(summaries).all() and np.isfinite(sequences).all(),'nonfinite audio evidence')
    flow=flow_profile(sequences)
    geom=geometry(members,starts,ends)
    morphology,physics=audio_temporal_views(sequences)
    require(morphology.shape==(n,490) and physics.shape==(n,294), 'original temporal morphology drift')
    require(np.isfinite(morphology).all() and np.isfinite(physics).all(), 'nonfinite temporal morphology')
    # Independent original audio-only signal descriptors; no pseudo K decision sources.
    out.mkdir(parents=True)
    row_id=np.asarray([f'{m}:{int(s)}' for m,s in zip(members,starts)])
    np.savez_compressed(out/'original_s45_audio_only.npz',row_id=row_id,
       summary58=summaries,flow245=flow,geometry3=geom,
       sequence42x49=sequences,morphology490=morphology,physics294=physics,member=members,
       start_samples=starts,end_samples=ends)
    full=False
    if decisions_path is not None:
        decisions=read(decisions_path,('row_id','baseline_K','global_K','poly_K'))
        require(np.array_equal(row_id,decisions['row_id']),'decision time or audio mismatch')
        b,g,p=[decisions[k].astype(np.int8) for k in ('baseline_K','global_K','poly_K')]
        require(all(np.isin(a,range(7)).all() for a in (b,g,p)),'invalid historical S5 K0..6')
        # 'designs' also verifies the original S11 32D contract.
        prepared={'native_position':np.arange(n),'global_index':np.arange(n),
                  'features':summaries}
        votes=designs(prepared,np.arange(n),b,g,p,geom)['votes_time']
        X=np.column_stack((votes,summaries,flow)).astype(np.float32)
        require(X.shape==(n,335),'original S45 acoustic 335D drift')
        np.savez_compressed(out/'s45_335_features.npz',
                            row_id=row_id,acoustic_features=X)
        full=True
    report={'status':'S97_ORIGINAL_AUDIO_FEATURES_REBUILT',
            'external_audio_events':n,'raw_dimensions':[58,245,3],
            'original_morphology490_ready':True,'original_physics294_ready':True,
            'S45_335_features_ready':full,'requires_true_S5_global_poly_sources':not full,
            'midi_read':False,'S43_S44_families_generated':False,
            'S80_independent_score':None,'promotion':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
def selftest():
    n=4;ids=np.arange(n)
    m=np.asarray(['example1_di.wav']*2+['example2_di.wav']*2)
    s=np.array([10000,20000,12000,30000])
    e=s+400
    geom=geometry(m,s,e)
    assert geom.shape==(n,3) and np.isfinite(geom).all()
    assert np.allclose(geom[[0,2],1],np.log(3.))
    summary=np.zeros((n,58),np.float32)
    prepared={'native_position':ids,'global_index':ids,'features':summary}
    votes=designs(prepared,ids,np.array([0,2,3,1]),np.array([1,2,4,1]),
                  np.array([0,3,3,2]),geom)['votes_time']
    seq=np.zeros((n,42,49),np.float32);f=flow_profile(seq)
    morph,phys=audio_temporal_views(seq)
    assert morph.shape==(n,490) and phys.shape==(n,294)
    assert np.column_stack((votes,summary,f)).shape==(n,335)
    print('S106 PASS: original geometry3, vote32, summary58, flow245, morph490, physics294, sequence42x49')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--groups',type=Path);p.add_argument('--audio',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--decisions',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        if not all((a.groups,a.audio,a.output)):p.error('missing groups/audio/output')
        print(json.dumps(extract(a.groups,a.audio,a.output,a.decisions),indent=2))
