"""S116 reconstruct the original S4 111-D control feature schema on unseen GOAT.

58 original waveform harmonic summaries, 46 authentic ownership/timing
descriptors, seven one-hot frozen reference K votes. No MIDI, target, labels.
Feature COLUMN ORDER is a research interpretation of the historical protocol;
an original S4 checkpoint replay is required to prove exact equivalence.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.extract_v273_open_cohort import ownership
from scripts.extract_v273_harmonic_trajectory import _frames,SAMPLE_RATE
from scripts.s85_external_pr16_policies import canon_name

def require(ok,msg):
    if not ok:raise ValueError(msg)
def canonical(member):
    value=str(member)
    return value if value in ('example1_di.wav','example2_di.wav') else canon_name(value)
def run(s97,s85,out):
    require(not out.exists(),'refuse original artifact overwrite')
    with np.load(s97,allow_pickle=False) as z:
        require(set(['row_id','summary58','member','start_samples','end_samples']).issubset(z.files),'missing signal fields')
        require(not set(z.files).intersection({'true_K','truth','target','label','midi'}),'labels in external features')
        row_id=z['row_id']; summary=z['summary58']; m=z['member'].astype(str)
        starts=z['start_samples'].astype(np.int64); ends=z['end_samples'].astype(np.int64)
    with np.load(s85,allow_pickle=False) as z:
        require(set(['member','start_samples','end_samples','incumbent_freeze_K']).issubset(z.files),'missing frozen reference contract')
        require(not set(z.files).intersection({'true_K','truth','target','label','midi'}),'annotations present in reference')
        members=np.asarray([canonical(v) for v in z['member']])
        keys=[(name,int(start),int(end)) for name,start,end in zip(members,z['start_samples'],z['end_samples'])]
        require(len(set(keys))==len(keys),'duplicate S85 event keys')
        baseline={key:int(k) for key,k in zip(keys,z['incumbent_freeze_K'])}
    n=len(starts)
    require(n==57 and summary.shape==(n,58),'GOAT feature signal mismatch')
    keys=[(str(mm),int(ss),int(ee)) for mm,ss,ee in zip(m,starts,ends)]
    require(len(set(keys))==n and all(k in baseline for k in keys),'external alignment mismatch')
    parent=np.asarray([baseline[k] for k in keys],np.int8)
    require(np.isin(parent,np.arange(7)).all(),'bad K')
    dummy=np.zeros(SAMPLE_RATE,dtype=np.float64)
    _,times=_frames(dummy,SAMPLE_RATE//2)
    require(len(times)==42,'42 frame time contract drift')
    geometry=np.empty((n,46),np.float32)
    overlaps=[]
    for name in np.unique(m):
        idx=np.flatnonzero(m==name)
        ordered=idx[np.argsort(starts[idx],kind='stable')]
        local_starts,local_ends=starts[ordered],ends[ordered]
        if (local_starts[1:]<=local_ends[:-1]).any():
            overlaps.append({'member':name,'conflicts':int((local_starts[1:]<=local_ends[:-1]).sum())})
            continue
        for j,k in enumerate(ordered):
            query=starts[k]+np.rint(times*SAMPLE_RATE/1000).astype(np.int64)
            mask=ownership(local_starts,local_ends,j,query).astype(np.float32)
            before=(starts[k]-local_ends[j-1])/10584 if j else 1.
            after=(local_starts[j+1]-starts[k])/10584 if j+1<len(ordered) else 1.
            geometry[k]=np.r_[mask,(ends[k]-starts[k])/1764.,
                            np.clip(before,0,1),np.clip(after,0,1),mask.mean()]
    result={'status':'S116_EXTERNAL_S4_CONTROL111_AUDIT',
            'event_count':n,'member_overlap_conflicts':overlaps,
            'S4_111D_schema_ready':not overlaps,'S4_111D_original_column_equivalence_proven':False,
            'S4_owned_527D_ready':False,'external_exact_K_scores':None,
            'independent_S80_validation':False,'promotion':False}
    out.mkdir(parents=True)
    if not overlaps:
        control=np.column_stack([summary,geometry,np.eye(7,dtype=np.float32)[parent]]).astype(np.float32)
        require(control.shape==(n,111) and np.isfinite(control).all(),'S4 111D shape or finite drift')
        require(np.all((geometry[:,:42]==0)|(geometry[:,:42]==1)),'ownership mask not boolean')
        np.savez_compressed(out/'external_s4_control111.npz',row_id=row_id,
                            control111=control,freeze_K=parent,geometry46=geometry)
        print('S116 PASS: authentic 58+46+7 features on 57 GOAT audio events; no truth')
    else:
        print('S116 BLOCKED: external V84 event overlap violates original GuitarSet ownership invariant')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--s97',type=Path,required=True)
    p.add_argument('--s85',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.s97,a.s85,a.out),indent=2))
