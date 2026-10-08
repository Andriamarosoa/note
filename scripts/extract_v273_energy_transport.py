"""Observational source/transport/survival audit on mono guitar-pickup energy.

We do not measure mechanical energy transport or solve Navier-Stokes.
For each observed STFT band b and adjacent frame t, use a PRE-only passive
continuation factor a_b to define R_b=E[b,t+1]-a_b*E[b,t].
A *descriptive* adjacent-band matching yields J_{b+1/2}, then
  R_b = (J_{b-1/2}-J_{b+1/2}) + B_b - D_b.
The identity is checked numerically. J is non-identifiable from mono audio:
its meaning here is a constrained observable, not a physical flux.
This source/transport/decay decomposition is deterministic and label-free.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.extract_v273_energy_flow import (
    SAMPLE_RATE,PRE_MS,POST_MS,_safe_segment,_band_energy
)
from scripts.summarize_v273_energy_flux_combined import FOLDS,require

EPS=1e-10
FAMILIES=("static","birth","transport","persistence","extinction")
def scalar_stat(result,prefix,arr):
    a=np.asarray(arr,float).reshape(-1)
    require(len(a)>0 and np.isfinite(a).all(),"empty/nonfinite stat: "+prefix)
    result[prefix+"_mean"]=float(np.mean(a))
    result[prefix+"_std"]=float(np.std(a))
    result[prefix+"_max"]=float(np.max(a))

def constrained_transport(residual):
    """Greedy nearest-band signed mass matching, exact conservation identity.

    Return signed transfer along b -> b+1 and leftover source/sink. Neither
    causality nor note identity follows from this matching convention.
    """
    R=np.asarray(residual,float)
    require(R.ndim==2 and R.shape[1]>=2 and np.isfinite(R).all(),"bad residual")
    left=R.copy()
    edges=np.zeros((R.shape[0],R.shape[1]-1),float)
    for t in range(len(R)):
        # Alternating sweep reduces ordering bias; transfer never crosses >1
        # band at once unless both adjacent matching steps are supported.
        for bands in (range(R.shape[1]-1),range(R.shape[1]-2,-1,-1)):
            for b in bands:
                a,c=left[t,b],left[t,b+1]
                if a < 0 < c:
                    move=min(-a,c)
                elif a > 0 > c:
                    move=-min(a,-c)
                else:
                    continue
                edges[t,b]+=move
                left[t,b]+=move
                left[t,b+1]-=move
    movement=np.zeros_like(R)
    movement[:,:-1]-=edges
    movement[:,1:]+=edges
    birth=np.maximum(left,0.)
    death=np.maximum(-left,0.)
    error=float(np.max(np.abs(R-(movement+birth-death))))
    scale=max(1.,float(np.max(np.abs(R))))
    require(error/scale<1e-10,"continuity identity failed")
    require(np.min(birth)>=0 and np.min(death)>=0,"signed sources")
    return edges,birth,death,error/scale

def normalized_energy(E):
    # Only a row-local scale, never labels or another event's energies.
    require(E.ndim==2 and np.isfinite(E).all() and np.min(E)>=0,"bad spectrum")
    return E/(np.median(np.sum(E,axis=1))+EPS)

def observable_from_energy(E,t):
    E=normalized_energy(np.asarray(E,float))
    t=np.asarray(t,float)
    require(E.shape[0]==len(t) and E.shape[1]>=4,"bad band-time axes")
    result={}
    pre=(t>=-60)&(t<-8)
    event=(t>=-8)&(t<=110)
    late=(t>=65)&(t<=130)
    require(pre.sum()>=3 and event.sum()>=10 and late.sum()>=3,"time windows")
    total=E.sum(axis=1)
    sharing=E/(total[:,None]+EPS)
    ent=-np.sum(sharing*np.log(sharing+EPS),axis=1)/np.log(E.shape[1])
    for name,m in (("pre",pre),("event",event),("late",late)):
        result[f"static__total_{name}"]=float(total[m].mean())
        result[f"static__entropy_{name}"]=float(ent[m].mean())
        result[f"static__dominant_{name}"]=float(np.max(sharing[m],axis=1).mean())
    result["static__event_pre_ratio"]=float(np.log((total[event].mean()+EPS)/(total[pre].mean()+EPS)))
    result["static__late_pre_ratio"]=float(np.log((total[late].mean()+EPS)/(total[pre].mean()+EPS)))
    result["static__mean_active_event_bands"]=float((sharing[event]>.05).sum(axis=1).mean())
    result["static__energy_concentration"]=float(np.sum(np.mean(sharing[event],axis=0)**2))

    ftime=(t[1:]+t[:-1])/2
    prem=(ftime>=-60)&(ftime<-8)
    mask=(ftime>=-8)&(ftime<=110)
    require(prem.sum()>=3 and mask.sum()>=10,"flow windows")
    floor=np.maximum(np.median(E[pre],axis=0)*.001,EPS)
    ratio=np.median((E[1:][prem]+floor)/(E[:-1][prem]+floor),axis=0)
    ratio=np.clip(ratio,.50,1.50)
    R=E[1:]-E[:-1]*ratio[None,:]
    J,B,D,identity_error=constrained_transport(R)
    J=J[mask];B=B[mask];D=D[mask]
    E0=E[:-1][mask];E1=E[1:][mask];time=ftime[mask]
    born=B.sum(axis=1);died=D.sum(axis=1)
    shifted=np.sum(np.abs(J),axis=1)
    raw=np.abs(R[mask]).sum(axis=1)
    posmass=float(B.sum());negmass=float(D.sum());tpmass=float(np.abs(J).sum())
    # Static controls deliberately independent of B,D,J.
    scalar_stat(result,"birth__frame_mass",born)
    result["birth__total_mass"]=posmass
    result["birth__fraction_of_residual"]=float(posmass/(raw.sum()+EPS))
    result["birth__active_bands"]=float(np.mean(np.sum(B>EPS,axis=1)))
    result["birth__band_concentration"]=float(np.sum((B.sum(axis=0)/(posmass+EPS))**2))
    scalar_stat(result,"extinction__frame_mass",died)
    result["extinction__total_mass"]=negmass
    result["extinction__fraction_of_residual"]=float(negmass/(raw.sum()+EPS))
    result["extinction__active_bands"]=float(np.mean(np.sum(D>EPS,axis=1)))
    result["extinction__band_concentration"]=float(np.sum((D.sum(axis=0)/(negmass+EPS))**2))
    scalar_stat(result,"transport__frame_mass",shifted)
    result["transport__total_mass"]=tpmass
    result["transport__fraction_of_residual"]=float(2*tpmass/(raw.sum()+EPS))
    result["transport__net_direction"]=float(np.sum(J)/(tpmass+EPS))
    result["transport__active_edges"]=float(np.mean(np.sum(np.abs(J)>EPS,axis=1)))
    result["transport__direction_reversals"]=float(np.mean(J[1:]*J[:-1]<0)) if len(J)>1 else 0.

    # Cause/effect lag: newly injected energy must still be observable in its
    # frequency band at a later time. It may belong to overlapping harmonics.
    for delay in (2,5,10):
        if len(B)<=delay:continue
        initial=B[:-delay]
        pers=np.minimum(E1[delay:],E1[:-delay])
        weight=np.sum(initial)+EPS
        result[f"persistence__birth_retain_{delay}"]=float(np.sum(initial*pers/(E1[:-delay]+floor+EPS))/weight)
        old=E1[:-delay]
        result[f"persistence__state_retain_{delay}"]=float(np.sum(np.minimum(old,E1[delay:]))/(np.sum(old)+EPS))
        sink=D[:-delay]
        result[f"extinction__sink_retained_{delay}"]=float(np.sum(sink*np.maximum(0,E1[:-delay]-E1[delay:])/(E1[:-delay]+floor+EPS))/(np.sum(sink)+EPS))
    for name,m in (("early",(time>=-8)&(time<35)),
                    ("middle",(time>=35)&(time<70)),
                    ("late",(time>=70)&(time<=110))):
        require(m.any(),"missing phase")
        result[f"birth__{name}_fraction"]=float(B[m].sum()/(posmass+EPS))
        result[f"extinction__{name}_fraction"]=float(D[m].sum()/(negmass+EPS))
        result[f"transport__{name}_fraction"]=float(np.abs(J[m]).sum()/(tpmass+EPS))
    result["persistence__passive_survival_mean"]=float(np.mean(ratio))
    result["persistence__passive_survival_std"]=float(np.std(ratio))
    result["persistence__positive_survival_band_fraction"]=float(np.mean(ratio>=1))
    result["persistence__birth_sink_balance"]=float((posmass-negmass)/(posmass+negmass+EPS))
    if len(B)>1:
        u,v=born[:-1],born[1:]
        result["persistence__birth_autocorrelation"]=float(np.corrcoef(u,v)[0,1]) if np.std(u)>EPS and np.std(v)>EPS else 0.
    else:result["persistence__birth_autocorrelation"]=0.
    require(identity_error < 1e-10,"continuity error")
    values=np.asarray(list(result.values()),float)
    require(np.isfinite(values).all(),"nonfinite observable")
    return result

def energy_observable(samples,start,perturb=None,seed=0):
    pre=int(round(PRE_MS*SAMPLE_RATE/1000))
    post=int(round(POST_MS*SAMPLE_RATE/1000))
    segment=_safe_segment(samples,int(start),pre,post)
    E,_,t=_band_energy(segment)
    if perturb:
        E=E.copy()
        mask=(t>=-8)&(t<=110)
        idx=np.flatnonzero(mask)
        if perturb=="scramble-time":
            E[idx]=E[np.random.default_rng(seed).permutation(idx)]
        elif perturb=="permute-bands":
            # Same permutation for all frames of a given event.
            E=E[:,np.random.default_rng(seed).permutation(E.shape[1])]
        else:raise ValueError(perturb)
    return observable_from_energy(E,t)

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--cases",type=Path,required=True)
    a.add_argument("--dataset",type=Path,required=True)
    a.add_argument("--fold",type=int,required=True)
    a.add_argument("--output",type=Path,required=True)
    args=a.parse_args()
    require(args.fold in FOLDS and not args.output.exists(),"bad fold/overwrite")
    cases=[json.loads(x) for x in args.cases.read_text().splitlines() if x.strip()]
    require(len(cases)==488,"frozen cohort drift")
    rows=[r for r in cases if int(r["fold"])==args.fold]
    require(rows and all(int(r["true_K"]) in (2,3) for r in rows),"K2/K3 cohort required")
    wanted={r["recording_id"] for r in rows}
    tracks={t.annotation_member:t for t in index_guitarset(args.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"missing audio")
    audio={}
    for member,track in tracks.items():
        wav=decode_pcm16_mono_wav(track.audio_zip,track.audio_member)
        require(int(wav.sample_rate)==SAMPLE_RATE,"sample rate drift")
        audio[member]=np.asarray(wav.samples,np.float64)/32768.
    output=[]
    for case in rows:
        x=audio[case["recording_id"]]
        start=int(case["start_sample"]);ident=int(case["row_id"])
        feat=energy_observable(x,start)
        scrambled=energy_observable(x,start,"scramble-time",seed=ident+17064)
        permuted=energy_observable(x,start,"permute-bands",seed=ident+17064)
        output.append(dict(row_id=ident,fold=args.fold,recording_id=case["recording_id"],
                           start_sample=start,true_k=int(case["true_K"]),features=feat,
                           scrambled=scrambled,permuted=permuted))
    cols=sorted(output[0]["features"])
    require(all(all(sorted(o[k])==cols for k in ("features","scrambled","permuted")) for o in output),"schema drift")
    args.output.mkdir(parents=True)
    (args.output/"rows.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in output))
    report={"experiment":"v273_energy_transport_extract","status":"completed","fold":args.fold,
            "rows":len(output),"features":len(cols),"families":{p:sum(k.startswith(p+"__") for k in cols) for p in FAMILIES},
            "labels_during_extraction":False,"player05_used":False,"fold3_used":False,
            "equation":"E(t+1)-a_pre*E(t)=neighbor_transfer+birth-death",
            "interpretation":"adjacent log-band matching proxy; not physically measured Navier-Stokes fluid flux",
            "time_window_ms":[-80,160],"automatic_promotion":False}
    (args.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,sort_keys=True),flush=True)

if __name__=="__main__":main()
