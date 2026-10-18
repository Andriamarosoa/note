"""S733 Onset-only actual FIT80 training and CAL32 causal event evaluation.

No other target is learned. The signed S708, its K logits, Sustain, Release,
the S707 NoteOff policy and production LIVE are immutable. S731 generates
only proposal evidence from earlier emitted source state. All FIT risk profile
features exclude the evaluated composition and CAL has FIT-only profiles.
A label never enters real audio inference; annotation opens after events.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from causal_note.strict_live import SR
from causal_note.s705_temporal_heads import predict_track
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s731_auditory_scene import scene_scores
from causal_note.s733_one_target_selector import (
    OneTargetClassConditionalSelector, target_loss,
    SingleTargetContract, ADOPT, SUPPRESS)
from scripts.s613_prepare_strict_live import download, labels, digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s706_train_sharp_temporal_loops import (
    record, load_model, FIT_COMPS, CAL_COMPS, HELD_COMPS)
from scripts.s705_train_temporal_loops import setup
from scripts.s704_train_joint_loops import targets
from scripts.s704_joint_metrics import SCORES,metric,aggregate,target_rank
from scripts.s731_auditory_scene_calibration import emit

REFERENCE="S708_frozen"
ARMS=("onset_adopt","onset_guarded_both")
H=5
MAX_TRAIN_ROWS_PER_RECORDING=1800

def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")

def acoustic_candidates(post,k,native,scene):
    """No oracle, no future. Shape [T,49,5] + context [T,49,16]."""
    t=len(post)
    on=post[:,:,0];sustain=post[:,:,1];release=post[:,:,2]
    original=native[:,:,0]
    novelty=np.stack([e.novelty for e in scene])
    explained=np.stack([e.explained for e in scene])
    ownership=np.stack([e.old_ownership for e in scene])
    n_support=np.stack([e.supports for e in scene]).astype(np.float32)/8
    kroom=np.array([e.count_room for e in scene],np.float32)[:,None]
    oldon=np.concatenate([on[:1],on[:-1]],axis=0)
    olds=np.concatenate([sustain[:1],sustain[:-1]],axis=0)
    rise=np.maximum(on-oldon,0.)
    rise_s=np.maximum(sustain-olds,0.)
    # H4 supplies actual missing-onset proposals. It is NOT a decision;
    # a trained validator can reject it. Hard-coded proposals are all frozen.
    trusted=(novelty>.08)&(sustain>.50)&(n_support>=.25)&(on>.035)
    bright=np.where(trusted,np.maximum(on,.82),on)
    candidate=np.stack([
        on,
        np.clip(on+.65*novelty,0,1),
        original,
        np.clip(on+.35*novelty+.25*rise+.15*np.maximum(original-on,0),0,1),
        bright],axis=-1).astype(np.float32)
    entropy=-(np.maximum(k,1.e-8)*np.log(np.maximum(k,1.e-8))).sum(-1)
    ctx=np.stack([
        on,sustain,release,original,novelty,explained,ownership,
        n_support,rise,rise_s,
        np.broadcast_to(kroom,(t,49)),
        np.broadcast_to(np.argmax(k,axis=1)[:,None]/6,(t,49)),
        np.broadcast_to(entropy[:,None]/np.log(7),(t,49)),
        np.broadcast_to(np.linspace(0,1,49,dtype=np.float32)[None,:],(t,49)),
        np.broadcast_to((k[:,2:].sum(-1))[:,None],(t,49)),
        np.broadcast_to(np.maximum(0.,on-original),(t,49))
        ],axis=-1).astype(np.float32)
    # Candidate gate based ONLY on observed frozen evidence. This prevents
    # a label-driven "candidate" from accidentally entering inference.
    eligible=(candidate.max(-1)>.07)|(rise_s>.035)
    return candidate,ctx,eligible

def head_schema(n):
    head_type=np.zeros((n,H,4),np.float32)
    head_type[:,0,0]=1.
    head_type[:,1:,1]=1.
    head_mask=np.ones((n,H),bool)
    kmask=np.ones((n,H,7),bool)
    # Only baseline and sustain-agnostic proposal can inform hypothesized K0.
    kmask[:,1:,0]=False
    return head_type,head_mask,kmask

def baseline_input(spec,archive,root,heads,selection,offset,network,policy,release):
    item=record(spec,archive,heads,selection,offset)
    item["native_features"]=item["x"][:,:,:126].copy()
    p,k=predict_track(network,item["x"])
    post=(policy.blend*p+(1-policy.blend)*item["native_post"]).astype(np.float32)
    original=emit(item,post,k,policy,offset,release)
    scene=scene_scores(item["x"],post,k,original,item["decision_ends"])
    candidates,ctx,eligible=acoustic_candidates(post,k,item["native_post"],scene)
    return item,post,k,original,candidates,ctx,eligible

def collected_examples(proposals,ctx,eligible,y,ktrue,kprobs,source,seed):
    # Output rows remain finite and all pitches used for truth scoring.
    rng=np.random.default_rng(seed)
    positives=(y==1)&eligible
    hard=(proposals.max(-1)>.15)|((ctx[:,:,1]>.45)&(ctx[:,:,8]>.02))
    negatives=eligible&(~positives)
    # All viable positive samples plus random negative hard/easy contexts.
    pos=np.flatnonzero(positives.ravel())
    neg=np.flatnonzero(negatives.ravel())
    npos=min(len(pos),MAX_TRAIN_ROWS_PER_RECORDING//2)
    if len(pos)>npos:pos=np.sort(rng.choice(pos,npos,replace=False))
    nneg=min(len(neg),MAX_TRAIN_ROWS_PER_RECORDING-len(pos))
    if len(neg)>nneg:
        sample_hard=neg[hard.ravel()[neg]]
        target_hard=min(len(sample_hard),int(nneg*.75))
        picked=rng.choice(sample_hard,target_hard,replace=False) if target_hard else np.array([],dtype=int)
        others=np.setdiff1d(neg,picked,assume_unique=False)
        picked_rest=rng.choice(others,min(nneg-len(picked),len(others)),replace=False)
        neg=np.sort(np.r_[picked,picked_rest])
    ix=np.sort(np.r_[pos,neg])
    _,n_pitch=eligible.shape
    times,pitches=np.divmod(ix,n_pitch)
    pp=proposals[times,pitches]
    yy=y[times,pitches].astype(np.uint8)
    kk=ktrue[times]
    base=pp[:,0]; alternatives=pp[:,1:].max(-1)
    # One target at a time: tune ADD or SUPPRESS, never another channel.
    actions=np.zeros(len(ix),np.int64)
    actions[(yy==1)&(base<.75)&(alternatives>=base+.035)]=ADOPT
    actions[(yy==0)&(base>=.15)]=SUPPRESS
    result=dict(p=pp,context=ctx[times,pitches],
                k=kprobs[times],y=yy,actions=actions,
                true_k=kk,source=source,idx=ix,
                label_coverage=dict(
                    total_positive=int(y.sum()),
                    positive_eligible=int(np.sum((y==1)&eligible)),
                    sampled_positives=int((yy==1).sum())))
    return result

def raw_risk_stats(data):
    # Count exactly where the candidate would correct or regress the binary
    # onset at the fixed low-attack threshold, using training labels only.
    stats=np.zeros((H,7,3),np.float64)
    yy=data["y"].astype(bool)
    base=data["p"][:,0]>=.15
    for ki in range(7):
        use=data["true_k"]==ki
        for h in range(H):
            pred=data["p"][:,h]>=.15
            change=(pred!=base)&use
            stats[h,ki]=[np.count_nonzero(change & (pred==yy) & (base!=yy)),
                         np.count_nonzero(change & (base==yy) & (pred!=yy)),
                         np.count_nonzero(change)]
    return stats

def fit_audit_rows(data,others,alpha=16.):
    stats=np.sum([raw_risk_stats(d) for d in others],axis=0)
    denom=np.zeros(7,np.float64)
    for d in others:
        denom+=np.bincount(d["true_k"],minlength=7)
    profile=(stats/(denom[None,:,None]+alpha)).reshape(1,H,21).astype(np.float32)
    return np.broadcast_to(profile,(len(data["y"]),H,21)).copy()

def rows_to_torch(data,audit,device="cpu"):
    n=len(data["y"])
    ht,hm,km=head_schema(n)
    return tuple(torch.as_tensor(x,device=device) for x in
            (data["p"],audit,ht,hm,km,data["context"],data["k"],data["p"][:,0]))

def fit_model(fit,checkpoint,out,epochs=3):
    torch.manual_seed(733)
    model=OneTargetClassConditionalSelector()
    opt=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.003)
    # Leave an entire FIT composition out of each risk profile.
    comp=sorted({d["source"] for d in fit})
    packed=[]
    for row in fit:
        other=[d for d in fit if d["source"]!=row["source"]]
        if not other:
            raise AssertionError("Requires >1 independent FIT composition")
        audited=fit_audit_rows(row,other)
        packed.append((rows_to_torch(row,audited),
                       torch.as_tensor(row["actions"]).long(),
                       torch.from_numpy(np.repeat(audited[:,:,:1],2,axis=2)).float()))
    sizes=[len(d["y"]) for d in fit]
    # Risk labels: rate of correcting/regressing observed training heads
    # inferred from OTHER compositions ONLY; no same-row correctness leak.
    train_history=[]
    for epoch in range(epochs):
        rng=np.random.default_rng(733+epoch)
        order=rng.permutation(len(fit))
        total=0.;n_steps=0
        model.train()
        for idx in order:
            x,action,risks=packed[int(idx)]
            for s in range(0,len(action),256):
                opt.zero_grad()
                inp=tuple(t[s:s+256] for t in x)
                outp=model(*inp)
                loss=target_loss(outp,action[s:s+256],risks[s:s+256],inp[3])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
                opt.step()
                total+=float(loss.detach());n_steps+=1
        train_history.append(dict(epoch=epoch+1,loss=total/max(1,n_steps)))
        print("S733 FIT EPOCH",json.dumps(train_history[-1]),flush=True)
    torch.save(dict(model=model.state_dict(),base_sha256=checkpoint,
                    target="onset",epochs=epochs),out/"s733_onset.pt")
    return model,train_history

def predict_onset(model,candidates,ctx,eligible,kprobs,audit,base_post,mode):
    t,n=eligible.shape
    inds=np.flatnonzero(eligible.ravel())
    proposed=base_post.copy()
    counts=Counter()
    if not len(inds):return proposed,dict(counts)
    model.eval()
    with torch.inference_mode():
        for st in range(0,len(inds),1024):
            batch=inds[st:st+1024]
            frame,pitch=np.divmod(batch,n)
            m=len(batch); htype,hmask,kmask=head_schema(m)
            args=tuple(torch.from_numpy(np.asarray(x,dtype=np.float32))
                   if j not in (3,4) else torch.from_numpy(x) for j,x in
                   enumerate((candidates[frame,pitch],
                     np.broadcast_to(audit,(m,H,21)).copy(),
                     htype,hmask,kmask,ctx[frame,pitch],kprobs[frame],
                     candidates[frame,pitch,0])))
            pred=model(*args)
            pr=torch.softmax(pred["action_logits"],-1).numpy()
            offer=pred["offered_probability"].numpy()
            for a,(ti,pi) in enumerate(zip(frame,pitch)):
                if pr[a,ADOPT]>=.55 and pr[a,ADOPT]>pr[a,0]:
                    new=float(max(base_post[ti,pi,0],offer[a]))
                    if new>base_post[ti,pi,0]+.001:
                        proposed[ti,pi,0]=new
                        counts["adopted"]+=1
                elif mode=="onset_guarded_both" and pr[a,SUPPRESS]>=.85:
                    if base_post[ti,pi,0]>.15:
                        proposed[ti,pi,0]=0.
                        counts["suppressed"]+=1
    if not np.array_equal(proposed[:,:,1:],base_post[:,:,1:]):
        raise AssertionError("Sustain or release altered by S733")
    return proposed,dict(counts)

def evaluate(specs,phase,archive,root,heads,selection,offset,network,policy,
             release,model,train_profiles,choices):
    gathered={REFERENCE:[]}
    paired={name:dict(lost_correct_offsets=0,lost_long=0) for name in choices}
    alterations=Counter()
    for name in choices:gathered[name]=[]
    by_texture={name:{key:[] for key in gathered} for name in ("comp","solo")}
    tracks=[]
    for i,spec in enumerate(specs):
        item,post,k,original,candidates,ctx,eligible=baseline_input(spec,archive,
                    root,heads,selection,offset,network,policy,release)
        emitted={REFERENCE:original}
        for arm in choices:
            modified,counts=predict_onset(model,candidates,ctx,eligible,k,
                                train_profiles,post,arm)
            alterations.update({arm+"_"+key:v for key,v in counts.items()})
            emitted[arm]=emit(item,modified,k,policy,offset,release)
        truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
        score={arm:metric(events,truth,item["ends"]) for arm,events in emitted.items()}
        old=score[REFERENCE]
        for arm in choices:
            candidate=score[arm]
            paired[arm]["lost_correct_offsets"]+=len(
                set(old["correct_offset_truth_ids"])-set(candidate["correct_offset_truth_ids"]))
            paired[arm]["lost_long"]+=len(
                set(old["long_onset_truth_ids"])-set(candidate["long_onset_truth_ids"]))
        texture="comp" if spec["name"].endswith("_comp") else "solo"
        by_texture[texture][REFERENCE].append(score[REFERENCE])
        for arm in choices:by_texture[texture][arm].append(score[arm])
        for arm,row in score.items():gathered[arm].append(row)
        tracks.append(dict(name=spec["name"],scores={
            name:{m:float(row[m]) for m in SCORES} for name,row in score.items()},
            events={name:dict(tp=row["tp"],fp=row["fp"],fn=row["fn"])
                    for name,row in score.items()}))
        print("S733",phase,i+1,len(specs),spec["name"],flush=True)
    aggregates={key:aggregate(rows) for key,rows in gathered.items()}
    return dict(scores={a:{m:float(row[m]) for m in SCORES}
                        for a,row in aggregates.items()},
        totals={a:{m:int(row[m]) for m in ("tp","fp","fn","frames","k_correct",
                            "offset_correct")} for a,row in aggregates.items()},
        by_texture={v:{a:{m:float(aggregate(rows)[m]) for m in SCORES}
                       for a,rows in choices_row.items()}
                    for v,choices_row in by_texture.items()},
        lost=paired,alterations=dict(alterations),tracks=tracks)

def qualifies(scores,loss,original):
    return (all(scores[m]>=original[m]-1.e-12 for m in SCORES)
            and scores.get("onset_f1",0)>=original["onset_f1"]
            and loss["lost_correct_offsets"]==0 and loss["lost_long"]==0)

def run(args):
    if args.out.exists():raise FileExistsError("S733 immutable out already exists")
    args.out.mkdir(parents=True)
    original=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=original["checkpoint_sha256"]:raise AssertionError("S708 SHA mismatch")
    SingleTargetContract("onset",sha,True).validate()
    policy=GuidedPolicy(**original["frozen_candidate"]["choice"]["policy"])
    release=original["frozen_candidate"]["choice"]["release"]
    if release!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("S707 release mismatch")
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    torch.set_num_threads(2)
    report=dict(schema="S733_ONSET_ONLY_LEARNED_CLASS_CONDITIONAL_RISK_SELECTOR",
          promoted=False,LIVE_changed=False,TEST_opened=False,
          base_sha256=sha,train_only="FIT80",held_dev_previously_exposed=True,
          sequence="onset only; Sustain/Release/K/S707 frozen",
          complete=False)
    save(args.out/"report.json",report)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            src=protocol["guitar_archives"][key]
            download(src["url"],root/"data"/name,src["size"],src["sha256"])
        sources=[s for s in protocol["sources"] if
                 s["corpus"]=="GuitarSet" and s["split"]=="train"]
        dev=[s for s in protocol["sources"] if
             s["corpus"]=="GuitarSet" and s["split"]=="dev"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio,\
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as annotation:
            expanded=expanded_train_specs(sources,audio,annotation)
            split={name:[s for s in expanded if s["composition"] in comps]
                   for name,comps in (("FIT",FIT_COMPS),("CAL",CAL_COMPS),
                                      ("HELD",HELD_COMPS))}
            if tuple(len(split[x]) for x in ("FIT","CAL","HELD"))!=(80,32,32):
                raise AssertionError("Unexpected FIT/CAL/HELD split")
            fit=[];proof=[]
            for i,spec in enumerate(split["FIT"]):
                item,post,k,original,candidates,ctx,eligible=baseline_input(
                    spec,audio,root,heads,selection,offset,network,policy,release)
                true=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                y,ktrue=targets(item["decision_ends"],true)
                row=collected_examples(candidates,ctx,eligible,y[:,:,0],
                              ktrue,k,spec["composition"],seed=733+i)
                fit.append(row)
                proof.append(dict(name=spec["name"],audio_sha256=item["sha256"],
                                  sample_rows=len(row["y"]),**row["label_coverage"]))
                print("S733 FIT ONSET",i+1,len(split["FIT"]),spec["name"],flush=True)
            report["FIT"]=dict(sources=proof,trained_rows=sum(len(r["y"]) for r in fit),
                               positive_labels=sum(int(r["y"].sum()) for r in fit),
                               direct_coverage=float(sum(x["positive_eligible"] for x in proof)/
                                   max(1,sum(x["total_positive"] for x in proof))))
            save(args.out/"report.json",report)
            model,history=fit_model(fit,sha,args.out,epochs=args.epochs)
            report["FIT"]["training_loss"]=history
            fit_profiles=np.sum([raw_risk_stats(row) for row in fit],axis=0)
            denominators=np.sum([np.bincount(row["true_k"],minlength=7)
                                 for row in fit],axis=0)
            audit=(fit_profiles/(denominators[None,:,None]+16)).reshape(H,21).astype(np.float32)
            score=evaluate(split["CAL"],"CAL",audio,root,heads,selection,
                           offset,network,policy,release,model,audit,ARMS)
            for met in SCORES:
                if abs(score["scores"][REFERENCE][met]-
                       original["frozen_candidate"]["CAL"][met])>1e-8:
                    raise AssertionError("S708 CAL replay mismatch "+met)
            valid=[REFERENCE]+[arm for arm in ARMS if qualifies(
                score["scores"][arm],score["lost"][arm],
                score["scores"][REFERENCE])]
            selected=max(valid,key=lambda name:(
                min(score["scores"][name][m] for m in SCORES),
                sum(score["scores"][name][m] for m in SCORES)))
            report["CAL"]=dict(**score,eligible=valid,selected=selected)
            save(args.out/"report.json",report)
            print("S733 CAL FROZEN",json.dumps(dict(
                 selected=selected,eligible=valid,scores=score["scores"],
                 altered=score["alterations"])),flush=True)
            for phase,specs in (("HELD",split["HELD"]),("DEV",dev)):
                pick=[] if selected==REFERENCE else [selected]
                row=evaluate(specs,phase,audio,root,heads,selection,offset,
                         network,policy,release,model,audit,pick)
                for met in SCORES:
                    if abs(row["scores"][REFERENCE][met]-
                           original[phase]["candidate"][met])>1e-8:
                        raise AssertionError("S708 held/dev replay mismatch "+phase+met)
                report[phase]=row
                save(args.out/"report.json",report)
    report["complete"]=True
    report["eligible_for_live"]=False
    save(args.out/"report.json",report)
    print("S733 FINAL",json.dumps(dict(selected=report["CAL"]["selected"],
          eligible=report["CAL"]["eligible"],promoted=False,
          cal=report["CAL"]["scores"])),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s708-dir",type=Path,required=True)
    parser.add_argument("--offset-checkpoint",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=3)
    with threadpool_limits(limits=2):run(parser.parse_args())
