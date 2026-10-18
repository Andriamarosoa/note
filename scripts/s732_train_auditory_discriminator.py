"""S732: FIT-learn whether a new source is needed beyond existing harmonics.

CAL/HELD/DEV remain isolated from learned model fitting. Evaluation uses the
original signed S708 and S707, same true emitted stream metrics, no oracle
labels in acoustic inference. Two identical HGB budgets: full auditory scene
vs original-only information. CAL predeclared candidate onset likelihood
gain/penalty, six nonregression metrics plus zero formerly correct events lost.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import zipfile
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from causal_note.strict_live import SR
from causal_note.s705_temporal_heads import predict_track
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s731_auditory_scene import SceneEvidence
from causal_note.s732_source_discriminator import (
    observation_matrix,fit_discriminator,sampled_fit_rows,
    predict_track_causal,apply_evidence)
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s706_train_sharp_temporal_loops import record,load_model
from scripts.s705_train_temporal_loops import setup
from scripts.s704_train_joint_loops import targets
from scripts.s704_joint_metrics import SCORES,metric,aggregate,target_rank
from scripts.s731_auditory_scene_calibration import emit
from scripts.s728_real_octave_overlap_audit import overlap_results

REFERENCE="S708_frozen"
GAIN=(.20,.45,.75)
PENALTY=(0.,.08)
MODES=("control","scene")
GRID={f"{name}_gain_{str(g).replace('.','p')}_pen_{str(p).replace('.','p')}":
     (name,g,p) for name in MODES for g in GAIN for p in PENALTY}


def audit_guards(original,candidate,paired):
    return (all(candidate[q]>=original[q]-1e-12 for q in SCORES)
            and candidate["tp"]>=original["tp"]
            and paired["lost_correct_offsets"]==0 and paired["lost_long"]==0)


def source_input(spec,archive,root,heads,selection,offset,model,policy,release):
    item=record(spec,archive,heads,selection,offset)
    x=item["x"];item["native_features"]=x[:,:,:126].copy()
    p,k=predict_track(model,x)
    post=policy.blend*p+(1-policy.blend)*item["native_post"]
    baseline=emit(item,post,k,policy,offset,release)
    evidence=observation_matrix(x,post,k,baseline,item["decision_ends"])
    return item,post,k,baseline,evidence


def fit_sources(specs,archive,root,heads,selection,offset,model,
                policy,release,out):
    rows=[];target=[];weights=[];proof=[]
    for i,spec in enumerate(specs):
        item,post,k,baseline,features=source_input(spec,archive,root,heads,
                    selection,offset,model,policy,release)
        # FIT: original causal scene / note stream generated first.
        # Then annotation may supervise the new independent-source classifier.
        true=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
        y,_=targets(item["decision_ends"],true)
        samples,yy,w=sampled_fit_rows(features,y[:,:,0],
                    seed=732+i,max_rows=4500)
        if samples.shape[0]:
            rows.append(samples);target.append(yy);weights.append(w)
        proof.append(dict(name=spec["name"],audio_sha256=item["sha256"],
                          supervised_onsets=int(y[:,:,0].sum()),
                          sampled_rows=len(samples),sampled_true_onsets=int(yy.sum())))
        print("S732 FIT SCENE",i+1,len(specs),spec["name"],flush=True)
    x=np.concatenate(rows);y=np.concatenate(target);w=np.concatenate(weights)
    if not (0<y.mean()<.40 and y.sum()>100):
        raise AssertionError("S732 FIT labels not sufficiently representative")
    result={}
    for name,has_scene in (("control",False),("scene",True)):
        fitted=fit_discriminator(x,y,w,with_scene=has_scene,seed=732)
        file=out/f"{name}.joblib"
        joblib.dump(fitted,file,compress=3)
        result[name]=dict(path=file.name,sha256=digest(file.read_bytes()),
                          fitted_on_FIT_only=True,uses_real_scene=has_scene,
                          n_observations=len(x),n_true_labeled_onsets=int(y.sum()))
        print("S732 FIT MODEL",name,json.dumps(result[name]),flush=True)
    return result,proof


def score_sources(specs,phase,archive,root,heads,selection,offset,network,
                  policy,release,models,choices):
    cumulative={REFERENCE:[]}
    guardloss={name:dict(lost_correct_offsets=0,lost_long=0)
                  for name in choices}
    overlap={name:{str(d):Counter() for d in (12,19)}
                  for name in [REFERENCE,*choices]}
    for name in choices:cumulative[name]=[]
    tracks=[]
    for i,spec in enumerate(specs):
        item,post,k,baseline,features=source_input(spec,archive,root,
                         heads,selection,offset,network,policy,release)
        q={name:predict_track_causal(m,features,with_scene=(name=="scene"))
               for name,m in models.items() if name in {a for a,b,c in choices.values()}}
        emitted={REFERENCE:baseline}
        for key,(name,g,p) in choices.items():
            measured=q[name]
            proposals=np.empty_like(post)
            for t in range(len(post)):
                f=features[t]
                state=SceneEvidence(novelty=f[:,4],
                   explained=f[:,5],novelty_shared=f[:,6],
                   novelty_unshared=f[:,7],old_ownership=f[:,3],
                   supports=(f[:,8]*8).astype(np.int16),count_room=float(f[0,9]))
                proposals[t]=apply_evidence(post[t],k[t],state,measured[t],
                      gain=g,penalty=p,use_novelty=name=="scene")
            emitted[key]=emit(item,proposals,k,policy,offset,release)
        # Genuine unmodified GuitarSet annotations read after all decisions.
        truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
        result={name:metric(events,truth,item["ends"])
                for name,events in emitted.items()}
        baseline_row=result[REFERENCE]
        oldoff=set(baseline_row["correct_offset_truth_ids"])
        oldlong=set(baseline_row["long_onset_truth_ids"])
        for name,r in result.items():
            cumulative[name].append(r)
            for d in (12,19):
                o=overlap_results(emitted[name],truth,d)
                for met in ("observed_real_targets","correct_real_onsets",
                            "missed_real_onsets","spurious_harmonic_child_onsets"):
                    overlap[name][str(d)][met]+=o[met]
            if name!=REFERENCE:
                guardloss[name]["lost_correct_offsets"]+=len(
                    oldoff-set(r["correct_offset_truth_ids"]))
                guardloss[name]["lost_long"]+=len(
                    oldlong-set(r["long_onset_truth_ids"]))
        tracks.append(dict(name=spec["name"],sha256=item["sha256"],
            scores={key:{s:r[s] for s in SCORES} for key,r in result.items()}))
        print("S732",phase,"SOURCE EVIDENCE",i+1,len(specs),spec["name"],flush=True)
    return ({key:aggregate(v) for key,v in cumulative.items()},
             guardloss,{name:{k:dict(v) for k,v in rows.items()}
                     for name,rows in overlap.items()},tracks)


def run(args):
    if args.out.exists():raise FileExistsError("S732 refuses mutable experiment")
    args.out.mkdir(parents=True)
    original=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if original["checkpoint_sha256"]!=sha:
        raise ValueError("S732 needs signed S708 original")
    policy=GuidedPolicy(**original["frozen_candidate"]["choice"]["policy"])
    release=original["frozen_candidate"]["choice"]["release"]
    assert release==dict(gate=.8,persistence=1,grace=24)
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    report=dict(schema="S732_FIT_LEARNED_AUDITORY_INDEPENDENT_SOURCE_EXPLANATION",
       promoted=False,LIVE_changed=False,TEST_opened=False,
       independent_guitar_sources_only_FIT_training=True,
       CAL_only_policy_selection=True,HELD_DEV_previously_exposed=True,
       original_S708_sha256=sha,original_ExactK_S707_frozen=True,
       scene="independent psychoacoustic source discrimination, explaining away old-source harmonics",
       arms="no_scene_control vs old_plus_new_scene all pitches",
       predeclared_policy_grid={a:dict(mode=b,gain=g,penalty=p)
                      for a,(b,g,p) in GRID.items()},
       policy=asdict(policy),release=release)
    def save():
        (args.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    save()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);(root/"data").mkdir()
        proto,heads,selector,offset=setup(root,args.offset_checkpoint)
        for key,file in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            src=proto["guitar_archives"][key]
            download(src["url"],root/"data"/file,src["size"],src["sha256"])
        sources=[s for s in proto["sources"] if
                 s["corpus"]=="GuitarSet" and s["split"]=="train"]
        dev=[s for s in proto["sources"] if
             s["corpus"]=="GuitarSet" and s["split"]=="dev"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive,zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            whole=expanded_train_specs(sources,archive,ann)
            fit=[s for s in whole if s["composition"] in original["fit_compositions"]]
            cal=[s for s in whole if s["composition"] in original["cal_compositions"]]
            held=[s for s in whole if s["composition"] in original["held_compositions"]]
            assert (len(fit),len(cal),len(held),len(dev))==(80,32,32,12)
            model_info,training_proof=fit_sources(fit,archive,root,heads,
                selector,offset,network,policy,release,args.out)
            models={name:joblib.load(args.out/desc["path"])
                        for name,desc in model_info.items()}
            report["fitted_models"]=model_info
            report["FIT_proof"]=training_proof
            save()
            result,paired,overlap,tracks=score_sources(cal,"CAL",archive,root,
                   heads,selector,offset,network,policy,release,models,GRID)
            old=result[REFERENCE]
            eligible=[REFERENCE]+[name for name in GRID
                           if audit_guards(old,result[name],paired[name])]
            winner=max(eligible,key=lambda name:target_rank(result[name]))
            report["CAL"]=dict(scores=result,paired=paired,overlap=overlap,
                                tracks=tracks,eligible=eligible,selected=winner)
            save()
            print("S732 CAL FROZEN",json.dumps(dict(
                selected=winner,eligible=eligible,
                scores={k:{m:v[m] for m in SCORES} for k,v in result.items()},
                octaves=overlap)),flush=True)
            options={} if winner==REFERENCE else {winner:GRID[winner]}
            for phase,specs in (("HELD",held),("DEV",dev)):
                score,loss,overlap,tracks=score_sources(specs,phase,archive,root,
                     heads,selector,offset,network,policy,release,models,options)
                for m in SCORES:
                    if abs(score[REFERENCE][m]-original[phase]["candidate"][m])>1e-8:
                        raise AssertionError("S732 signed historical replay divergence "+m)
                safe=True if winner==REFERENCE else audit_guards(
                              score[REFERENCE],score[winner],loss[winner])
                report[phase]=dict(baseline={m:score[REFERENCE][m] for m in SCORES},
                    candidate={m:score[winner][m] for m in SCORES},
                    safe=safe,octaves=overlap,tracks=tracks)
                save()
                print("S732",phase,"CHECK",json.dumps(dict(selected=winner,
                                  safe=safe,candidate=report[phase]["candidate"])),flush=True)
    report["shadow_qualifies"]=bool(winner!=REFERENCE and
                           report["HELD"]["safe"] and report["DEV"]["safe"])
    report["promoted"]=False
    save()
    print("S732 FINAL",json.dumps(dict(selected=winner,
           qualifies=report["shadow_qualifies"],promoted=False)),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s708-dir",type=Path,required=True)
    ap.add_argument("--offset-checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    with threadpool_limits(limits=2):run(ap.parse_args())
