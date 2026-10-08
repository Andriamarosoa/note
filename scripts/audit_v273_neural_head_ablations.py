"""Same-checkpoint neural ablations: the 14-head K-conditional Exact-K router.

Unlike diagnostic equal-weight subset votes, this experiment retrains the
IDENTICAL frozen class-conditional architecture on the same outer splits,
then changes head availability during inference *without retraining weights*.
It records model-output changes, restored correct predictions, destroyed
correct predictions, 7-way K breakdowns, and 4x64 class-transition subset
interventions. No true K is used to decide any mask or subset.

H0 is the sole structural fallback for K0/K1; removing it would make the
trained model undefined for those outputs. Thus of each 127 theoretical
seven-head subsets, only the 64 containing H0 are faithfully ablatable.
The other 63 are excluded, not silently replaced by a fake fallback.

Ablation changes activations and is causal for THIS fitted network's
outputs, not evidence of deployment gains or a trained 127-selector.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from scripts.evaluate_v273_dynamic_heads import EXPERTS, oof_experts, outer_experts
from scripts.evaluate_v273_neural_history_mix import (
    H, HEADS, KEEP, build_candidate_logits, cross_audits, full_train_audits,
    inputs, fit_and_predict,
)
from scripts.summarize_v273_harmonic_global import load_cohort, load_features, get_matrix
from scripts.yourmt3_exactk_common import FOLDS, paired, metrics, require


ATTACKS=((2,3,6),(3,2,7),(3,4,8),(4,3,9))
GROUPS={
    "all_spectral_features_H1_H5":(1,2,3,4,5),
    "all_correction_actions_C23_C43":(6,7,8,9),
    "all_keep_fixes_Fkeep2_Fkeepany":(10,11,12,13),
    "all_except_baseline_H0":tuple(range(1,H)),
    "source_H2_and_H5":(2,5),
    "harmonic_H3_H4":(3,4),
}
INTERVENTIONS={
    **{f"drop_{HEADS[h]}":(h,) for h in range(1,H)},
    **{f"drop_group_{name}":indices for name,indices in GROUPS.items()},
}
# H0 stays on by architectural necessity; others may be turned off.
assert H==14 and 0 not in {h for v in INTERVENTIONS.values() for h in v}
FULL=59309
ELIGIBLE=7493


def masking(batch, keep):
    """Drop absent heads AND class eligibility, retaining H0 as fallback."""
    keep=set(keep)
    require(0 in keep,"H0 cannot be removed without redefining K0/K1 head contract")
    sample={k:np.asarray(v).copy() for k,v in batch.items()}
    availability=sample["head_mask"].copy()
    for h in range(H):
        if h not in keep:availability[:,h]=False
    support=sample["head_k_mask"] & availability[:,:,None]
    require(np.all(availability[:,0]) and np.all(support[:,0,:]),"H0 fallback lost")
    require(np.all(np.any(support,axis=1)),"a class lost all heads")
    sample["head_mask"]=availability
    sample["head_k_mask"]=support
    return sample


def mask_drop(batch,drop):
    return masking(batch,set(range(H))-set(drop))


def observe(model,batch,base):
    result=model(batch,training=False)
    logits=result["action_logits"].numpy()
    require(logits.shape==(len(base),8),"bad neural decision shape")
    action=np.argmax(logits,axis=-1)
    return np.where(action==KEEP,base,action).astype(int)


def per_class(y,b,p,mask=None):
    if mask is None:mask=np.ones(len(y),bool)
    changes=(p!=b) & mask
    fix=changes & (p==y)
    regression=changes & (b==y)
    summary={}
    for k in range(7):
        m=mask & (y==k)
        summary[str(k)]=dict(
            n=int(m.sum()),
            actions=int(np.sum(changes&m)),
            corrections=int(np.sum(fix&m)),
            regressions=int(np.sum(regression&m)),
            neutral=int(np.sum(changes&m & ~(fix|regression))),
            net=int(np.sum(fix&m)-np.sum(regression&m)),
        )
    return summary


def summarize(y,b,original,new):
    y=np.asarray(y);b=np.asarray(b);original=np.asarray(original);new=np.asarray(new)
    require(y.shape==b.shape==original.shape==new.shape,"mismatched audit")
    base_correct=y==b
    changed=new!=b
    cor=changed &(y==new)
    reg=changed &(y==b)
    origgood=y==original
    newgood=y==new
    original_k2to1=(y==2)&(b==2)&(original==1)
    return {
        "events":int(len(y)),
        "baseline_actions":int(np.sum(original!=b)),
        "ablation_actions":int(np.sum(changed)),
        "baseline_correct_total":int(np.sum(base_correct)),
        "normal_neural_correct":int(np.sum(origgood)),
        "ablation_neural_correct":int(np.sum(newgood)),
        "ablation_corrections_vs_freeze":int(np.sum(cor)),
        "ablation_regressions_vs_freeze":int(np.sum(reg)),
        "ablation_neutral_actions_vs_freeze":int(np.sum(changed &~(cor|reg))),
        "net_vs_freeze":int(np.sum(cor)-np.sum(reg)),
        "rescued_neural_errors":int(np.sum(~origgood & newgood)),
        "newly_harmed_neural_correct":int(np.sum(origgood & ~newgood)),
        "net_vs_unablated_neural":int(np.sum(newgood)-np.sum(origgood)),
        "regressions_K2_to_K1_original":int(np.sum(original_k2to1)),
        "regressions_K2_to_K1_after_ablation":int(np.sum((y==2)&(b==2)&(new==1))),
        "K2_to_K1_regressions_rescued_to_correct_K2":int(np.sum(original_k2to1 &(new==2))),
        "K2_to_K1_regressions_newly_created":int(np.sum((y==2)&(b==2)&(original!=1)&(new==1))),
        "neural_prediction_changes":int(np.sum(original!=new)),
        "by_true_K":per_class(y,b,new),
    }


def agg(items):
    """Merge additive audit counters; no overlap in outer held-out folds."""
    if not items:raise ValueError("no fold rows")
    out={}
    for k,v in items[0].items():
        if k=="by_true_K":
            out[k]={str(t):{key:int(sum(z[k][str(t)][key] for z in items))
                             for key in v[str(t)]} for t in range(7)}
        elif isinstance(v,int):
            out[k]=int(sum(z[k] for z in items))
    return out


def process_fold(model,val,y,b,indices,fold_id):
    require(len(b)==len(y)==len(indices),"row alignment")
    original=observe(model,val,b)
    single={}
    allpred={}
    for label,drop in INTERVENTIONS.items():
        active_drop=np.any(val["head_mask"][:,list(drop)],axis=1)
        if not active_drop:
            candidate=original.copy()
        else:
            candidate=observe(model,mask_drop(val,drop),b)
            # On events where a head is inactive, intervening on it has no
            # effect on class eligibility; the unchanged global model
            # calculation should agree for singletons.
            if len(drop)==1:
                unaffected=~val["head_mask"][:,drop[0]]
                require(np.array_equal(candidate[unaffected],original[unaffected]),
                        "inactive-head ablation unexpectedly changed predictions")
        z=summarize(y,b,original,candidate)
        z.update(fold=int(fold_id),label=label,
                 masked_heads="+".join(HEADS[h] for h in drop),
                 masked_head_count=len(drop),
                 original_model_actions=int(np.sum(original!=b)),
                 active_row_count=int(np.sum(active_drop)),
                 type="leave_one_out" if len(drop)==1 else "group_ablation")
        single[label]=z
        allpred[label]=candidate

    grouped={}
    # Exactly 64 faithful subsets of each 7-head directional bank (H0
    # compulsory). The other 63 theoretical H0-free sets cannot be
    # replayed with this architecture.
    for source,target,correction in ATTACKS:
        rows=(b==source)
        if not np.any(rows):continue
        bank=(0,1,2,3,4,5,correction)
        free=bank[1:]
        row_val={name:v[rows] for name,v in val.items()}
        labels=y[rows];orig=original[rows];rb=b[rows]
        for subset_no in range(64):
            keep=(0,)+tuple(free[i] for i in range(6)
                            if subset_no&(1<<i))
            ablated=observe(model,masking(row_val,keep),rb)
            z=summarize(labels,rb,orig,ablated)
            # True K is used strictly AFTER inference.
            to_target=(ablated==target)&(rb!=target)
            z.update(
                fold=int(fold_id),
                scenario=f"K{source}->K{target}",
                source=source,target=target,subset_no=subset_no,
                included_heads="+".join(HEADS[h] for h in keep),
                included_count=len(keep),
                predicted_target=int(np.sum(to_target)),
                target_corrections=int(np.sum(to_target&(labels==target))),
                target_regressions=int(np.sum(to_target&(labels==source))),
                target_neutral=int(np.sum(to_target&(labels!=target)&(labels!=source))),
                target_net=int(np.sum(to_target&(labels==target))-
                               np.sum(to_target&(labels==source))),
            )
            grouped[f"{source}-{target}-{subset_no}"]=z
    print(json.dumps({"fold":int(fold_id),
                      "events":int(len(y)),
                      "normal_neural_K2toK1":int(np.sum((y==2)&(b==2)&(original==1))),
                      "single_ablation_count":len(single),
                      "conditional_subset_experiments":len(grouped)}),flush=True)
    return original,single,grouped,allpred


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cohort",type=Path,required=True)
    p.add_argument("--features",type=Path,required=True)
    p.add_argument("--reference-predictions",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    require(not a.output.exists(),"refuse overwrite")
    with np.load(a.reference_predictions,allow_pickle=False) as z:
        reference={name:z[name] for name in z.files}
    y,b,ids,fold,member,starts=load_cohort(a.cohort)
    indices,rows,names,_=load_features(a.features,y,b,ids,fold,member,starts)
    require(len(y)==FULL and len(indices)==ELIGIBLE,"frozen cohort drift")
    require(np.array_equal(reference["true_k"],y) and
            np.array_equal(reference["baseline"],b) and
            np.array_equal(reference["global_index"],ids) and
            np.array_equal(reference["fold"],fold),"reference drift")
    require(np.array_equal(reference["eligible_indices"],ids[indices]),
            "OOF reference eligibility drift")
    yc=y[indices];bc=b[indices];fc=fold[indices]
    x={name:get_matrix(rows,names,field,prefixes)
       for name,(field,prefixes) in EXPERTS.items()}
    context=[n for n in names if
             n.startswith(("spectral__","birth__","persistence__","damping__"))][:26]
    raw=np.asarray([[r["features"][name] for name in context] for r in rows],float)
    require(np.isfinite(raw).all(),"bad acoustic context")
    prediction=np.full(len(indices),-1,int)
    ablate_preds={label:np.full(len(indices),-1,int) for label in INTERVENTIONS}
    singleton_fold_rows=[]
    directional_fold_rows=[]
    by_single=defaultdict(list)
    by_subset=defaultdict(list)

    for vf in FOLDS:
        tr=fc!=vf;val=fc==vf
        ptr=oof_experts({name:v[tr] for name,v in x.items()},
                        yc[tr],bc[tr],fc[tr])
        pv=outer_experts({name:{"train":v[tr],"test":v[val]}
                          for name,v in x.items()},yc[tr],bc[val])
        train_logits,train_mask,train_kind,train_support=build_candidate_logits(ptr,bc[tr])
        held_logits,held_mask,held_kind,held_support=build_candidate_logits(pv,bc[val])
        audits_train=cross_audits(yc[tr],bc[tr],fc[tr],train_logits)
        audits_hold=np.broadcast_to(
            full_train_audits(yc[tr],bc[tr],fc[tr],train_logits),
            (int(val.sum()),H,21)).copy()
        std=StandardScaler().fit(raw[tr])
        train=inputs(train_logits,train_mask,train_kind,train_support,
                     audits_train,np.clip(std.transform(raw[tr]),-6,6),bc[tr])
        held=inputs(held_logits,held_mask,held_kind,held_support,
                    audits_hold,np.clip(std.transform(raw[val]),-6,6),bc[val])
        action,weights_class,weights_avg,risk,model=fit_and_predict(
            train,yc[tr],bc[tr],held,return_model=True)
        original=np.where(action==KEEP,bc[val],action)
        replay,singles,subsets,ablate=process_fold(
            model,held,yc[val],bc[val],ids[indices[val]],vf)
        require(np.array_equal(original,replay),"normal inference drift within run")
        prediction[val]=original
        for label,row in singles.items():
            singleton_fold_rows.append(row)
            by_single[label].append(row)
            ablate_preds[label][val]=ablate[label]
        for label,row in subsets.items():
            directional_fold_rows.append(row)
            by_subset[label].append(row)
    require(np.all(prediction>=0) and all(np.all(x>=0) for x in ablate_preds.values()),
            "incomplete fold coverage")
    # Reproducibility is explicitly checked, not assumed. Any mismatch means
    # these are ablations of a newly-trained replica, not exactly the model
    # that generated run 37747861591.
    replay_difference=int(np.sum(
        prediction!=reference["corrected"][indices]))
    report={
        "experiment":"v273_real_neural_head_and_subsets_ablation",
        "status":"completed",
        "native_rows":len(y),
        "eligible_rows":len(indices),
        "train_folds":list(FOLDS),
        "baseline_unchanged":True,
        "normal_neural_replay_disagreements_vs_original_run":replay_difference,
        "exact_replica":replay_difference==0,
        "original_conditional_run":37747861591,
        "controls":{
            "H0_mandatory":True,
            "available_directional_structural_subsets":127,
            "faithful_subsets_including_H0":64,
            "unavailable_H0_free_subsets":63,
            "per_class_source_target_scenarios":[f"K{a}->K{b}" for a,b,c in ATTACKS],
            "same_checkpoint_per_mask":True,
            "masks_never_see_true_K":True,
            "original_train_OOF_only":True,
        },
        "original_replayed_full":metrics(y,np.where(
            np.isin(ids,ids[indices]),b,b)), # populated below
        "single_and_groups":{},
        "directional_subsets":{},
        "limitations":[
            "Ablation explains sensitivity of the retrained network to individual heads and groups, not independent generalization.",
            "The neural network was retrained using identical code and folds because previous workflow did not save model weights. Exact output match to prior run is checked.",
            "Removing H0 would violate K0/K1 output support: 63 H0-free theoretical subsets per seven-head bank cannot be validly ablated.",
            "The 64 subsets are interventions on trained attention, NOT an independently trained subset-selector.",
            "Subset rankings based on heldout labels are descriptive and must not be deployed without fresh independent evaluation.",
            "Outer fold3/player05 excluded; audio context up to +160ms, not causal online.",
            "14 candidate heads include newly-designed correction/KEEP adapters, not all historical fixes.",
        ],
        "automatic_promotion":False,
    }
    full_orig=b.copy();full_orig[indices]=prediction
    report["original_replayed_full"]=metrics(y,full_orig)
    report["original_replayed_paired"]=paired(y,b,full_orig)
    for label,foldrows in by_single.items():
        require(len(foldrows)==len(FOLDS),"missing singleton outer fold")
        report["single_and_groups"][label]=agg(foldrows)
    for label,foldrows in by_subset.items():
        require(len(foldrows)==len(FOLDS),"missing subset outer fold")
        summary=agg(foldrows)
        summary.update(
            scenario=foldrows[0]["scenario"],source=foldrows[0]["source"],
            target=foldrows[0]["target"],subset_no=foldrows[0]["subset_no"],
            included_heads=foldrows[0]["included_heads"],
            included_count=foldrows[0]["included_count"]
        )
        for col in ("predicted_target","target_corrections",
                    "target_regressions","target_neutral","target_net"):
            summary[col]=int(sum(row[col] for row in foldrows))
        report["directional_subsets"][label]=summary
    require(len(report["directional_subsets"])==4*64,"missing subset ablations")
    a.output.mkdir(parents=True)
    (a.output/"rootcause-ablation.json").write_text(json.dumps(
        report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    csv_fields=[
        "name","kind","fold","scenario","subset_no","heads","events",
        "ablation_actions","ablation_corrections_vs_freeze",
        "ablation_regressions_vs_freeze","net_vs_freeze",
        "normal_neural_correct","ablation_neural_correct",
        "rescued_neural_errors","newly_harmed_neural_correct",
        "net_vs_unablated_neural",
        "regressions_K2_to_K1_original",
        "regressions_K2_to_K1_after_ablation",
        "K2_to_K1_regressions_rescued_to_correct_K2",
        "K2_to_K1_regressions_newly_created",
        "predicted_target","target_corrections",
        "target_regressions","target_net",
    ]
    with (a.output/"all-ablation-folds.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=csv_fields);writer.writeheader()
        for r in singleton_fold_rows:
            writer.writerow({
                "name":r["label"],"kind":r["type"],"fold":r["fold"],
                "heads":r["masked_heads"],**{k:r[k] for k in csv_fields if k in r},
            })
        for r in directional_fold_rows:
            writer.writerow({
                "name":f"{r['scenario']}:S{r['subset_no']:02}",
                "kind":"neural_subset_intervention",
                "fold":r["fold"],"heads":r["included_heads"],
                **{k:r[k] for k in csv_fields if k in r}
            })
    np.savez_compressed(a.output/"decisions-by-head.npz",
        eligible_global_index=ids[indices],true_k=yc,baseline_k=bc,
        fold=fc,normal_neural=prediction,
        **{label:pred for label,pred in ablate_preds.items()})
    lines=[
        "# Ablation réelle du réseau neuronal (mêmes poids par fold)",
        "",
        f"Réplication du run initial : {replay_difference} prédictions différentes "
        f"sur {ELIGIBLE} événements admissibles.",
        f"Exact-K global réentraîné: {report['original_replayed_full']['exact']*100:.4f} % ; "
        f"poly: {report['original_replayed_full']['poly']['exact']*100:.4f} %.",
        "",
        "## Les têtes dont le retrait modifie le plus la performance",
        "",
        "| Retrait | Regressions K2->K1 retrouvées | "
        "Erreurs neurales corrigées | Nouvelles erreurs | "
        "Différence bonne prédictions nettes |",
        "|---|---:|---:|---:|---:|",
    ]
    for key,r in sorted(report["single_and_groups"].items(),
                        key=lambda row: -row[1]["K2_to_K1_regressions_rescued_to_correct_K2"]):
        lines.append(f"| {key} | "
                     f"{r['K2_to_K1_regressions_rescued_to_correct_K2']} | "
                     f"{r['rescued_neural_errors']} | "
                     f"{r['newly_harmed_neural_correct']} | "
                     f"{r['net_vs_unablated_neural']:+d} |")
    lines+=["","## Les 64 masquages exacts, par transition","",
            "Sur 127 sous-ensembles théoriques de 7 têtes, **64 incluent H0** "
            "et peuvent être évalués fidèlement sur le modèle actuel ; "
            "**63 excluent H0** et rendraient impossible la tête de repli "
            "K0/K1, donc ils ne sont pas évalués. Aucun faux réseau de repli "
            "n'a été ajouté.",
            "",
            "| Transition | Meilleur gain net de retrait (descriptif) | "
            "Pire perte nette |",
            "|---|---:|---:|"]
    for a_,b_,_ in ATTACKS:
        rows=[r for r in report["directional_subsets"].values()
              if r["source"]==a_ and r["target"]==b_]
        require(len(rows)==64,"missing subset family")
        best=max(rows,key=lambda r:r["net_vs_unablated_neural"])
        worst=min(rows,key=lambda r:r["net_vs_unablated_neural"])
        lines.append(f"| K{a_}->K{b_} | "
                     f"S{best['subset_no']:02} ({best['included_heads']}): "
                     f"{best['net_vs_unablated_neural']:+d} | "
                     f"S{worst['subset_no']:02} ({worst['included_heads']}): "
                     f"{worst['net_vs_unablated_neural']:+d} |")
    lines+=["","## Interprétation interdite","",
            "Le meilleur sous-ensemble a été identifié **après lecture des "
            "vraies classes du fold**. Ce classement ne valide pas un "
            "déploiement et ne mesure pas la contribution de combinaisons "
            "choisies par un routeur entraîné sur leurs effets.",
            "",
            "Les actions K0/K1 doivent toujours être auditées séparément "
            "des sous-ensembles K2/K3/K4. Le décodeur libre peut choisir "
            "une classe avec très peu d'évidence d'expert.",
            "",
            "## Limites"]+["- "+v for v in report["limitations"]]
    (a.output/"verdict.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

if __name__=="__main__":
    main()
