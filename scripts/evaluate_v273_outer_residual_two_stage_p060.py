"""Exploratory outer fold-3 evaluation of the residual guard + two veto stages.

Policy frozen before touching outer labels:
  source residual action:
    B_low + robust base K3 + pair/triplet residual LR p(K2)>=0.5
  stage1 veto:
    pre_norm >= internal median AND triplet amp_range <= internal median
  stage2 veto:
    balanced LogisticRegression(C=.1) on [exc_shared_range,onset_contrast]
    p(K3 regression) >= 0.60

Stage1 thresholds and stage2 model are fitted only from the frozen 108 K2-correction
/125 K3-regression OOF internal cases. The outer B_low assignment is independently
reconstructed from all internal folds and required to match the previously frozen
outer B_low mask from run 37316567815.

Fold 3 has been exposed historically; this is exploratory confirmation, not fresh
independent validation. No outer tuning or promotion.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_internal_b_like_boundary_corrector import (
    nested_base,predict,structural_features,NEURONS,SEED
)
from scripts.train_v273_group_gate_ab import build_model,metrics
from scripts.v273_window_experiment import load_bundle,require
from scripts.audit_v273_internal_b_low_harmonic_strata import audio_rows
from scripts.v273_residual_audit import extracted_matrix,fit_classifier
from scripts.audit_v273_stage2_exclusivity_failure import features as stage_features

P_STAGE2=.60

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}

def load_cases(path):
    rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    rows=[r for r in rows if r["group"] in ("K2_corrected","K3_regressed")]
    require(sum(r["group"]=="K2_corrected" for r in rows)==108,"K2 evidence drift")
    require(sum(r["group"]=="K3_regressed" for r in rows)==125,"K3 evidence drift")
    return rows

def metrics_simple(y,p):
    y=np.asarray(y,int);p=np.asarray(p,int);poly=y>=2
    return {
      "rows":int(len(y)),"exact":float(np.mean(y==p)),
      "poly_exact":float(np.mean(y[poly]==p[poly])) if poly.any() else None,
      "under":int(np.sum(p<y)),"over":int(np.sum(p>y)),
      "by_k":{str(k):{"rows":int(np.sum(y==k)),
                       "exact":float(np.mean(p[y==k]==k)) if np.any(y==k) else None}
              for k in range(7)}
    }

def effect(y,before,after):
    y=np.asarray(y);before=np.asarray(before);after=np.asarray(after)
    changed=before!=after
    corr=changed&(before!=y)&(after==y)
    reg=changed&(before==y)&(after!=y)
    return {"changed":int(changed.sum()),"corrections":int(corr.sum()),
            "regressions":int(reg.sum()),"net":int(corr.sum()-reg.sum())}

def action_count(y,mask):
    y=np.asarray(y,int);m=np.asarray(mask,bool)
    c=int(np.sum(m&(y==2)));r=int(np.sum(m&(y==3)))
    return {"actions":int(m.sum()),"corrections":c,"regressions":r,
            "other_k":int(np.sum(m&~np.isin(y,(2,3)))),"net":c-r}

def build_robust_model(uniform_weights,freeze_weights):
    mu=build_model("learned_gate",SEED);mu.load_weights(uniform_weights)
    mg=build_model("learned_gate",SEED);mg.load_weights(freeze_weights)
    bu=nested_base(mu);bg=nested_base(mg)
    ku,bu_bias=[np.asarray(x).copy() for x in bu.get_layer("candidate_hidden1").get_weights()]
    kg,bg_bias=[np.asarray(x).copy() for x in bg.get_layer("candidate_hidden1").get_weights()]
    ids=np.asarray(NEURONS,np.int64)
    kg[:,ids]=ku[:,ids];bg_bias[ids]=bu_bias[ids]
    bg.get_layer("candidate_hidden1").set_weights([kg,bg_bias])
    return mg

def reconstruct_blow(Xf,Xo,yf,Gf):
    fail=np.isin(yf,(2,3,4))&(Gf!=yf)
    coarse=Pipeline([("scale",StandardScaler()),
                     ("km",KMeans(n_clusters=2,n_init=50,random_state=27370))])
    coarse.fit(Xf[fail]);cf=coarse.predict(Xf);co=coarse.predict(Xo)
    stats={}
    for c in (0,1):
        m=fail&(cf==c);under=int(np.sum(Gf[m]<yf[m]))
        stats[c]={"rows":int(m.sum()),"under":under,
                  "under_rate":float(under/max(1,int(m.sum())))}
    a_like=max((0,1),key=lambda c:(stats[c]["under_rate"],stats[c]["rows"]))
    b_like=1-a_like
    bfail=fail&(cf==b_like)
    sub=Pipeline([("scale",StandardScaler()),
                  ("km",KMeans(n_clusters=2,n_init=80,random_state=27420))])
    sub.fit(Xf[bfail]);sf=sub.predict(Xf);so=sub.predict(Xo)
    sstats={}
    for c in (0,1):
        m=bfail&(sf==c);under=int(np.sum(Gf[m]<yf[m]))
        sstats[c]={"rows":int(m.sum()),"under":under,
                   "under_rate":float(under/max(1,int(m.sum())))}
    low=max((0,1),key=lambda c:(sstats[c]["under_rate"],sstats[c]["rows"]))
    return (cf==b_like)&(sf==low),(co==b_like)&(so==low),stats,sstats,b_like,low

def audio_feature_map(dataset,items):
    wanted={m for _,m,_ in items}
    tracks={t.annotation_member:t for t in index_guitarset(dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"audio coverage")
    by={}
    for key,m,start in items:by.setdefault(m,[]).append((key,start))
    out={}
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member)
        s=np.asarray(au.samples,float)/32768.
        for key,start in by[member]:
            q=stage_features(s,int(start));require(q is not None,"stage feature missing")
            out[key]=q
        print(json.dumps({"recording":member,"stage_features":len(out)}),flush=True)
    return out

def stage_model(cases,case_features):
    pre=np.asarray([case_features[i]["pre"] for i in range(len(cases))],float)
    amp=np.asarray([case_features[i]["amp_range"] for i in range(len(cases))],float)
    y=np.asarray([1 if r["group"]=="K3_regressed" else 0 for r in cases],int)
    pre_thr=float(np.median(pre));amp_thr=float(np.median(amp))
    veto1=(pre>=pre_thr)&(amp<=amp_thr)
    survive=~veto1
    X=np.asarray([[case_features[i]["exc_shared_range"],case_features[i]["onset_contrast"]]
                  for i in range(len(cases)) if survive[i]],float)
    yy=y[survive]
    require(len(np.unique(yy))==2,"stage2 internal binary collapse")
    m=Pipeline([("scale",StandardScaler()),
                ("lr",LogisticRegression(C=.1,max_iter=3000,class_weight="balanced"))])
    m.fit(X,yy)
    return pre_thr,amp_thr,m,{
      "cases":len(cases),"stage1_veto_K2":int(np.sum(veto1&(y==0))),
      "stage1_veto_K3":int(np.sum(veto1&(y==1))),
      "stage2_train_rows":int(survive.sum()),
      "stage2_train_K2":int(np.sum(yy==0)),"stage2_train_K3":int(np.sum(yy==1))
    }

def main():
    ap=argparse.ArgumentParser()
    for n in ("bundle","config","uniform-weights","freeze-weights","outer-group-predictions",
              "outer-b-low-predictions","dataset","internal-cases","stage1-report",
              "confidence-report","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),"refusing overwrite")
    a.output.mkdir(parents=True)

    s1sel=json.loads(a.stage1_report.read_text())
    require(s1sel["strict_internal_pass"]["all_folds_nonnegative_gain"] is True,"stage1 not locked")
    conf=json.loads(a.confidence_report.read_text())
    q60=conf["results"]["0.6"]
    require(q60["all_folds_nonnegative"] and q60["all_players_nonnegative"] and q60["all_styles_nonnegative"],
            "p=.60 robustness report drift")
    require(q60["incremental_gain"]==13,"p=.60 internal gain drift")

    cache,parts,_=load_bundle(a.bundle,a.config)
    fit=np.asarray(parts["final_fit"],np.int64);outer=np.asarray(parts["outer"],np.int64)
    require(len(outer)==15279 and not np.intersect1d(fit,outer).size,"partition drift")
    yall=np.minimum(cache["exact"].astype(np.int32),6);yf=yall[fit];yo=yall[outer]

    model=build_robust_model(a.uniform_weights,a.freeze_weights)
    Pf,Gf=predict(model,cache,fit);Po,Go=predict(model,cache,outer)

    frozen_group=load_npz(a.outer_group_predictions)
    np.testing.assert_array_equal(frozen_group["global_index"],outer)
    np.testing.assert_array_equal(frozen_group["k"],yo)
    np.testing.assert_array_equal(frozen_group["predicted"],Go)
    np.testing.assert_allclose(frozen_group["probability"],Po,rtol=0,atol=1e-6)

    Xf=structural_features(cache,fit,Pf);Xo=structural_features(cache,outer,Po)
    blow_f,blow_o,cstats,sstats,bid,lid=reconstruct_blow(Xf,Xo,yf,Gf)

    frozen_b=load_npz(a.outer_b_low_predictions)
    np.testing.assert_array_equal(frozen_b["global_index"],outer)
    np.testing.assert_array_equal(frozen_b["k"],yo)
    np.testing.assert_array_equal(frozen_b["base_predicted"],Go)
    frozen_mask=frozen_b["B_low_mask"].astype(bool)
    reconstructed_apply=blow_o&np.isin(Go,(2,3,4))
    require(np.array_equal(frozen_mask,reconstructed_apply),"outer B_low mask mismatch")

    # Fit the source pair/triplet residual classifier on all internal B_low+base-K3.
    cand_f=blow_f&(Gf==3);cand_o=blow_o&(Go==3)
    idsf=fit[cand_f];idso=outer[cand_o]
    rf=audio_rows(cache,idsf,a.dataset);ro=audio_rows(cache,idso,a.dataset)
    Xrf,_,vf,_=extracted_matrix(rf);Xro,_,vo,_=extracted_matrix(ro)
    yfc=yf[cand_f];yoc=yo[cand_o]
    train=vf&np.isin(yfc,(2,3))
    require(np.any(yfc[train]==2) and np.any(yfc[train]==3),"residual train collapse")
    residual=fit_classifier(Xrf[train],(yfc[train]==2).astype(int))
    pout=np.full(len(idso),np.nan)
    pout[vo]=residual.predict_proba(Xro[vo])[:,1]
    source_local=vo&(pout>=.5)
    source_ids=idso[source_local]
    source_y=yoc[source_local]

    # Train the two veto stages only from the frozen OOF 108/125 internal cases.
    cases=load_cases(a.internal_cases)
    train_items=[(i,str(r["recording_id"]),int(r["start_sample"])) for i,r in enumerate(cases)]
    casefeat=audio_feature_map(a.dataset,train_items)
    pre_thr,amp_thr,s2,stage_training=stage_model(cases,casefeat)

    outer_items=[(int(gid),str(cache["members"][gid]),int(cache["cluster_start_samples"][gid]))
                 for gid in source_ids]
    outerfeat=audio_feature_map(a.dataset,outer_items) if len(outer_items) else {}
    veto1=np.zeros(len(source_ids),bool);veto2=np.zeros(len(source_ids),bool);p3=np.full(len(source_ids),np.nan)
    for i,gid in enumerate(source_ids):
        q=outerfeat[int(gid)]
        veto1[i]=q["pre"]>=pre_thr and q["amp_range"]<=amp_thr
        if not veto1[i]:
            xx=np.asarray([[q["exc_shared_range"],q["onset_contrast"]]],float)
            p3[i]=float(s2.predict_proba(xx)[0,1])
            veto2[i]=p3[i]>=P_STAGE2

    keep1=~veto1;keep2=~(veto1|veto2)
    source_mask_global=np.zeros(len(outer),bool)
    stage1_keep_global=np.zeros(len(outer),bool)
    final_keep_global=np.zeros(len(outer),bool)
    pos={int(g):i for i,g in enumerate(outer)}
    for gid,k1,k2 in zip(source_ids,keep1,keep2):
        j=pos[int(gid)];source_mask_global[j]=True;stage1_keep_global[j]=bool(k1);final_keep_global[j]=bool(k2)

    pred_source=Go.copy();pred_source[source_mask_global]=2
    pred_s1=Go.copy();pred_s1[stage1_keep_global]=2
    pred_final=Go.copy();pred_final[final_keep_global]=2

    base_m=metrics_simple(yo,Go);src_m=metrics_simple(yo,pred_source)
    s1_m=metrics_simple(yo,pred_s1);final_m=metrics_simple(yo,pred_final)
    source_counts=action_count(yo,source_mask_global)
    s1_block=action_count(yo,source_mask_global&~stage1_keep_global)
    s2_block=action_count(yo,stage1_keep_global&~final_keep_global)
    final_actions=action_count(yo,final_keep_global)

    report={
      "status":"completed",
      "protocol":{
        "experiment":"v273_outer_residual_two_stage_p060",
        "outer_fold":3,"fresh_independent_outer_validation":False,
        "reason_not_fresh":"fold 3 has been exposed historically in prior B_low/diagnostic evaluations",
        "outer_tuning":False,
        "base":"freeze_local_combo + candidate_hidden1 [42,52,61,64]",
        "B_low_convention":"same pooled-internal clustering and seeds as outer B_low run 37316567815",
        "source_residual":"pair/triplet LR C=1 balanced; action p(K2)>=0.5",
        "stage1":"veto if pre>=internal OOF median AND amp_range<=internal OOF median",
        "stage2":"balanced LR C=.1 on [exc_shared_range,onset_contrast], veto p(K3)>=0.60",
        "stage2_threshold":P_STAGE2,"automatic_promotion":False
      },
      "verification":{
        "outer_group_prediction_replay_equal":True,
        "outer_B_low_mask_replay_equal":True,
        "internal_stage1_selection_run":37532627926,
        "internal_stage2_confidence_run":37535903476
      },
      "internal_fit":{
        "coarse_clusters":{str(k):v for k,v in cstats.items()},
        "B_like_cluster":int(bid),"B_subclusters":{str(k):v for k,v in sstats.items()},
        "B_low_cluster":int(lid),
        "residual_candidate_rows":int(cand_f.sum()),
        "residual_valid_rows":int(vf.sum()),
        "residual_K23_train_rows":int(train.sum()),
        "stage1_pre_threshold":pre_thr,"stage1_amp_range_threshold":amp_thr,
        "stage_policy_training":stage_training
      },
      "outer":{
        "B_low_rows":int(frozen_mask.sum()),
        "B_low_baseK3_candidates":int(cand_o.sum()),
        "valid_residual_candidates":int(vo.sum()),
        "source_residual_actions":source_counts,
        "stage1_veto_blocked":s1_block,
        "stage2_veto_blocked":s2_block,
        "final_actions":final_actions,
        "base_metrics":base_m,"source_metrics":src_m,
        "stage1_metrics":s1_m,"final_metrics":final_m,
        "source_effect_vs_base":effect(yo,Go,pred_source),
        "stage1_effect_vs_base":effect(yo,Go,pred_s1),
        "final_effect_vs_base":effect(yo,Go,pred_final)
      }
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    np.savez_compressed(a.output/"predictions.npz",
      global_index=outer,k=yo,base_predicted=Go,source_predicted=pred_source,
      stage1_predicted=pred_s1,final_predicted=pred_final,
      source_action_mask=source_mask_global.astype(np.uint8),
      stage1_keep_mask=stage1_keep_global.astype(np.uint8),
      final_keep_mask=final_keep_global.astype(np.uint8))

    def pct(x):return 100*x
    lines=["# Exploratory outer fold 3 — residual + two-stage p0.60","",
      "**Fold 3 is historically exposed; this is not fresh independent validation.** No outer tuning occurred.","",
      "| Measure | Robust base | source residual | + stage1 | + stage2 p0.60 |",
      "|---|---:|---:|---:|---:|",
      f"| Exact-K global | {pct(base_m['exact']):.3f}% | {pct(src_m['exact']):.3f}% | {pct(s1_m['exact']):.3f}% | {pct(final_m['exact']):.3f}% |",
      f"| Exact-K poly | {pct(base_m['poly_exact']):.3f}% | {pct(src_m['poly_exact']):.3f}% | {pct(s1_m['poly_exact']):.3f}% | {pct(final_m['poly_exact']):.3f}% |","",
      f"Source residual actions: **{source_counts['corrections']}/{source_counts['regressions']}**, other-K {source_counts['other_k']}, net **{source_counts['net']:+d}**.",
      f"Stage1 blocks: **{s1_block['regressions']} regressions / {s1_block['corrections']} corrections**, gain **{s1_block['regressions']-s1_block['corrections']:+d}**.",
      f"Stage2 blocks: **{s2_block['regressions']} regressions / {s2_block['corrections']} corrections**, gain **{s2_block['regressions']-s2_block['corrections']:+d}**.",
      f"Final kept actions: **{final_actions['corrections']}/{final_actions['regressions']}**, other-K {final_actions['other_k']}, net **{final_actions['net']:+d}**.",
      f"Final paired Exact-K vs robust base: **{report['outer']['final_effect_vs_base']['net']:+d}**.","",
      "No automatic promotion."
    ]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":main()
