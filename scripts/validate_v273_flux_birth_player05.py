"""Frozen flow/birth model vs reused unseen-performer player05 evaluation.

'infer' never accepts target labels. It freezes the trained 124-feature classifier,
a global threshold fitted on 4 internal folds, and holdout predictions before
'score' is permitted to open the separate label archive.
Player05 has been used in earlier, unrelated experiments: not a fresh holdout.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.extract_v273_energy_flow import energy_flow_features, SAMPLE_RATE
from scripts.extract_v273_energy_continuity import continuity_features
from scripts.summarize_v273_energy_flux_combined import FOLDS, clf, inner, choose, require

PREFIXES = ("flux__", "absolute__", "logflow__", "relative__", "weighted_relative__")

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_train(root):
    rows=[]
    seen=set()
    for path in sorted(root.rglob("report.json")):
        rep=json.loads(path.read_text())
        if rep.get("experiment") != "v273_energy_flux_combined_extract":continue
        f=int(rep["fold"]);require(f in FOLDS and f not in seen,"bad fold")
        seen.add(f)
        rr=[json.loads(s) for s in (path.parent/"rows.jsonl").read_text().splitlines() if s.strip()]
        require(len(rr)==rep["rows"] and all(r["fold"]==f for r in rr),"row/fold mismatch")
        rows.extend(rr)
    require(seen==set(FOLDS) and len(rows)==488,"train cohort drift")
    rows.sort(key=lambda r:(r["fold"],r["row_id"]))
    names=sorted(n for n in rows[0]["features"] if n.startswith(PREFIXES))
    require(len(names)==124 and len({r["row_id"] for r in rows})==488,"feature/id drift")
    y=np.asarray([r["true_k"]==2 for r in rows],int)
    folds=np.asarray([r["fold"] for r in rows],int)
    require(y.sum()==216 and all(r["true_k"] in (2,3) for r in rows),"train labels drift")
    X=np.asarray([[r["features"][n] for n in names] for r in rows],float)
    require(np.isfinite(X).all(),"train nonfinite")
    return X,y,folds,names

def infer(args):
    require(not args.output.exists(),"refusing overwrite")
    frozen=args.frozen
    expected=args.checksum.read_text().strip()
    require(len(expected)==64 and digest(frozen)==expected,"prior frozen holdout input SHA drift")
    X,y,folds,names=load_train(args.input_root)
    oof=inner(X,y,folds,np.ones(len(y),bool))
    choice=choose(y,oof)
    threshold=float(choice["threshold"])
    estimator=clf().fit(X,y)

    with np.load(frozen,allow_pickle=False) as z:
        require(set(("members","cluster_start_samples","robust_base_predicted","residual_candidate_ids")) <= set(z.files),"holdout schema drift")
        members=z["members"].astype(str)
        samples=z["cluster_start_samples"].astype(np.int64)
        base=z["robust_base_predicted"].astype(np.int32)
        ids=z["residual_candidate_ids"].astype(np.int64)
    require(len(base)==len(members)==len(samples),"holdout array length drift")
    require(len(ids)>0 and np.all((ids>=0)&(ids<len(base))) and len(set(ids.tolist()))==len(ids),"bad holdout ids")
    require(np.all(base[ids]==3),"holdout must be base K3 candidates")
    require(set(m[:2] for m in members)=={"05"},"wrong performer")
    wanted=set(members[ids])
    tracks={t.annotation_member:t for t in index_guitarset(args.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"holdout audio coverage missing")
    audio={}
    for member,track in tracks.items():
        w=decode_pcm16_mono_wav(track.audio_zip,track.audio_member)
        require(int(w.sample_rate)==SAMPLE_RATE,"audio sample-rate drift")
        audio[member]=np.asarray(w.samples,np.float64)/32768.0
    hold=[]
    for i in ids:
        e=energy_flow_features(audio[members[i]],int(samples[i]))
        c=continuity_features(audio[members[i]],int(samples[i]))
        feat={**e,**c}
        require(len(feat)==len(e)+len(c),"feature collision")
        hold.append([float(feat[n]) for n in names])
    Xh=np.asarray(hold,float)
    require(Xh.shape==(len(ids),124) and np.isfinite(Xh).all(),"holdout feature invalid")
    prob=estimator.predict_proba(Xh)[:,1]
    actions=prob>=threshold
    proposal=base.copy()
    proposal[ids[actions]]=2
    require(np.all(np.isin(proposal,np.arange(7))),"prediction out of bounds")
    args.output.mkdir(parents=True)
    dest=args.output/"predictions-before-labels.npz"
    np.savez_compressed(dest,robust_base_predicted=base,flux_birth_predicted=proposal,
                        residual_candidate_ids=ids,flux_birth_candidate_probability=prob,
                        flux_birth_action_mask=(proposal!=base).astype(np.uint8),
                        members=members,cluster_start_samples=samples)
    h=digest(dest)
    (args.output/"prediction-sha256.txt").write_text(h+"\n")
    info=dict(experiment="v273_flow_birth_player05_transfer",phase="prelabel_inference",
              previous_holdout_sha256=expected,prediction_sha256=h,
              trained_rows=len(y),train_folds=list(FOLDS),train_K2=int(y.sum()),train_K3=int(len(y)-y.sum()),
              features=len(names),frozen_threshold=threshold,inner_selected_net=choice["net"],
              holdout_candidates=len(ids),holdout_predicted_actions=int(actions.sum()),
              holdout_tracks=len(wanted),holdout_label_file_opened=False,
              no_player05_training=True,previously_exposed_holdout=True,
              automatic_promotion=False)
    (args.output/"prelabel-report.json").write_text(json.dumps(info,indent=2,sort_keys=True)+"\n")
    print(json.dumps(info,sort_keys=True),flush=True)

def simple(y,p):
    kpoly=y>=2
    return dict(rows=len(y),global_exact=float(np.mean(y==p)),
                poly_rows=int(kpoly.sum()),poly_exact=float(np.mean(y[kpoly]==p[kpoly])),
                per_K={str(k):dict(rows=int(np.sum(y==k)),correct=int(np.sum((y==k)&(p==k)))) for k in range(7)})

def score(args):
    require(not args.output.exists(),"refusing overwrite")
    predicted=args.predictions
    expected=args.checksum.read_text().strip()
    require(digest(predicted)==expected,"prelabel freeze corrupted")
    with np.load(predicted,allow_pickle=False) as z:
        base=z["robust_base_predicted"].astype(int)
        proposal=z["flux_birth_predicted"].astype(int)
        ids=z["residual_candidate_ids"].astype(int)
        probabilities=z["flux_birth_candidate_probability"]
    with np.load(args.labels,allow_pickle=False) as z:
        require("true_k" in z.files and "robust_base_predicted" in z.files,"label archive schema drift")
        y=z["true_k"].astype(int)
        np.testing.assert_array_equal(z["robust_base_predicted"],base)
    require(len(y)==len(base)==len(proposal) and np.isin(y,range(7)).all(),"bad score arrays")
    changed=proposal!=base
    require(np.all(~changed | (base==3)), "changes outside base K3")
    require(set(np.flatnonzero(changed)).issubset(set(ids)),"action outside preselected candidates")
    corrections=int(np.sum(changed&(y==2)))
    regressions=int(np.sum(changed&(y==3)))
    other=int(np.sum(changed&~np.isin(y,(2,3))))
    before=simple(y,base);after=simple(y,proposal)
    net=int(np.sum(proposal==y)-np.sum(base==y))
    require(net==corrections-regressions,"non-additive Exact-K scoring")
    info=dict(status="completed",experiment="v273_flow_birth_player05_transfer",
              previously_exposed_holdout=True,performer_disjoint_from_training=True,
              independent_fresh_holdout=False,automatic_promotion=False,
              previous_holdout_frozen_sha256=digest(args.labels),
              new_predictions_sha256=expected,
              baseline=before,corrected=after,
              actions=int(changed.sum()),corrections=corrections,
              regressions=regressions,other_K_actions=other,exact_K_net=net,
              global_exact_pp=(after["global_exact"]-before["global_exact"])*100,
              poly_exact_pp=(after["poly_exact"]-before["poly_exact"])*100)
    args.output.mkdir(parents=True)
    (args.output/"report.json").write_text(json.dumps(info,indent=2,sort_keys=True)+"\n")
    lines=["# Flow/birth transfer to reused player05 holdout","",
           "Player05 is unseen by this classifier but previously used in other research audits; not fresh.",
           "Predictions were frozen before their labels were read in a different CI job.","",
           "| Measure | Robust base | Flux/birth 124 |","|---|---:|---:|",
           f"| Exact-K global | {before['global_exact']*100:.4f}% | {after['global_exact']*100:.4f}% |",
           f"| Exact-K poly | {before['poly_exact']*100:.4f}% | {after['poly_exact']*100:.4f}% |","",
           f"Actions {info['actions']}, corrections {corrections}, regressions {regressions}, other-K {other}, net {net:+d}.",
           "No automatic promotion."]
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("infer")
    for k in ("input-root","frozen","checksum","dataset","output"):
        q.add_argument("--"+k,type=Path,required=True)
    q=sub.add_parser("score")
    for k in ("predictions","checksum","labels","output"):
        q.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    if a.command=="infer":infer(a)
    else:score(a)

if __name__=="__main__":
    main()
