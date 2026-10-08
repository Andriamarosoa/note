"""Native K0..K6 paired Exact-K evaluation of harmonic-life-cycle corrections.

All 59,309 ORIGINAL events are scored; only the original model's 7,493
K2/K3/K4 predictions are eligible for revision. True labels never select
candidate rows. Outer folds 0/1/2/4 are held out by recording and player05
and fold3 excluded. Thresholds are chosen using train-fold-only internal OOF.
No weights/thresholds of freeze_local_combo are modified.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scripts.yourmt3_exactk_common import FOLDS,digest,metrics,paired,require
from scripts.extract_v273_harmonic_global import EXPERIMENT as EXTRACTION

ARMS={
    "base_class_only":(("_base",),),
    "harmonic_spectral":(("features",("spectral__",)),),
    "harmonic_lifecycle":(("features",("birth__","persistence__","damping__")),),
    "harmonic_full":(("features",tuple()),),
    "fundamental_full":(("fundamental",tuple()),),
    "detuned_full":(("detuned",tuple()),),
    "scrambled_full":(("scrambled",tuple()),),
}
CLASSES=np.arange(7,dtype=int)

def model():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=0.1, max_iter=3000, solver="lbfgs", random_state=27402
    ))

def choose_threshold(y,base,proposal,margin):
    """Find best abstaining threshold on TRAINING OOF only, never outer val."""
    require(len(y)==len(base)==len(proposal)==len(margin),"selection dimension mismatch")
    eligible=proposal!=base
    if not eligible.any():
        return dict(threshold=1e6,train_oof_net=0,train_oof_actions=0)
    values=np.unique(margin[eligible])
    # Maximal net gain; exact tie favors fewer corrections (higher cutoff).
    thresholds=np.r_[np.nextafter(values.max(),np.inf),values]
    best=(0,0,1e6)
    for t in thresholds:
        act=eligible & (margin>=t)
        corrected=np.where(act,proposal,base)
        net=int(np.sum(corrected==y)-np.sum(base==y))
        actions=int(np.sum(act))
        key=(net,-actions,float(t))
        if key>best:best=key
    return dict(threshold=best[2],train_oof_net=best[0],train_oof_actions=-best[1])

def inner_oof(X,y,base,fold):
    oof_pred=np.full(len(y),-1,int)
    oof_probs=np.full((len(y),7),np.nan)
    for f in FOLDS:
        val=fold==f
        if not val.any():continue
        tr=fold!=f
        require(tr.any() and set(y[tr])==set(CLASSES.tolist()),"inner fold missing label")
        clf=model().fit(X[tr],y[tr])
        require(np.array_equal(clf[-1].classes_,CLASSES),"class order changed")
        oof_probs[val]=clf.predict_proba(X[val])
        oof_pred[val]=np.argmax(oof_probs[val],axis=1).astype(int)
    require(np.isfinite(oof_probs).all() and np.all(oof_pred>=0),"OOF incomplete")
    delta=oof_probs[np.arange(len(y)),oof_pred]-oof_probs[np.arange(len(y)),base]
    return oof_pred,delta

def load_cohort(root):
    records=[]
    manifests=json.loads((root/"manifest.json").read_text())
    require(manifests["status"]=="verified" and
            manifests["fold_3_evaluated"] is False and
            manifests["player_05_evaluated"] is False,"cohort security drift")
    for fold in FOLDS:
        file=root/f"fold-{fold}.npz"
        require(digest(file)==manifests["by_fold"][str(fold)]["cohort_sha256"],"SHA mismatch")
        with np.load(file,allow_pickle=False) as z:
            idx=z["global_index"].astype(int)
            y=z["k"].astype(int)
            p=z["baseline"].astype(int)
            mem=z["member"].astype(str)
            starts=z["starts"].astype(int)
        require(len(idx)==len(y)==len(p)==len(mem)==len(starts),"cohort lengths")
        require(all(m[:2] in ("00","01","02","03","04") for m in mem),"player leakage")
        records.append(dict(fold=int(fold),global_index=idx,y=y,base=p,member=mem,start=starts))
    y=np.concatenate([r["y"] for r in records])
    base=np.concatenate([r["base"] for r in records])
    ids=np.concatenate([r["global_index"] for r in records])
    folds=np.concatenate([np.full(len(r["y"]),r["fold"],int) for r in records])
    member=np.concatenate([r["member"] for r in records])
    start=np.concatenate([r["start"] for r in records])
    require(len(y)==59309 and int(np.sum(y==base))==48454,"global baseline drift")
    require(int(np.sum(y>=2))==7385 and int(np.sum((y==base)&(y>=2)))==2530,"poly baseline drift")
    require(len(set(ids))==len(ids),"duplicate global events")
    bymember={}
    for m,f in zip(member,folds):
        if m in bymember:require(bymember[m]==f,"RECORDING LEAKAGE")
        else:bymember[m]=f
    return y,base,ids,folds,member,start

def load_features(root,y,base,ids,folds,member,start):
    lookup={int(i):pos for pos,i in enumerate(ids)}
    collected=[];seen=set()
    reports={}
    for file in sorted(root.rglob("report.json")):
        d=json.loads(file.read_text())
        if d.get("experiment")!=EXTRACTION:continue
        f=int(d["fold"]);require(f in FOLDS and f not in reports,"unexpected/duplicate fold")
        reports[f]=d
        rows=[json.loads(line) for line in (file.parent/"rows.jsonl").read_text().splitlines() if line.strip()]
        require(len(rows)==d["eligible_rows"],"feature rows drift")
        for row in rows:
            i=int(row["global_index"])
            require(i in lookup and i not in seen,"unrecognized/duplicate id")
            seen.add(i)
            ix=lookup[i]
            require(int(row["fold"])==int(folds[ix])==f,"fold identity changed")
            require(row["recording_id"]==member[ix] and int(row["start_sample"])==int(start[ix]),"timestamp changed")
            require(int(row["true_k"])==int(y[ix]) and int(row["baseline_k"])==int(base[ix]),"target/base drift")
            require(base[ix] in (2,3,4),"label-dependent candidate selection")
            collected.append((ix,row))
    require(set(reports)==set(FOLDS),"missing fold artifacts")
    chosen=np.flatnonzero(np.isin(base,(2,3,4)))
    require(len(chosen)==7493 and set(chosen)=={i for i,_ in collected},"candidate coverage gap")
    collected.sort(key=lambda t:t[0])
    positions=np.asarray([p for p,_ in collected],int)
    rows=[r for _,r in collected]
    names=sorted(rows[0]["features"])
    require(all(sorted(row[key])==names for row in rows for key in ("features","fundamental","detuned","scrambled")),"feature schema mismatch")
    return positions,rows,names,reports

def get_matrix(rows,names,source,prefixes):
    colnames=names if not prefixes else [n for n in names if n.startswith(prefixes)]
    require(len(colnames)>0,"empty feature family")
    A=np.asarray([[row[source][col] for col in colnames] for row in rows],float)
    require(np.isfinite(A).all(),"nonfinite feature")
    return A

def evaluate_arm(key,parts,rows,names,y,base,fold,position):
    yc=y[position];bc=base[position];fc=fold[position]
    if key=="base_class_only":
        data=np.zeros((len(position),0),float)
    else:
        data=np.column_stack([get_matrix(rows,names,source,prefs) for source,prefs in parts])
    # Always condition learned decision on the untouched baseline K class.
    onehot=np.eye(3)[bc-2]
    X=np.column_stack([data,onehot])
    oof_raw=np.full(len(y),-1,int)
    oof_gated=np.full(len(y),-1,int)
    oof_raw[:]=base;oof_gated[:]=base
    fold_diagnostics={}
    for f in FOLDS:
        fit=fc!=f
        val=fc==f
        require(int(np.sum(val))>0 and set(yc[fit])==set(CLASSES.tolist()),"outer fold missing class")
        Xtr=X[fit]; ytr=yc[fit]; btr=bc[fit]; ftr=fc[fit]
        inner_pred,inner_margin=inner_oof(Xtr,ytr,btr,ftr)
        chosen=choose_threshold(ytr,btr,inner_pred,inner_margin)
        trained=model().fit(Xtr,ytr)
        probs=trained.predict_proba(X[val])
        proposal=np.argmax(probs,axis=1).astype(int)
        margin=probs[np.arange(len(proposal)),proposal]-probs[np.arange(len(proposal)),bc[val]]
        action=(proposal!=bc[val])&(margin>=chosen["threshold"])
        val_idx=position[val]
        oof_raw[val_idx]=proposal
        oof_gated[val_idx]=np.where(action,proposal,bc[val])
        cr=paired(y[val_idx],base[val_idx],oof_gated[val_idx])["global"]
        fold_diagnostics[str(f)]=dict(
            candidates=int(val.sum()),
            selected_threshold=chosen["threshold"],
            train_inner_oof_net=chosen["train_oof_net"],
            train_inner_oof_actions=chosen["train_oof_actions"],
            outer_actions=int(action.sum()),
            outer_paired=cr,
            outer_baseline_exact=int(np.sum(y[val_idx]==base[val_idx])),
            outer_corrected_exact=int(np.sum(y[val_idx]==oof_gated[val_idx])),
        )
    return dict(dimensions=int(X.shape[1]),folds=fold_diagnostics,
                raw=metrics(y,oof_raw),raw_paired=paired(y,base,oof_raw),
                gated=metrics(y,oof_gated),gated_paired=paired(y,base,oof_gated),
                raw_predictions=oof_raw, gated_predictions=oof_gated)

def pct(v):return f"{100*v:.4f}%"
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cohort",type=Path,required=True)
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    require(not args.output.exists(),"refusing overwrite")
    y,base,ids,fold,member,start=load_cohort(args.cohort)
    position,rows,names,reps=load_features(args.input_root,y,base,ids,fold,member,start)
    baseline=metrics(y,base)
    report=dict(status="completed",
        experiment="v273_harmonic_lifecycle_native_exactk_global",
        cohort=dict(rows=len(y),eligible_rows=len(position),
                    immutable_rows=len(y)-len(position),
                    folds=list(FOLDS),fold3_used=False,player05_used=False,
                    native_original_global_correct=baseline["correct"],
                    native_original_poly_correct=baseline["poly"]["correct"]),
        baseline=baseline,
        setting="Seven-class K0..K6 model allowed to revise only frozen baseline predictions K2,3,4",
        policy="fixed 7-way logistic C=0.1 with baseline one-hot; gating selected train-only OOF",
        interpretation_limits=[
            "Native full 59309 global Exact-K is measured, not inferred from a restricted K2 K3 K4 balanced classifier",
            "Predictor only adjusts original baseline K2/3/4 outputs; all others remain exactly unchanged",
            "Harmonic and source observables use 160ms post-onset future audio, so NOT online causal",
            "Existing development-exposed folds; not a genuinely independent validation",
            "Template trajectories are proxies, no confirmed physical note isolation or true Navier-Stokes",
            "K means native attack assignment count, not necessarily number of concurrently sustained notes",
            "Player05 and fold3 excluded; freeze_local_combo unchanged; no automatic promotion",
        ],
        arms={},automatic_promotion=False)
    args.output.mkdir(parents=True)
    saved=dict(global_index=ids,true_k=y,baseline=base,fold=fold)
    for key,parts in ARMS.items():
        q=evaluate_arm(key,parts,rows,names,y,base,fold,position)
        raw=q.pop("raw_predictions");gate=q.pop("gated_predictions")
        saved[key+"_raw"]=raw;saved[key+"_gated"]=gate
        report["arms"][key]=q
        require(q["gated"]["rows"]==59309 and q["gated"]["poly"]["rows"]==7385,"partial metric")
        require(q["gated"]["correct"]==baseline["correct"]+q["gated_paired"]["global"]["net"],"global net mismatch")
    np.savez_compressed(args.output/"predictions-all-native.npz",**saved)
    (args.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    lines=["# Actual native K0..K6 Exact-K — energy/harmonic trajectory corrections","",
           f"Frozen reference: **{baseline['correct']}/{baseline['rows']} = {pct(baseline['exact'])}** global; "+
           f"**{baseline['poly']['correct']}/{baseline['poly']['rows']} = {pct(baseline['poly']['exact'])}** poly K2..K6.",
           f"Full population {len(y)}; eligible based only on baseline predicted K2/K3/K4: {len(position)}; other predictions unchanged.",
           "", "| Arm, train-fold-gated | global Exact-K | Δ global (pp) | Poly Exact-K | Δ poly (pp) | fixes | regressions | net | fold 2 net |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key,q in report["arms"].items():
        met=q["gated"];pair=q["gated_paired"]["global"]
        lines.append(f"| {key} | {pct(met['exact'])} | {100*(met['exact']-baseline['exact']):+.4f} | {pct(met['poly']['exact'])} | {100*(met['poly']['exact']-baseline['poly']['exact']):+.4f} | {pair['corrections']} | {pair['regressions']} | {pair['net']:+d} | {q['folds']['2']['outer_paired']['net']:+d} |")
    lines+=["", "## Per-K, gated variants (all native K0..K6 events)", "",
            "| Arm | " + " | ".join(f"K{k}" for k in range(7)) + " |",
            "|---|"+"---:|"*7]
    for key,q in report["arms"].items():
        lines.append("| "+key+" | "+" | ".join(pct(q["gated"]["by_k"][str(k)]["exact"]) for k in range(7))+" |")
    lines+=["","## Raw (no abstention) control","",
            "| Arm | Exact-K global | Exact-K poly | Net paired |",
            "|---|---:|---:|---:|"]
    for key,q in report["arms"].items():
        lines.append(f"| {key} | {pct(q['raw']['exact'])} | {pct(q['raw']['poly']['exact'])} | {q['raw_paired']['global']['net']:+d} |")
    lines += (
        ["", "## Limits and decision"]
        + ["- " + x for x in report["interpretation_limits"]]
        + ["", "No reference promotion."]
    )
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

if __name__=="__main__":main()
