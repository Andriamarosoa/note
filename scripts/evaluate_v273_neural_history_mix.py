"""First native K0..K6 neural mixture with learnable corrective/fix heads.

Only VERIFIED and temporally aligned H0..H5 expert predictions can be
connected; archived head registry has thousands of source candidates but
these are NOT automatically executable on the native 59309-row cohort.

Eight NEW action adapters are available: 2->3, 3->2, 3->4, 4->3
and three preserve-K guards plus default keep. These are selectable heads,
not hardcoded post-prediction fixes. Their predicted probabilities and
per-true-K audits are fed into the same Keras attention selector.

Each outer fold uses cross-fitted inner-head outputs for selector training.
Each inner fold's audit descriptors exclude that fold, preventing a
row's label from entering its audit input.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from scripts.evaluate_v273_dynamic_heads import (
    EXPERTS, oof_experts, outer_experts
)
from scripts.summarize_v273_harmonic_global import (
    load_cohort, load_features, get_matrix
)
from scripts.learn_v273_neural_head_selector import (
    ClassConditionalAuditSelector, audited_head_descriptor, form_targets,
    train_loss, KEEP
)
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require

SEED=27402
CORRECTIVE=((2,3),(3,2),(3,4),(4,3))
FIXES=(2,3,4)
# The eighth action adapter preserves the original decision in uncertain cases.
ACTION_HEADS=("C23","C32","C34","C43","F_keep2","F_keep3","F_keep4","F_keep_any")
EXPERT_NAMES=("H0_base","H1_spectral","H2_lifecycle","H3_harmonics",
              "H4_fundamentals","H5_sources")
HEADS=EXPERT_NAMES+ACTION_HEADS
H=len(HEADS)
assert H==14
EPOCHS=32

def build_candidate_logits(P,base):
    """Convert OOF/outside-train posteriors to eight-action head interface.

    Proposals are determined only by observable probabilities and baseline.
    A correction adapter exists separately from an accuracy/lifecycle expert.
    """
    P=np.asarray(P,np.float32);b=np.asarray(base,int)
    require(P.ndim==3 and P.shape[1:]==(6,5) and P.shape[0]==len(b),
            "expert alignment")
    n=len(b);p=np.zeros((n,H,8),"float32")
    active=np.zeros((n,H),bool)
    kind=np.zeros((n,H,4),"float32")
    # Structural support differs from learned usefulness. No true K used.
    # H0 is always a fallback; trained specialist outputs cover K2..K6.
    # Correction actions propose only their destination K.
    # Fix/KEEP actions influence abstention, not any forced target class.
    support=np.zeros((n,H,7),bool)
    support[:,0,:]=True
    support[:,1:6,2:7]=True
    # Standard experts: proposal posterior K2..K6, plus an abstention token.
    for i in range(6):
        q=np.maximum(P[:,i],1e-8)
        q=q/q.sum(axis=1,keepdims=True)
        share=.78 if i==0 else .93
        p[:,i,2:7]=share*q
        p[:,i,KEEP]=1-share
        active[:,i]=True
        kind[:,i,1 if i==0 else 0]=1.
    for ix,(source,dest) in enumerate(CORRECTIVE,start=6):
        kind[:,ix,2]=1.
        active[:,ix]=b==source
        support[:,ix,dest]=active[:,ix]
        # Multiple acoustic specialists (not the fixed baseline) determine
        # continuous proposal strength for the same correction transition.
        q=np.mean(P[:,1:,dest-2],axis=1)
        p[:,ix,KEEP]=1-.95*q
        p[:,ix,dest]=.95*q
    for i,k in enumerate(FIXES,start=10):
        kind[:,i,3]=1.
        active[:,i]=b==k
        p[:,i,KEEP]=.98
        p[:,i,k]=.02
    kind[:,13,3]=1.
    active[:,13]=True
    p[:,13,KEEP]=1.
    # Invalid actions are masked and assigned KEEP so a missing / inactive
    # correction has no decision effects or gradient contributions.
    for ix in range(H):
        p[~active[:,ix],ix,:]=0.
        p[~active[:,ix],ix,KEEP]=1.
    require(np.max(np.abs(p.sum(axis=2)-1))<1e-5,"bad action mass")
    require(np.all(active[:,0]),"no baseline fallback")
    require(np.all(support[:,0,:]),"no class fallback")
    require(np.all(~support[:,10:,:]),"KEEP-only fix has target-K weight")
    return np.log(np.maximum(p,1e-8)),active,kind,support


def cross_audits(y,baseline,fold,logits):
    """Crossfit audit priors across the 3 inner folds."""
    proposal=np.argmax(logits,axis=-1)
    audit=np.full((len(y),H,21),np.nan,"float32")
    for f in np.unique(fold):
        val=fold==f
        tr=~val
        stats=audited_head_descriptor(y,baseline,proposal,fold,tr)
        audit[val]=stats.reshape(1,H,21)
    require(np.isfinite(audit).all(),"cross-fitted audits incomplete")
    return audit


def full_train_audits(y,baseline,fold,logits):
    proposal=np.argmax(logits,axis=-1)
    return audited_head_descriptor(
        y,baseline,proposal,fold,np.ones(len(y),bool)
    ).reshape(1,H,21)


def inputs(logits,mask,kind,class_support,audit,context,baseline):
    return dict(
        head_logits=logits.astype("float32"),
        audit=audit.astype("float32"),
        head_type=kind.astype("float32"),
        head_mask=mask,
        head_k_mask=class_support,
        context=context.astype("float32"),
        baseline=np.eye(7,dtype="float32")[baseline]
    )


def fit_and_predict(train,truth,base_train,heldout):
    tf.keras.utils.set_random_seed(SEED)
    model=ClassConditionalAuditSelector(hidden=48)
    action=np.argmax(train["head_logits"],axis=-1)
    target,risk=form_targets(truth,base_train,action)
    optimizer=tf.keras.optimizers.Adam(.002)
    ds=tf.data.Dataset.from_tensor_slices(
        (train,target.astype("int32"),risk.astype("float32"))
    ).shuffle(len(target),seed=SEED,reshuffle_each_iteration=True).batch(256)
    for epoch in range(EPOCHS):
        losses=[]
        for b,t,reg in ds:
            with tf.GradientTape() as tape:
                out=model(b,training=True)
                loss=train_loss(out,t,reg,b["head_mask"],risk_weight=.25)
            grad=tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grad,model.trainable_variables))
            losses.append(float(loss))
        if epoch in (0,15,EPOCHS-1):
            print(json.dumps({"epoch":epoch+1,
                              "loss":float(np.mean(losses))}),flush=True)
    out=model(heldout,training=False)
    action=np.argmax(out["action_logits"].numpy(),axis=1)
    return (action,out["class_selection_weights"].numpy(),
            out["selection_weights"].numpy(),out["head_risk"].numpy())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cohort",type=Path,required=True)
    parser.add_argument("--features",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    require(not a.output.exists(),"refusing to overwrite native evaluation")
    y,base,ids,folds,member,start=load_cohort(a.cohort)
    pos,rows,names,_=load_features(a.features,y,base,ids,folds,member,start)
    require(len(pos)==7493 and len(y)==59309,"native cohort changed")
    yc=y[pos];bc=base[pos];fc=folds[pos]
    x={name:get_matrix(rows,names,field,prefix)
       for name,(field,prefix) in EXPERTS.items()}
    context_names=[z for z in names if
                   z.startswith(("spectral__","birth__","persistence__",
                                 "damping__"))][:26]
    raw=np.asarray([[r["features"][z] for z in context_names]
                    for r in rows],float)
    require(np.isfinite(raw).all(),"context invalid")

    pred=base.copy()
    all_weights=np.full((len(pos),H),np.nan,"float32")
    all_weights_per_k=np.full((len(pos),H,7),np.nan,"float32")
    all_risk=np.full((len(pos),H,2),np.nan,"float32")
    folds_info={}
    for vf in FOLDS:
        tr=fc!=vf;va=fc==vf
        require(tr.any() and va.any(),"invalid fold")
        ptr=oof_experts(
            {key:arr[tr] for key,arr in x.items()},yc[tr],bc[tr],fc[tr]
        )
        vtest=outer_experts(
            {key:{"train":arr[tr],"test":arr[va]}
             for key,arr in x.items()},yc[tr],bc[va]
        )
        train_logits,train_mask,train_kind,train_support=build_candidate_logits(ptr,bc[tr])
        val_logits,val_mask,val_kind,val_support=build_candidate_logits(vtest,bc[va])
        audits_train=cross_audits(yc[tr],bc[tr],fc[tr],train_logits)
        audits_test=np.broadcast_to(
            full_train_audits(yc[tr],bc[tr],fc[tr],train_logits),
            (int(va.sum()),H,21)
        ).copy()
        std=StandardScaler().fit(raw[tr])
        ctxtr=np.clip(std.transform(raw[tr]),-6,6)
        ctxval=np.clip(std.transform(raw[va]),-6,6)
        train=inputs(train_logits,train_mask,train_kind,train_support,
                     audits_train,ctxtr,bc[tr])
        val=inputs(val_logits,val_mask,val_kind,val_support,
                   audits_test,ctxval,bc[va])
        actions,class_weights,weights,risk=fit_and_predict(train,yc[tr],bc[tr],val)
        # KEEP is an explicit learned neural action. No manually-chosen
        # thresholds or after-the-fact correction direction vetoes.
        proposed=np.where(actions==KEEP,bc[va],actions)
        pred[pos[va]]=proposed
        all_weights[va]=weights
        all_weights_per_k[va]=class_weights
        all_risk[va]=risk
        idx=pos[va]
        z=paired(y[idx],base[idx],pred[idx])["global"]
        folds_info[str(vf)]=dict(
            heldout_events=int(len(idx)),changes=z["changed"],
            corrections=z["corrections"],regressions=z["regressions"],net=z["net"],
            mean_head_weight={n:float(np.mean(weights[:,i]))
                              for i,n in enumerate(HEADS)},
            # These are conditional on HYPOTHETICAL output K, not the real K.
            mean_head_weight_per_candidate_k={
                str(k):{name:float(np.mean(class_weights[:,i,k]))
                         for i,name in enumerate(HEADS)}
                for k in range(7)
            }
        )
        print(json.dumps({"fold":vf,"net":z["net"],
                          "corrections":z["corrections"],
                          "regressions":z["regressions"]}),flush=True)
    require(np.isfinite(all_weights).all() and
            np.isfinite(all_weights_per_k).all() and np.isfinite(all_risk).all(),
            "missing learning output")
    baseline=metrics(y,base)
    result=metrics(y,pred)
    comparison=paired(y,base,pred)
    require(result["correct"]==baseline["correct"]+comparison["global"]["net"],
            "full native global metrics mismatch")
    report=dict(
        status="completed",
        experiment="v273_trainable_class_conditional_attention_with_action_heads",
        original=baseline,neural=result,paired=comparison,by_fold=folds_info,
        head_names=list(HEADS),
        neural_architecture="Keras multi-head attention, distinct conditional head weights for each hypothetical K, independent KEEP attention, learned correction/regression risk",
        candidates=dict(full_events=len(y),active_prediction_rows=len(pos),
                        unchanged_rows=len(y)-len(pos)),
        data_integrity=dict(folds=list(FOLDS),fold3_used=False,
                            player05_used=False),
        limitations=[
            "Historical registry inventories all branch sources; only H0..H5 have verified aligned adapters currently",
            "Four correction heads and four keep/fix heads are NEW ACTION ADAPTERS based on available OOF specialists; they are not replays of every historical fix",
            "OOF audit descriptors exclude every inner train row's fold; heldout outer labels never enter gate or audit inputs",
            "Class-specific structural eligibility derives from head proposal capabilities, never true K labels",
            "Class-specific weights are learned from audio, heads, and OOF audits; true K unknown at inference",
            "For a baseline-predicted K2 event, candidate K3 heads remain eligible if they can propose K3",
            "KEEP-only fix heads cannot vote directly for a class; they inform learned abstention",
            "Classification includes K0..K6 and KEEP but only initially predicted K2/K3/K4 rows are eligible",
            "Audio feature horizon extends to 160ms after onset, not causal online",
            "Folds 0,1,2,4 exposed in prior research, thus NOT independent confirmation",
            "No promotion to freeze_local_combo; corrections/regressions on per K must be inspected"
        ],automatic_promotion=False
    )
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    np.savez_compressed(a.output/"neural-heads-predictions.npz",
                        global_index=ids,true_k=y,baseline=base,fold=folds,
                        corrected=pred,eligible_indices=ids[pos],
                        router_head_weights=all_weights,
                        router_head_weights_by_hypothetical_k=all_weights_per_k,
                        router_head_risk=all_risk)
    lines=["# Trainable class-conditional attention selector with correction/fix heads","",
           f"Native K0..K6 on {len(y)} events; {len(pos)} gated rows; "
           "no player05/fold3; original model unchanged.","",
           "| Model | Global Exact-K | Poly Exact-K | Corrections | Regressions | Net |",
           "|---|---:|---:|---:|---:|---:|"]
    for name,m,p in [("Original",baseline,None),
                     ("Neural selector",result,comparison["global"])]:
        lines.append(f"| {name} | {100*m['exact']:.4f}% | "
                     f"{100*m['poly']['exact']:.4f}% | "
                     f"{'-' if p is None else p['corrections']} | "
                     f"{'-' if p is None else p['regressions']} | "
                     f"{'-' if p is None else p['net']} |")
    lines+=["","## Per-K corrections and regressions","",
            "| True K | Fixes | Regressions | Net |",
            "|---|---:|---:|---:|"]
    for k in range(7):
        z=comparison["by_k"][str(k)]
        lines.append(f"| {k} | {z['corrections']} | {z['regressions']} | {z['net']:+d} |")
    lines+=["","## Fold-wise diagnostics","",
            "| Fold | Corrections | Regressions | Net |",
            "|---|---:|---:|---:|"]
    for f in FOLDS:
        z=folds_info[str(f)]
        lines.append(f"| {f} | {z['corrections']} | {z['regressions']} | {z['net']:+d} |")
    lines+=["","## Scientific caveats"]+["- "+t for t in report["limitations"]]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

if __name__=="__main__":main()
