"""One-shot outer evaluation of the internally selected candidate_hidden1 group.

Selection source:
  V27.3 candidate_hidden1 exact internal group audit, run 37285354716.
Selected group:
  candidate_hidden1 output neurons [42, 52, 61, 64].

Selection used only internal folds 0,1,2,4. This script performs no training,
no outer search, and no post-hoc group selection.

Starting checkpoint:
  freeze_local_combo from run 37233793945.
Donor:
  matched uniform anchor from run 37208199201.
Intervention:
  restore the four selected candidate_hidden1 kernel columns plus matching
  bias entries from uniform into freeze_local_combo.
Evaluation:
  frozen outer fold 3 only.

No automatic promotion.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

from scripts import train_v260_count_weighting as v260
from scripts.train_v273_group_gate_ab import build_model,metrics,transitions
from scripts.train_v273_loss_weighting_ab import batches
from scripts.v273_window_experiment import load_bundle,require

SEED=v260.SEED+1003
LAYER="candidate_hidden1"
NEURONS=(42,52,61,64)
SELECTION_RUN=37285354716

def nested_base(model):
    import tensorflow as tf
    xs=[x for x in model.layers if isinstance(x,tf.keras.Model)]
    require(len(xs)==1,f"nested base mismatch: {[x.name for x in xs]}")
    return xs[0]

def predict(model,cache,outer):
    p=np.asarray(model.predict(batches(cache,outer),workers=0,max_queue_size=1,verbose=0),np.float32)
    require(p.shape==(len(outer),7) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5),
            "bad probabilities")
    return p,p.argmax(1).astype(np.int32)

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:np.asarray(z[k]) for k in z.files}

def load_cluster_a(path,global_index):
    rows=list(csv.DictReader(open(path,newline="")))
    by={}
    for r in rows:
        c=int(r["cluster"]);y=int(r["true_k"]);p=int(r["predicted_k"])
        d=by.setdefault(c,{"n":0,"under":0})
        d["n"]+=1;d["under"]+=p<y
    cid=max(by,key=lambda c:(by[c]["under"]/by[c]["n"],by[c]["n"]))
    ids={int(r["global_index"]) for r in rows if int(r["cluster"])==cid}
    return int(cid),np.isin(np.asarray(global_index,np.int64),list(ids))

def class_net(y,a,b):
    return {str(k):int(np.sum((b==y)&(y==k))-np.sum((a==y)&(y==k))) for k in range(7)}

def transition_hist(y,a,b,mask):
    from collections import Counter
    c=Counter(f"{int(x)}->{int(z)}" for x,z in zip(np.asarray(a)[mask],np.asarray(b)[mask]))
    return dict(sorted(c.items(),key=lambda kv:(-kv[1],kv[0])))

def main():
    ap=argparse.ArgumentParser()
    for n in ("bundle","config","uniform-weights","freeze-weights",
              "uniform-predictions","freeze-predictions","cluster-rows",
              "selection-report","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args()
    require(not a.output.exists(),"refusing overwrite")
    a.output.mkdir(parents=True)

    # Freeze the internal selection before touching outer predictions.
    sel=json.loads(a.selection_report.read_text())
    require(sel.get("outer_fold_3_used") is False,"selection report used outer fold")
    chosen=sel.get("selected_group")
    require(chosen is not None,"no selected internal group")
    require(tuple(chosen.get("neurons",[]))==NEURONS,
            f"selection mismatch: expected {NEURONS}, got {chosen.get('neurons')}")
    require(chosen.get("strict_robust") is True,"selected group not strict robust")

    cache,parts,_=load_bundle(a.bundle,a.config)
    outer=np.asarray(parts["outer"],np.int64)
    inner=np.asarray(parts["final_fit"],np.int64)
    require(len(outer)==15279 and not np.intersect1d(outer,inner).size,"partition drift")
    y=np.minimum(cache["exact"][outer].astype(np.int32),6)

    up=load_npz(a.uniform_predictions)
    fp=load_npz(a.freeze_predictions)
    np.testing.assert_array_equal(up["global_index"],outer)
    np.testing.assert_array_equal(fp["global_index"],outer)
    np.testing.assert_array_equal(up["k"],y)
    np.testing.assert_array_equal(fp["k"],y)
    U=np.asarray(up["predicted"],np.int32)
    F=np.asarray(fp["freeze_local_combo_predicted"],np.int32)

    mu=build_model("learned_gate",SEED);mu.load_weights(a.uniform_weights)
    mf=build_model("learned_gate",SEED);mf.load_weights(a.freeze_weights)
    bu=nested_base(mu);bf=nested_base(mf)

    wu=[np.asarray(x).copy() for x in bu.get_layer(LAYER).get_weights()]
    wf=[np.asarray(x).copy() for x in bf.get_layer(LAYER).get_weights()]
    require(len(wu)==2 and len(wf)==2,"expected Dense kernel+bias")
    ku,bu_bias=wu;kf,bf_bias=wf
    require(ku.shape==kf.shape and bu_bias.shape==bf_bias.shape and ku.shape[1]==96,
            "candidate_hidden1 shape mismatch")

    # Verify exact replay of the published freeze reference.
    _,F_replay=predict(mf,cache,outer)
    np.testing.assert_array_equal(F_replay,F)

    # Fixed, internally selected intervention.
    kf2=kf.copy(); bf2=bf_bias.copy()
    ids=np.asarray(NEURONS,np.int64)
    kf2[:,ids]=ku[:,ids]
    bf2[ids]=bu_bias[ids]
    bf.get_layer(LAYER).set_weights([kf2,bf2])

    P,T=predict(mf,cache,outer)

    cid,A=load_cluster_a(a.cluster_rows,outer)
    require(int(A.sum())==499,"Cluster A drift")

    um=metrics(y,U);fm=metrics(y,F);tm=metrics(y,T)
    reg=(F==y)&(T!=y)
    cor=(F!=y)&(T==y)

    report={
      "status":"completed",
      "training":False,
      "selection":{
        "source_internal_run":SELECTION_RUN,
        "selected_neurons":list(NEURONS),
        "strict_robust_internal":True,
        "selection_used_outer":False,
        "internal_total_global_net":int(chosen["total_global_net"]),
        "internal_total_low_net":int(chosen["total_low_net"]),
        "internal_total_poly_net":int(chosen["total_poly_net"]),
        "internal_min_global_net":int(chosen["min_global_net"]),
        "internal_min_low_net":int(chosen["min_low_net"]),
        "internal_min_poly_net":int(chosen["min_poly_net"])
      },
      "protocol":{
        "experiment":"v273_outer_eval_inner_selected_candidate_hidden1_group_42_52_61_64",
        "outer_fold":3,
        "outer_evaluation_only":True,
        "layer":LAYER,
        "neurons":list(NEURONS),
        "swap_unit":"kernel output columns + matching bias entries",
        "automatic_promotion":False
      },
      "metrics":{
        "uniform":um,
        "freeze_local_combo":fm,
        "selected_group_transplant":tm,
        "cluster_A":{
          "freeze_local_combo":metrics(y[A],F[A]),
          "selected_group_transplant":metrics(y[A],T[A])
        }
      },
      "paired":{
        "group_vs_uniform":transitions(y,U,T),
        "group_vs_freeze_local_combo":transitions(y,F,T),
        "class_net_vs_uniform":class_net(y,U,T),
        "class_net_vs_freeze_local_combo":class_net(y,F,T),
        "regressions_vs_freeze":int(reg.sum()),
        "corrections_vs_freeze":int(cor.sum()),
        "regression_true_k":{str(k):int(np.sum(reg&(y==k))) for k in range(7)},
        "correction_true_k":{str(k):int(np.sum(cor&(y==k))) for k in range(7)},
        "regression_transitions":transition_hist(y,F,T,reg),
        "correction_transitions":transition_hist(y,F,T,cor)
      }
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    np.savez_compressed(a.output/"predictions.npz",
      global_index=outer,k=y,probability=P,predicted=T,
      uniform_predicted=U,freeze_local_combo_predicted=F)

    lines=[
      "# Outer evaluation — internally selected candidate_hidden1 group [42, 52, 61, 64]","",
      "The group was selected only from internal folds 0/1/2/4. No outer selection or training occurred.","",
      "| Measure | Uniform | freeze_local_combo | + group transplant |",
      "|---|---:|---:|---:|"
    ]
    lines += [
      f"| exact global | {100*um['exact']:.3f}% | {100*fm['exact']:.3f}% | {100*tm['exact']:.3f}% |",
      f"| exact poly | {100*um['poly_exact']:.3f}% | {100*fm['poly_exact']:.3f}% | {100*tm['poly_exact']:.3f}% |"
    ]
    for k in range(6):
        lines.append(
          f"| K{k} exact | {100*um['by_k'][str(k)]['exact']:.2f}% | "
          f"{100*fm['by_k'][str(k)]['exact']:.2f}% | {100*tm['by_k'][str(k)]['exact']:.2f}% |"
        )
    lines += [
      f"| Cluster A exact | — | {100*report['metrics']['cluster_A']['freeze_local_combo']['exact']:.2f}% | "
      f"{100*report['metrics']['cluster_A']['selected_group_transplant']['exact']:.2f}% |","",
      f"Paired vs uniform net: **{report['paired']['group_vs_uniform']['net_correct']:+d}**.",
      f"Paired vs freeze_local_combo net: **{report['paired']['group_vs_freeze_local_combo']['net_correct']:+d}**.",
      f"Corrections/regressions vs freeze_local_combo: **{int(cor.sum())}/{int(reg.sum())}**.","",
      "No automatic promotion."
    ]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
