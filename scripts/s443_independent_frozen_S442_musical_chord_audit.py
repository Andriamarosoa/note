"""S443 -- independent replay score of S442 frozen note names (NO model fitting)."""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import numpy as np
from scripts.s402_k_input_first_supervised_midi_pitch_network import load,labels,need
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha

def count_notes(row):
    return collections.Counter(int(p) for p in row if p>=40)

def measure(base,pred,target,ix):
    before=after=wins=losses=btp=atp=tct=pc=0
    for i in ix:
        b=count_notes(base[i]);a=count_notes(pred[i]);t=count_notes(target[i])
        bb=b==t;aa=a==t
        before+=int(bb);after+=int(aa)
        wins+=int(aa and not bb);losses+=int(bb and not aa)
        btp+=sum((b&t).values());atp+=sum((a&t).values())
        tct+=sum(t.values());pc+=sum(a.values())
    return dict(events=len(ix),baseline_exact_chords=int(before),
       candidate_exact_chords=int(after),corrections=int(wins),
       regressions=int(losses),net=int(wins-losses),
       baseline_note_tp=int(btp),candidate_note_tp=int(atp),
       delta_note_tp=int(atp-btp),note_f1_candidate=float(
           2*atp/(tct+pc) if tct+pc else 1.))

def run(s442dir,s428,source,s80,truth,annotations,out):
    need(not out.exists(),"S443 cannot overwrite another audit")
    report=json.loads((s442dir/"report.json").read_text())
    need(report["status"]=="S442_LEARNED_JOINT_NOTE_SET_SONGHELD_K_LOCKED_DEV"
         and report["no_promotion"]
         and report["composition_disjoint_S442_folds"]
         and report["inner_gates_only_other_compositions"],
         "S443 untrusted S442 model gate provenance")
    path=s442dir/"S442_FROZEN_NOTE_NAMES_KLOCKED_59309.npz"
    notes=s442dir/"S442_FROZEN_NOTE_NAMES_KLOCKED_59309.jsonl"
    need(sha(path)==report["output_npz_sha256"]
         and sha(notes)==report["output_names_sha256"],
         "S443 model results are not original frozen S442 outputs")
    s=load(source);k=load(s80);ref=load(s428);held=load(path);truthK=load(truth)
    ids=s["row_id"].astype(str)
    need(len(ids)==59309 and len(set(ids))==len(ids)
         and all(np.array_equal(ids,z["row_id"].astype(str))
                 for z in (k,ref,held,truthK)),
         "S443 unaligned source event IDs")
    K=k["s80_K"].astype(int)
    need(np.array_equal(ref["s80_K"],K)
         and np.array_equal(held["s80_K"],K)
         and np.array_equal((ref["midi_pitch_slots"]>=0).sum(1),K)
         and np.array_equal((held["midi_pitch_slots"]>=0).sum(1),K),
         "S443 original S80 K changed under note decoder")
    base=ref["midi_pitch_slots"]
    cand=held["midi_pitch_slots"]
    need(np.array_equal(cand[~np.isin(K,(2,3,4))],
                        base[~np.isin(K,(2,3,4))]),
         "S443 candidate changed protected K0/K1/K5/K6")
    _,target,details=labels(s,annotations,truthK)
    true_K=truthK["true_K"].astype(int)
    poly=np.flatnonzero(true_K>=2)
    total=measure(base,cand,target,poly)
    need(total["baseline_exact_chords"]==3025
         and total["events"]==7385,
         "S443 historic S428 reference not independently reproduced")
    need(total["candidate_exact_chords"]==
         report["development_poly_comparison"]["net"]+3025
         and total["corrections"]==report["development_poly_comparison"]["corrections"]
         and total["regressions"]==report["development_poly_comparison"]["regressions"],
         "S443 independent replay disagrees with S442 reported paired results")
    by_true={str(i):measure(base,cand,target,np.flatnonzero(true_K==i))
             for i in range(7)}
    by_pred={str(i):measure(base,cand,target,np.flatnonzero(K==i))
             for i in range(7)}
    shifted=int(sum(
        count_notes(a)!=count_notes(b) for a,b in zip(base,cand)))
    result=dict(status="S443_INDEPENDENT_FROZEN_S442_S428_PAIRED_NOTE_REPLAY",
        original_S80_K_unchanged=True,n=59309,
        original_S80_exactK_global=53529,original_S80_exactK_poly=6054,
        S428_poly_chords=3025,S442_poly_chords=total["candidate_exact_chords"],
        delta_poly_chords=total["net"],
        modified_note_sets=shifted,
        by_true_K=by_true,by_predicted_K=by_pred,
        poly_K2plus=total,
        S442_metrics_independently_reproduced=True,
        training_weights_not_refit=True,
        source_S80_historically_exposed_GuitarSet=True,
        no_fully_nested_stack_validation=True,
        independent_outside_GuitarSet_validation=False,
        no_promotion=True,
        original_S442_output_sha256=sha(path),
        annotation_md5_checked_by_workflow=True,
        source_annotation_diagnostics=details)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S443 INDEPENDENT",json.dumps(
        dict(poly=total,modified_sets=shifted,per_true_K=by_true)),flush=True)

def selftest():
    base=np.array([[52,55,-1],[52,59,-1]],np.int16)
    cand=np.array([[52,59,-1],[52,59,-1]],np.int16)
    true=np.array([[52,59,-1],[52,55,-1]],np.int16)
    d=measure(base,cand,true,[0,1])
    assert d["corrections"]==1 and d["regressions"]==0 and d["net"]==1
    assert d["delta_note_tp"]==1
    print("S443 SELFTEST PASS independent chord set audit")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    for item in ("s442dir","s428","source","s80","truth","annotations","out"):
        ap.add_argument("--"+item,type=Path)
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        for item in ("s442dir","s428","source","s80","truth","annotations","out"):
            need(getattr(a,item)is not None,"S443 missing --"+item)
        run(a.s442dir,a.s428,a.source,a.s80,a.truth,a.annotations,a.out)
