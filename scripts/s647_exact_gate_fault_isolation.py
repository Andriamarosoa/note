"""S647: exactly locate S642 high-q incompatibility with NOTE original gates.

For every original TRAIN and DEV audio, retain the S614 native
time-causal active-note states, q score, and training annotations.
All reports are diagnostics: no threshold choice, no model training,
no altered reference, and no use of original TEST.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import json
import tempfile
import zipfile
import numpy as np
import joblib
from causal_note.strict_live import Selection,CausalSelection,load_heads
from causal_note.s642_dual_learned import candidate_scores
from scripts.s644_gate_rescue_loop import CANDIDATE_SHA,S614_SHA
from scripts.s626_train_sharp_native_onset import get_source,sharper_positive
from scripts.s623_mixed_source_event_head import pick_original_guitar_training
from scripts.s616_live_dev_ablation import ROOT,MODEL
from scripts.s613_prepare_strict_live import download,digest

def diagnose(row,q,config):
    n=len(q)
    past=np.zeros_like(q)
    # Recent score minimum from the six strictly preceding hops. Missing
    # pre-recording hops are silent q=0, matching S645 runtime.
    delayed=[]
    for k in range(1,7):
        v=np.zeros_like(q)
        v[k:]=q[:-k]
        delayed.append(v)
    past=np.minimum.reduce(delayed)
    edge=(q>=.99)&((q-past)>=.10)
    p=row["p"]
    onset=p[:,:,0]>=.02
    sustain=p[:,:,1]>=.05
    physical=onset&sustain
    sel=CausalSelection(config)
    idle=np.zeros((n,49),np.bool_)
    for i,(end,pframe,k) in enumerate(zip(row["ends"],p,row["k"])):
        idle[i]=~sel.active
        sel.step(pframe,k,int(end),int(end))
    y=sharper_positive(row["ends"],row["truth"])
    any_correct= y & edge
    check={
      "rising_q":edge,
      "rising_q_near_true_attack":any_correct,
      "rising_q_near_no_truth":edge&~y,
      "rising_q_inactive":edge&idle,
      "rising_q_onset_above_002":edge&onset,
      "rising_q_sustain_above_005":edge&sustain,
      "rising_q_both_acoustic":edge&physical,
      "rising_q_inactive_and_onset":edge&idle&onset,
      "rising_q_inactive_and_sustain":edge&idle&sustain,
      "rising_q_inactive_and_both":edge&idle&physical,
      "rising_q_active_and_both":edge&~idle&physical,
      "rising_q_active":edge&~idle,
      "rising_q_inactive_low_onset":edge&idle&~onset,
      "rising_q_inactive_low_sustain":edge&idle&~sustain,
      "near_truth_attack_frames":y,
      "near_truth_q_ge_099":y&(q>=.99),
    }
    out={key:int(mask.sum()) for key,mask in check.items()}
    assert out["rising_q"]==out["rising_q_inactive"]+out["rising_q_active"]
    assert out["rising_q_near_true_attack"]+out["rising_q_near_no_truth"]==out["rising_q"]
    assert out["rising_q_inactive_and_both"]<=out["rising_q_both_acoustic"]
    return out

def main(out,checkpoint):
    if out.exists():raise FileExistsError("S647 immutable audit path")
    out.mkdir(parents=True)
    if digest(checkpoint.read_bytes())!=CANDIDATE_SHA:raise ValueError("Candidate checksum changed")
    model=joblib.load(checkpoint)
    protocol=json.loads((ROOT/"protocol.json").read_text())
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    if digest(MODEL.read_bytes())!=S614_SHA or digest((ROOT/"protocol.json").read_bytes())!=frozen["protocol_sha256"]:
        raise ValueError("S614 original / protocol changed")
    native=load_heads(MODEL,expected_sha256=S614_SHA)
    cfg=Selection(**native.metadata["selection"])
    train=[s for s in protocol["sources"] if s["split"]=="train" and s["corpus"]=="URMP"]
    train+=pick_original_guitar_training(protocol)
    dev=[s for s in protocol["sources"] if s["split"]=="dev" and s["corpus"] in ("URMP","GuitarSet")]
    assert (len(train),len(dev))==(35,17)
    results={}
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for key in ("GuitarSet_audio","GuitarSet_annotations"):
            z=protocol["guitar_archives"][key]
            download(z["url"],root/"data"/(key+".zip"),z["size"],z["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            for stage,sources in (("TRAIN",train),("DEV",dev)):
                found=[]
                for s in sources:
                    row=get_source(s,root,native,archive if s["corpus"]=="GuitarSet" else None)
                    q=candidate_scores(row,model)
                    stages=diagnose(row,q,cfg)
                    found.append(dict(name=s["name"],corpus=s["corpus"],mechanisms=stages))
                    print("S647",stage,s["name"],"rising",stages["rising_q"],
                        "active",stages["rising_q_active"],
                        "both_physical",stages["rising_q_both_acoustic"],
                        "truth",stages["rising_q_near_true_attack"],flush=True)
                corpus_totals={}
                for corpus in ("GuitarSet","URMP"):
                    this=[x["mechanisms"] for x in found if x["corpus"]==corpus]
                    corpus_totals[corpus]={key:int(sum(row[key] for row in this))
                                          for key in found[0]["mechanisms"]}
                results[stage]=dict(n_sources=len(found),by_corpus=corpus_totals,by_source=found)
    report=dict(status="S647_AUDITED_TRUE_EXCLUSIVE_RISE_GATE_BLOCKERS",
       checkpoint_sha256=CANDIDATE_SHA,S614_sha256=S614_SHA,
       original_models_unchanged=True,annotation_used_only_diagnostically=True,
       no_new_weights=True,no_test_sources=True,no_promotion=True,
       results=results)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S647 CAUSAL INTERSECTIONS",json.dumps(
       {key:r["by_corpus"] for key,r in results.items()},indent=2),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    o=p.parse_args();main(o.out,o.candidate)
