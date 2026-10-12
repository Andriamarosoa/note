"""S173: label-blind S35 original trained models, NEW explicit deployment adaptation.

9 source expert models per outer fold + original S35 dynamic Q gates.
Uses S20 in place of unavailable original S29 AB and mean-of-fold models,
so it is NOT historical S35 equivalence and NOT an S80 promotion.
No model fitting, true_K, MIDI, or GOAT label access.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from scripts.s141_external_original_s40_cnn import load_input, strict
from scripts.s169_s45_ten_original_sources_risks import FAMILIES as TEN, need
from scripts.train_v273_extended_dynamic_router import (
    TRAINABLE, ACTIONS, POLY_HEADS, dynamic_route, risk_features)
from scripts.v273_extended_head_bank import bank_feature_views
from scripts.train_v273_s45_specialized_risk_heads import make_inputs,choose_best

FOLDS=(0,1,2,4)
S35="S35_multiroute"
class IdentityScaler:
    def transform(self, x):
        return x
class EnsembleGate:
    def __init__(self, states):
        self.states=states
    def predict(self, X):
        vals=[]
        for s in self.states:
            xx=np.clip(s["scaler"].transform(X),-5,5)
            vals.append(np.asarray(s["model"].predict(xx),np.float32))
        return np.mean(np.stack(vals),axis=0)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_s35(s35,views):
    """4 outer-fold pretrained expert heads, averaged only on new unseen events."""
    bank={}
    proof=[]
    for head in TRAINABLE:
        probs=[]
        for fold in FOLDS:
            path=s35/"models"/f"fold{fold}__outer__{head}.joblib"
            need(path.is_file(),"missing true original S35 outer expert "+str(path))
            obj=joblib.load(path)  # trusted pinned historic run 37868356632
            need(obj["name"]==head and int(obj["outer_fold"])==fold and
                 obj["inner_fold"] is None,"original S35 outer expert provenance")
            scaled=np.clip(obj["scaler"].transform(views[head]),-5,5)
            q=np.asarray(obj["model"].predict_proba(scaled),np.float32)
            labels=np.asarray(obj["model"].classes_,np.int64)
            expected=([1,2] if head=="E12_second_note" else
                      [2,3,4,5,6] if head in POLY_HEADS else list(range(7)))
            need(labels.tolist()==expected and q.shape==(len(scaled),len(expected)),
                 "original S35 estimator classes mismatch")
            out=np.zeros((len(scaled),7),np.float32)
            out[:,labels]=q
            need(np.isfinite(out).all() and np.allclose(out.sum(1),1,atol=1e-4),
                 "bad pretrained S35 probabilities")
            probs.append(out)
            proof.append(dict(model="expert",name=head,fold=fold,sha256=sha(path)))
        bank[head]=np.mean(probs,axis=0)
    gates={}
    gate_counts={}
    for head in ACTIONS:
        states=[]
        for fold in FOLDS:
            path=s35/"models"/f"outer{fold}__Q__{head}.joblib"
            if not path.is_file():continue
            obj=joblib.load(path)
            need(obj["head"]==head and int(obj["outer"])==fold and
                 obj["model"].n_features_in_==115 and
                 obj["scaler"].n_features_in_==115,
                 "original Q gate 115D contract")
            states.append(obj)
            proof.append(dict(model="risk_gate",name=head,fold=fold,sha256=sha(path)))
        if states:
            gates[head]=dict(model=EnsembleGate(states),scaler=IdentityScaler())
        gate_counts[head]=len(states)
    need(len(bank)==9 and len(proof)>=36 and len(gates)>0,
         "original S35 checkpoint bank incomplete")
    return bank,gates,proof,gate_counts

def run(s97,s85,s135,s169,s35,s45,catalog,out):
    need(not out.exists(),"immutable S173 native audio proof")
    ids,X,sequence,_=load_input(s97,s85,s135)
    audio=strict(s97,('row_id','summary58','flow245','geometry3','sequence42x49',
                  'morphology490','physics294','member','start_samples','end_samples'))
    need(np.array_equal(ids,audio["row_id"].astype(str)),"S35 original audio ID mismatch")
    views=bank_feature_views(X,audio["morphology490"],audio["physics294"])
    with np.load(s169,allow_pickle=False) as z:
        need(set(z.files)=={"row_id","S18_parent_K","original_ten_source_family_ids",
                "ten_source_K","ten_partial_source_agreement","original_ridge_risks",
                "ten_partial_selected_head","ten_partial_candidate_K"},
             "unaltered S169 source schema")
        need(np.array_equal(ids,z["row_id"].astype(str)),"S169 native join")
        parent=z["S18_parent_K"].astype(np.int8)
        families=list(map(str,z["original_ten_source_family_ids"]))
        ten=z["ten_source_K"].astype(np.int8)
    need(families==list(TEN) and ten.shape==(57,10),"not TEN actual prior source columns")
    # S29 original AB trained source cannot be reconstructed on new WAV.
    # S20 original-trained repeated ABA is a documented NEW adapter for AB.
    ab=ten[:,families.index("S20_repeated_ABA")]
    bank,gates,proof,gate_counts=load_s35(s35,views)
    new,routing=dynamic_route(X,parent,bank,ab,gates,1.,0.)
    need(new.shape==(57,) and np.isin(new,range(7)).all(),"S35 adapted route invalid")
    need(routing.shape==(57,3),"S35 must perform original 3-pass dynamic routing")
    combined=np.column_stack([ten,new]).astype(np.int8)
    fams=families+[S35]
    need(len(fams)==11 and len(set(fams))==11,"no duplicate S35 source")
    catalog_data=json.loads(catalog.read_text())
    index={q["family"]:q for q in catalog_data["original_families"]}
    need(len(index)==16 and index[S35]["original_variant"]=="series35__lambda1__threshold0",
         "wrong original S44 champion")
    moved=combined!=parent[:,None]
    agreement=np.zeros((57,11),np.int16)
    for j in range(11):
        agreement[:,j]=np.sum(
          (combined==combined[:,j,None])&
          (combined!=parent[:,None]),axis=1)-moved[:,j]
    # Risk recalculated across ALL eleven proposals; previous Ridge risks cannot
    # be reused because the consensus feature itself may change.
    risks=np.full((57,11,2),np.nan,np.float32)
    fitted=0
    changes_by_family={}
    for j,fam in enumerate(fams):
        paths=sorted((s45/"models").glob("*__"+fam+".joblib"))
        need(len(paths)==19,"one original S45 family fit per historical piece "+fam)
        ii=np.flatnonzero(moved[:,j])
        preds=[]
        for path in paths:
            obj=joblib.load(path)
            need(obj["family"]==fam and
                 obj["source_original_sha256"]==index[fam]["original_vector_sha256"] and
                 obj["model"].n_features_in_==obj["scaler"].n_features_in_==156,
                 "S45 original trained family risk provenance")
            if ii.size:
                inp=make_inputs(X[ii],parent[ii],combined[ii,j],agreement[ii,j])
                y=np.asarray(obj["model"].predict(np.clip(obj["scaler"].transform(inp),-6,6)),np.float32)
                need(y.shape==(len(ii),2) and np.isfinite(y).all(),"S45 two risks invalid")
                preds.append(y)
            fitted+=1
        if ii.size:risks[ii,j]=np.clip(np.mean(preds,axis=0),0,1)
        changes_by_family[fam]=int(len(ii))
    need(fitted==209 and np.isfinite(risks[moved]).all() and
         np.isnan(risks[~moved]).all(),"11 original S45×19 trained-risk contract")
    selected,head=choose_best(parent,combined,risks,1.,0.)
    out.mkdir(parents=True)
    np.savez_compressed(out/"S173_S35_adapted_11_family_GOAT.npz",
        row_id=ids,S18_parent_K=parent,original_family_ids=np.asarray(fams),
        source_K=combined,S35_new_K=new,S35_selected_actions=routing,
        agreement=agreement,trained_Ridge_risks=risks,
        partial_eleven_selected_head=head,partial_eleven_candidate_K=selected)
    report={
      "status":"S173_ORIGINAL_TRAINED_S35_WEIGHTS_EXECUTED_ADAPTED_S29_AND_11_SOURCE_RISKS",
      "S35_original_run":37868356632,
      "S35_original_champion_rule":"series35__lambda1__threshold0",
      "GOAT_native_events":57,
      "new_model_training":False,
      "original_trained_S35_expert_outer_models":sum(x["model"]=="expert" for x in proof),
      "original_trained_S35_risk_gate_models":sum(x["model"]=="risk_gate" for x in proof),
      "gate_counts":gate_counts,
      "all_original_checkpoint_sha256":proof,
      "S35_AB_S29_original_present":False,
      "S35_AB_adapter":"S20_repeated_ABA original weights; NOT original S29",
      "S35_original_new_wav_equivalent":False,
      "S35_four_fold_aggregation":"new mean of four original outer expert checkpoints and available original Q gate checkpoints, no GOAT fitting",
      "S35_number_changed_vs_S18":int((new!=parent).sum()),
      "S35_action_histogram":{str(j):int((routing==j).sum()) for j in np.unique(routing) if j>=0},
      "source_changes":changes_by_family,
      "S45_original_ridge_models_checked":fitted,
      "S45_partial_family_count":11,
      "S45_partial_changes_vs_S18":int((selected!=parent).sum()),
      "S45_partial_K_histogram":np.bincount(selected,minlength=7).tolist(),
      "S45_original_all_16_source_agreement":False,
      "S5_S18_original_new_audio_equivalent":False,
      "GOAT_MIDI_or_true_K_opened":False,
      "independent_ExactK_scored":False,
      "official_S80_promotion":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in (
      "status","original_trained_S35_expert_outer_models",
      "original_trained_S35_risk_gate_models","S35_number_changed_vs_S18",
      "S45_partial_family_count","S45_original_ridge_models_checked",
      "S45_partial_changes_vs_S18","official_S80_promotion")},indent=2))

def selftest():
    need(len(ACTIONS)==19 and len(TRAINABLE)==9 and len(TEN)==10,
         "historical S35 or S169 definitions drift")
    z=np.zeros((2,335),np.float32)
    from scripts.v273_extended_head_bank import bank_feature_views
    v=bank_feature_views(z,np.zeros((2,490),np.float32),np.zeros((2,294),np.float32))
    need(set(v)==set(TRAINABLE),"nine model source input feature mappings")
    need(risk_features(z,np.ones(2,np.int8),np.ones(2,np.int8),
                       np.ones(2,np.int8),np.ones(2),0,np.full(2,-1)).shape==(2,115),
         "original Q gate 115D")
    print("S173 PASS pretrained source 9-way and original risk 115D input contracts")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("s97","s85","s135","s169","s35","s45","catalog","out"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:selftest()
    else:
        need(all(getattr(args,n) is not None for n in
             ("s97","s85","s135","s169","s35","s45","catalog","out")),
             "need all pinned label-free inputs and original trained models")
        run(*[getattr(args,n) for n in
            ("s97","s85","s135","s169","s35","s45","catalog","out")])
