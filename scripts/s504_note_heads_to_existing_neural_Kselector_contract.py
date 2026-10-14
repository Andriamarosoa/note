"""S504 — exact numeric adapter from real S502 note heads into the ORIGINAL
K0..K6 + KEEP ClassConditionalAuditSelector input contract.

Source model: scripts/learn_v273_neural_head_selector.py. The same learned
attention architecture can be used for note-head ROUTING, but a separate
note-pitch-set decoder/action execution is required. This adapter never
fits a neural model, does not manufacture OOF audits, and refuses to mark
specialists active until training-only composition-held audits exist.
It is not a deployed selector or evidence of generalization.

TensorFlow is not needed to verify interface shapes; model training and
full note-action decoding will come from a distinct held-song pipeline.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s502_note_birth_head_selection_adapters import HEADS,need

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def convert(src):
    with np.load(src,allow_pickle=False) as z:
        rowid=z["canonical_event_id"].astype(str)
        hm=z["head_mask"].astype(bool)
        km=z["head_k_mask"].astype(bool)
        typ=z["head_type"].astype(np.float32)
        feats=z["context"].astype(np.float32)
        proposal=z["head_proposal_K"].astype(np.int8)
        promote=z["production_eligible"].astype(bool)
    N,H=proposal.shape
    need(H==len(HEADS)==13 and
         hm.shape==(N,H) and km.shape==(N,H,7) and
         typ.shape==(N,H,4) and feats.shape==(N,H,16)
         and promote.shape==(N,H),
         "S504 S502 class conditional NN input tensor ABI changed")
    # Original ClassConditionalAuditSelector type classes:
    # (feature,predictor,correction,fix) rather than our adapter
    # (baseline,predictor,correction,fix); both type 0 mean fallback features.
    types=typ.copy()
    logits=np.full((N,H,8),-8.,np.float32)
    for j,h in enumerate(HEADS):
        if h.action in ("KEEP","MERGE_DUPLICATE","REATTACK","ABSTAIN"):
            logits[:,j,7]=8.  # KEEP counts are not implicit pitch edit.
        else:
            ix=np.arange(N)
            logits[ix,j,np.clip(proposal[:,j],0,6)]=8.
    # Source fallback BP-born note count; not final historical S80 K,
    # because the canonical source does not contain a unique S80 row.
    base=np.eye(7,dtype=np.float32)[np.clip(proposal[:,0],0,6)]
    # Inputs to the historical neural selector:
    # head_logits [N,H,8]; head_type [N,H,4]; audit [N,H,21];
    # context [N,F]; baseline [N,7]; head_mask [N,H];
    # head_k_mask [N,H,7].
    # Training-only OOF audit profiles are NOT available on original IDMT
    # 5 guitar and MAESTRO 3 piano test excerpts. Do not fabricate them as
    # zero and then FIT an apparently valid NN: leave training false.
    context=feats[:,0,:]
    need(np.isfinite(context).all() and
         np.all(promote[:,0]) and not np.any(promote[:,1:]),
         "S504 untrained routing would leak into official note corrections")
    need(km[:,0,:].all() and np.all(~km[~hm]) and
         logits.shape==(N,H,8),"S504 original neural selector masks wrong")
    return dict(canonical_event_id=rowid,head_logits=logits,head_type=types,
       context=context,baseline=base,head_mask=hm,
       head_k_mask=km,training_only_audit_ready=np.zeros((N,H),bool),
       historical_fitted_selector_applied=np.zeros(N,bool),
       promotion_eligible=promote)
def run(inp,report,out):
    need(not out.exists(),"S504 freeze one original learned-selector interface")
    parent=json.loads(report.read_text())
    need(parent["status"]=="S502_CLASS_CONDITIONAL_NOTE_HEADS_SELECTIONS_ARCHITECTURE_PRELABEL"
         and parent["only_fallback_head_currently_eligible_for_production"]
         and parent["specialist_OOF_crossfit_not_yet_available"]
         and parent["no_NN_training_or_external_ground_truth_used"],
         "S504 cannot masquerade exposed external audio as OOF model training")
    out.mkdir(parents=True);summ={}
    for k in ("guitar_S80","piano_V88"):
        file=inp/f"S502_{k}_HEAD_K_MASKS_PRELABEL.npz"
        need(parent["by_source"][k]["matrix_SHA256"]==sha(file),
             "S504 original trained note head evidence source changed")
        d=convert(file)
        p=out/f"S504_{k}_NEURAL_CLASS_CONDITIONAL_INPUT_ONLY_NOT_FITTED.npz"
        np.savez_compressed(p,**d)
        summ[k]=dict(events=len(d["canonical_event_id"]),
            features=int(d["context"].shape[1]),heads=13,
            target_K_classes=7,original_action_classes=8,
            output_path=p.name,sha256=sha(p),
            no_original_nn_model_fitted=True)
        print("S504 K HEAD NN INPUT",json.dumps(dict(corpus=k,**summ[k])),flush=True)
    (out/"report.json").write_text(json.dumps(dict(
        status="S504_ORIGINAL_EXACTK_CLASSCONDITIONAL_NN_NOTE_HEAD_ADAPTER_NO_FIT",
        exact_training_architecture="ClassConditionalAuditSelector",
        compatible_with_existing_Head_K_mask_attention_contract=True,
        not_a_trained_note_classifier=True,
        absence_of_crossfit_training_audits_is_explicit_and_prevents_fitting=True,
        no_pseudo_labels_from_exposed_IDMT_or_MAESTRO=True,
        requires_new_held_song_real_note_events_for_selector_fit=True,
        original_source_S80_Live_unchanged=True,no_promotion=True,
        by_source=summ),indent=2)+"\n")
def selftest():
    # Validate real K mask compatibility [N,H,7] and no arbitrary
    # action outside supported notes; absent OOF cannot be promoted.
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        file=Path(root)/"a.npz"
        N,H=2,13
        mask=np.ones((N,H),bool)
        classes=np.zeros((N,H,7),bool);classes[:,0]=True
        types=np.zeros((N,H,4),np.float32);types[:,:,0]=1
        prop=np.ones((N,H),np.int8);prop[:,0]=2
        available=np.zeros((N,H),bool);available[:,0]=True
        np.savez_compressed(file,canonical_event_id=np.array(["A","B"]),
            head_mask=mask,head_k_mask=classes,head_type=types,
            context=np.zeros((N,H,16),np.float32),
            head_proposal_K=prop,production_eligible=available)
        data=convert(file)
        need(data["head_logits"].shape==(2,13,8)
             and data["head_k_mask"].shape==(2,13,7)
             and data["baseline"].shape==(2,7)
             and not data["training_only_audit_ready"].any(),
             "S504 original learned-class-conditional architecture contract broke")
    print("S504 SELFTEST PASS original Exact-K neural class selector input compatibility; training disabled")
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--self-test",action="store_true")
    for k in ("inp","report","out"):ap.add_argument("--"+k,type=Path)
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("inp","report","out"):need(getattr(a,k)is not None,"S504 missing "+k)
        run(a.inp,a.report,a.out)
