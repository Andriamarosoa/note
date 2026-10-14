"""S502: Exact-K-style heads, actions and class-conditional NOTE selections.

Mirrors learn_v273_neural_head_selector contract: source head proposals,
structural masks [event,head,K0..K6], observed audio features, action types,
fallback KEEP and audit-trained-only selection. This is a REAL adapter
bank built from S498/S500 predictions, NOT a fitted neural selector.
All specialist heads stay unpromotable until matched OOF song-held audits.
No ground truth labels or historical S80/Live/onset/offset are changed.
"""
from __future__ import annotations
from dataclasses import dataclass
import argparse,hashlib,json
from pathlib import Path
import numpy as np

K_CLASSES=7
ACTIONS=("KEEP","ADD_BIRTH","REMOVE_FALSE_NOTE","CHANGE_PITCH",
         "MERGE_DUPLICATE","REATTACK","TRANSFER_EVENT","REVISE_K","ABSTAIN")
TYPES=("baseline","predictor","correction","fix")

@dataclass(frozen=True)
class Head:
    name:str
    kind:str
    role:str
    target:str
    latency_ms:int
    action:str="REVISE_K"
    requires_oof:bool=True
    causal:bool=False
    def __post_init__(self):
        if self.kind not in TYPES or self.action not in ACTIONS or self.latency_ms<0:
            raise ValueError("Invalid head kind, action, latency")

HEADS=(
 Head("H0_BP_NEW_KEEP","baseline","keep raw BP new note set","new",0,"KEEP",False),
 Head("H1_BP_ACTIVE","predictor","sounding pitches at birth","active",30),
 Head("H2_SOURCE_KNEW","predictor","original S80/V88 soft K_new","new",30),
 Head("H3_SOURCE_KACTIVE","predictor","original S80/V88 soft K_active","active",30),
 Head("H4_NEW_CONFIDENCE","correction","actual top ranked pitches by new K","new",30),
 Head("H5_NEW_HARMONICS","correction","harmonic discounted pitches by new K","new",30),
 Head("H6_ACTIVE_CONFIDENCE","correction","ranked sounding pitches","active",30),
 Head("H7_ACTIVE_HARMONICS","correction","discounted sounding pitches","active",30),
 Head("H8_NO_DUPLICATE","fix","native aliases share one musical birth","new",80,"MERGE_DUPLICATE"),
 Head("H9_REATTACK","predictor","repeated pitch can have genuine new onset","new",80,"REATTACK"),
 Head("H10_PROTECT_POLY","fix","do not delete strongly supported poly notes","new",30,"KEEP"),
 Head("H11_K_DISAGREEMENT","predictor","Knew and Kactive mismatch","new",30),
 Head("H12_DOMAIN_RISK","fix","weak evidence or unknown source","new",80,"ABSTAIN"),
)
assert len(HEADS)==13

def need(c,msg):
    if not c:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def noteset(x):
    z=tuple(sorted(set(map(int,x))))
    need(all(0<=p<=127 for p in z),"S502 proposed pitch outside MIDI range")
    return z
def make_proposals(c,n):
    born=noteset(c["bp_note_pitches"])
    active=noteset(c["bp_active_pitches_at_birth_plus30"])
    proposals=[
        born,active,born,active,
        noteset(n["joint_new_confidence_pitch_MIDI"]),
        noteset(n["joint_new_harmonic_discount_pitch_MIDI"]),
        noteset(n["joint_active_confidence_pitch_MIDI"]),
        noteset(n["joint_active_harmonic_discount_pitch_MIDI"]),
        born,born,born,born,born,
    ]
    observed=set(born)|set(active)
    need(all(set(z)<=observed for z in proposals),
         "S502 head cannot invent a pitch not observed acoustically")
    return proposals
def structural_eligible(j,c):
    n=int(c["K_new"])
    k_act=int(c["K_active"])
    source=c["original_source_nearest_positive_K_prior"]
    confidence=float(c["lowest_independent_note_score"])
    alias=int(c["source_alias_candidate_count"])
    if j in (2,3):return source is not None
    if j==8:return alias>0
    if j==9:return bool(c["reattack_during_existing_same_pitch_sustain"])
    if j==10:return n>=2 and confidence>=.5
    if j==11:return n!=k_act or (source is not None and int(source)!=n)
    if j==12:return confidence<.65 or source is None
    return True
def make_head_arrays(c,n):
    proposals=make_proposals(c,n)
    H=len(HEADS)
    mask=np.zeros(H,bool)
    Kmask=np.zeros((H,7),bool)
    types=np.zeros((H,len(TYPES)),np.float32)
    context=np.zeros((H,16),np.float32)
    cards=np.zeros(H,np.int8)
    provenance=[]
    nnew=int(c["K_new"]);nactive=int(c["K_active"])
    source=c["original_source_nearest_positive_K_prior"]
    alias=int(c["source_alias_candidate_count"])
    for j,head in enumerate(HEADS):
        pitch=proposals[j]
        eligible=bool(structural_eligible(j,c))
        if not pitch and j!=0:eligible=False
        mask[j]=eligible
        types[j,TYPES.index(head.kind)]=1.
        cards[j]=min(6,len(pitch))
        if j==0:
            Kmask[j]=True
        elif eligible and 1<=len(pitch)<=6:
            Kmask[j,len(pitch)]=True
        # Edit-veto and event-ownership heads can condition ANY possible K.
        if eligible and head.action in ("KEEP","ABSTAIN","MERGE_DUPLICATE","REATTACK"):
            Kmask[j]=True
        context[j]=np.asarray([
           min(len(pitch),6)/6,min(nnew,6)/6,min(nactive,6)/6,
           0 if source is None else int(source)/6,float(source is not None),
           float(c["lowest_independent_note_score"]),
           min(alias,8)/8,
           min(int(c["source_original_native_candidates_within80ms"]),16)/16,
           min(int(c["source_original_native_positive_candidates_within80ms"]),8)/8,
           float(c["reattack_during_existing_same_pitch_sustain"]),
           float(head.kind=="fix"),float(head.kind=="correction"),
           float(head.target=="active"),float(nnew!=nactive),
           float(len(pitch)==nnew),float(len(pitch)==nactive)
        ],np.float32)
        provenance.append(dict(head=head.name,kind=head.kind,action=head.action,
           role=head.role,target=head.target,note_pitches=list(pitch),
           note_count=len(pitch),possible_K=np.flatnonzero(Kmask[j]).tolist(),
           structural_available=eligible,requires_OOF_training=head.requires_oof,
           live_causality_proven=head.causal,latency_budget_ms=head.latency_ms))
    need(mask[0] and Kmask[0].all() and np.all(~Kmask[~mask]),
         "S502 model fallback/structural mask contract broken")
    need(np.isfinite(context).all(),"S502 invalid acoustic-only features")
    # Do not turn a register of heads into untrained live model decisions.
    production=mask&np.asarray([not h.requires_oof for h in HEADS])
    need(production.tolist()==[True]+[False]*(H-1),
         "S502 untrained head used to edit source predictions")
    return dict(head_mask=mask,head_k_mask=Kmask,head_type=types,
         context=context,proposed_K=cards,production_eligible=production,
         ledger=provenance)
def run(graph,graph_report,names,names_report,out):
    need(not out.exists(),"S502 preserve immutable prelabel head records")
    g=json.loads(graph_report.read_text());n=json.loads(names_report.read_text())
    need(g["status"]=="S498_TWO_SOURCE_CANONICAL_NOTE_ONSET_GRAPH_SOURCE_K_AS_SOFT_PRIOR_PRELABEL"
         and g["no_original_MIDI_XML_annotations_read_in_canonical_model"]
         and n["status"]=="S500_TWO_SOURCE_JOINT_EXACTK_CONSTRAINED_REAL_NOTE_NAMES_PRELABEL"
         and n["no_training_no_XML_MIDI_label_access"],
         "S502 requires genuine original pre-XML note-pitch evidence")
    out.mkdir(parents=True);summary={}
    for corpus in ("guitar_S80","piano_V88"):
        gf=graph/f"S498_{corpus}_CANONICAL_MUSICAL_EVENTS_PRELABEL.jsonl"
        nf=names/f"S500_{corpus}_K_CONSTRAINED_ACTUAL_NOTE_NAMES_PRELABEL.jsonl"
        need(g["frozen_group_sha256"][corpus]==sha(gf)
             and n["frozen_output_shas"][corpus]==sha(nf),
             "S502 original canonical event and pitch prediction fingerprint changed")
        canonical=[json.loads(x) for x in gf.read_text().splitlines()]
        ranked=[json.loads(x) for x in nf.read_text().splitlines()]
        lookup={x["canonical_event_id"]:x for x in ranked}
        need(len(canonical)==len(ranked)==len(lookup),
             "S502 canonical birth event identity not unique")
        data=[make_head_arrays(c,lookup[c["canonical_event_id"]]) for c in canonical]
        output_json=out/f"S502_{corpus}_HEAD_SELECTIONS_PRELABEL.jsonl"
        output_json.write_text("".join(json.dumps(dict(
            canonical_event_id=c["canonical_event_id"],
            source=corpus,head_selections=z["ledger"],
            fitted_neural_selector=False,all_specialist_heads_blocked_until_OOF=True))
            +"\n" for c,z in zip(canonical,data)))
        out_npz=out/f"S502_{corpus}_HEAD_K_MASKS_PRELABEL.npz"
        np.savez_compressed(out_npz,
             canonical_event_id=np.array([x["canonical_event_id"] for x in canonical]),
             head_mask=np.stack([x["head_mask"] for x in data]),
             head_k_mask=np.stack([x["head_k_mask"] for x in data]),
             head_type=np.stack([x["head_type"] for x in data]),
             context=np.stack([x["context"] for x in data]),
             head_proposal_K=np.stack([x["proposed_K"] for x in data]),
             production_eligible=np.stack([x["production_eligible"] for x in data]))
        available=np.stack([x["head_mask"] for x in data])
        summary[corpus]=dict(canonical_events=len(data),
            heads=len(HEADS),structurally_available={
                h.name:int(available[:,j].sum()) for j,h in enumerate(HEADS)},
            pretrained_specialist_heads_promoted=0,
            original_ExactK_and_note_predictions_modified=0,
            ledger_SHA256=sha(output_json),matrix_SHA256=sha(out_npz))
        print("S502 REAL HEAD BANK",json.dumps(dict(source=corpus,**summary[corpus])),flush=True)
    (out/"report.json").write_text(json.dumps(dict(
        status="S502_CLASS_CONDITIONAL_NOTE_HEADS_SELECTIONS_ARCHITECTURE_PRELABEL",
        original_complete_S80_guitar_and_original_V88_piano_distinct=True,
        same_principle_as_neural_ExactK_selector=True,
        conditional_masks_shape="events X 13 heads X K0toK6",
        action_types=list(ACTIONS),
        number_of_registered_heads=len(HEADS),
        only_fallback_head_currently_eligible_for_production=True,
        specialist_OOF_crossfit_not_yet_available=True,
        no_NN_training_or_external_ground_truth_used=True,
        no_claim_of_gain_from_untrained_selections=True,
        original_S80_Live_onset_offsets_unchanged=True,
        no_promotion=True,by_source=summary),indent=2)+"\n")
def selftest():
    c=dict(bp_note_pitches=[52,55],bp_active_pitches_at_birth_plus30=[52,55,59],
       K_new=2,K_active=3,source_alias_candidate_count=2,
       source_original_native_candidates_within80ms=3,
       source_original_native_positive_candidates_within80ms=1,
       reattack_during_existing_same_pitch_sustain=False,
       lowest_independent_note_score=.91,original_source_nearest_positive_K_prior=3)
    n=dict(joint_new_confidence_pitch_MIDI=[52,55],
      joint_new_harmonic_discount_pitch_MIDI=[52,55],
      joint_active_confidence_pitch_MIDI=[52,55,59],
      joint_active_harmonic_discount_pitch_MIDI=[52,55,59])
    z=make_head_arrays(c,n)
    need(z["head_mask"][0] and z["head_k_mask"][0].all()
      and z["head_mask"][8] and not z["head_mask"][9]
      and z["head_mask"][10] and z["head_mask"][11],
      "S502 conditional note-head selection masks wrong")
    need(z["head_k_mask"][4].tolist()==[False,False,True,False,False,False,False]
      and z["head_k_mask"][6].tolist()==[False,False,False,True,False,False,False],
      "S502 notes are not conditional by target K")
    need(int(z["production_eligible"].sum())==1,
      "S502 unfitted neural specialists not allowed to go live")
    try:
        make_head_arrays(c,n|dict(joint_new_confidence_pitch_MIDI=[52,55,60]))
        raise AssertionError("S502 permitted an invented pitch")
    except ValueError:pass
    print("S502 SELFTEST PASS 13 heads, K masks, KEEP and no invented pitches")
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--self-test",action="store_true")
    for k in ("graph","graph-report","names","names-report","out"):
        ap.add_argument("--"+k,type=Path)
    z=ap.parse_args()
    if z.self_test:selftest()
    else:
        for k in ("graph","graph_report","names","names_report","out"):
            need(getattr(z,k)is not None,"S502 missing --"+k)
        run(z.graph,z.graph_report,z.names,z.names_report,z.out)
