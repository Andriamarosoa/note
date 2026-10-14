"""S450: external IDMT original S80 acoustic events -> K-locked note names.

Five previously frozen IDMT dataset3 genuine WAV sources (506 native V88
events). S283 provides authentic 42x49 acoustic trajectories; S297 supplies
the *actual original* S80 predicted K0..6. No estimated peak K or oracle K.
S430 predicts the ORIGINAL four-head external baseline; frozen S446 models
may edit a note ONLY when two distinct fold heads vote for the SAME
(old_note,new_note) change at their frozen per-predicted-K thresholds.
No XML/MIDI/truth or training may enter this file. This is an external
domain transfer experiment, NOT a new blind-label population.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import joblib,numpy as np,torch
from scripts.s430_infer_note_names_from_new_S80_sound_events import run as original_infer
from scripts.s446_per_predicted_k_joint_note_songheld import proposals
from scripts.s416_native_Kfirst_temporal_MIDI_note_decoder import NativeKTimePitchNet
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import note_name,sha
from scripts.s402_k_input_first_supervised_midi_pitch_network import need

N=506
FOLDS=4
AGREEMENT=2

def consensus(votes, old):
    """Require matching interventions from >=2 different trained folds."""
    counts=collections.Counter(pair for fold,pair in votes)
    eligible=[(pair,n) for pair,n in counts.items() if n>=AGREEMENT
              and pair[0] in old and pair[1] not in old]
    if not eligible:return old.copy(),None
    chosen=sorted(eligible,key=lambda z:(-z[1],z[0][0],z[0][1]))[0]
    after=old.copy()
    after[after.index(chosen[0][0])]=chosen[0][1]
    need(len(after)==len(old) and len(set(after))==len(old),
         "S450 swap violated original S80 cardinality")
    return after,dict(removed=chosen[0][0],added=chosen[0][1],votes=chosen[1])

def run(source,source_report,s80,s80_report,
        s402,s416,s428,s419_report,s428_report,
        s446,s446_report,out):
    need(not out.exists(),"S450 prospective original IDMT result already exists")
    evidence=json.loads(source_report.read_text())
    orig=json.loads(s80_report.read_text())
    expert=json.loads(s446_report.read_text())
    need(evidence["status"]=="S283_EXTERNAL_ORIGINAL_S97_NATIVE_506_ONLY"
         and evidence["labels_MIDI_XML_read"] is False
         and evidence["output_SHA256"]==sha(source),
         "S450 requires exact original unlabelled S283 acoustic evidence")
    need(orig["status"]=="S297_COMPLETE_GENUINE_ORIGINAL_TRAINED_S80_UNSEEN_NATIVE_IDMT"
         and orig["original_native_WAV_events"]==N
         and orig["S80_K0_K2_original_final_formula_exact"]
         and orig["S79_original_trained_K1_models"]==4,
         "S450 original IDMT source K must be actual S297 pretrained full S80")
    need(expert["status"]=="S446_LEARNED_JOINT_NOTE_SET_SONGHELD_K_LOCKED_DEV"
         and expert["threshold_gates_separate_by_predicted_K"]
         and expert["original_S80_K_regressions"]==0
         and expert["no_promotion"],
         "S450 must use the original selected S446 frozen note-swap network")
    with np.load(source,allow_pickle=False) as z:
        ids=z["row_id"].astype(str)
        members=z["member"].astype(str)
        seq=z["sequence42x49"].astype(np.float32)
        starts=z["start_samples"].astype(np.int64)
        ends=z["end_samples"].astype(np.int64)
    with np.load(s80,allow_pickle=False) as z:
        kid=z["row_id"].astype(str)
        original_k=z["s80_K"].astype(np.int8)
    need(len(ids)==N and len(set(ids))==N
         and np.array_equal(ids,kid)
         and seq.shape==(N,42,49)
         and np.isin(original_k,range(7)).all()
         and np.isfinite(seq).all() and np.min(seq)>=0
         and all(x==str(m)+":"+str(int(st)) for x,m,st in zip(ids,members,starts))
         and len(set(members))==5,
         "S450 506 original IDMT event IDs or S80 source K mismatch")
    out.mkdir(parents=True)
    acoustic=out/"source_acoustic.npz"
    true_s80=out/"source_original_S80_K.npz"
    np.savez_compressed(acoustic,row_id=ids,sequence42x49=seq)
    np.savez_compressed(true_s80,row_id=ids,s80_K=original_k)
    baseline_dir=out/"unchanged_S430"
    original_infer(acoustic,true_s80,s402,s416,s428,s419_report,s428_report,
                   baseline_dir)
    with np.load(baseline_dir/"notes_from_new_sound.npz",allow_pickle=False) as z:
        need(np.array_equal(z["row_id"].astype(str),ids)
             and np.array_equal(z["s80_K"],original_k),
             "S450 original S430 event/K alignment diverged")
        base=z["midi_pitch_slots"].astype(np.int16)
    need(np.array_equal((base>=0).sum(1),original_k),
         "S450 original S430 note-head changed S80 K")
    all_votes=collections.defaultdict(list)
    enabled=collections.Counter()
    model_sha={}
    torch.set_num_threads(2)
    for fold in range(FOLDS):
        fname=s446/f"S446_joint_note_swap_fold{fold}.joblib"
        metadata=expert["per_fold"][fold]
        need(metadata["fold"]==fold and sha(fname)==metadata["model_sha256"],
             "S450 trained S446 fold SHA drift")
        obj=joblib.load(fname)
        need(obj["held_fold"]==fold and obj["uses_S80_predicted_K"],
             "S450 loaded wrong S446 learned head")
        gates={int(k):float(v) for k,v in obj["thresholds_by_predicted_K"].items()}
        need(all(k in (2,3,4) and 0<=v<=1 for k,v in gates.items()),
             "S450 invalid learned K-specific gate")
        cp=s416/f"S416_native_Kfirst_temporal_note_fold{fold}.pt"
        payload=torch.load(cp,map_location="cpu",weights_only=True)
        need(payload["architecture"]=="NativeKTimePitchNet"
             and payload["K_model_input"] is True
             and payload["K_output_locked"] is True
             and payload["training_uses_YourMT3"] is False
             and payload["fold"]==fold,
             "S450 genuine pretrained K-fed S416 provenance failed")
        net=NativeKTimePitchNet().eval()
        net.load_state_dict(payload["model"],strict=True)
        indices=np.flatnonzero(np.isin(original_k,list(gates)))
        with torch.no_grad():
            for start in range(0,len(indices),192):
                ii=indices[start:start+192]
                ps=net(torch.from_numpy(seq[ii]),
                       torch.from_numpy(original_k[ii].astype(np.int64))).sigmoid().numpy()
                for j,q in zip(ii,ps):
                    k=int(original_k[j])
                    ft,swaps=proposals(seq[j],q,base[j],k)
                    confidences=obj["model"].predict_proba(ft)[:,1]
                    winner=int(np.argmax(confidences))
                    if float(confidences[winner])>=gates[k]:
                        old,new=map(int,swaps[winner])
                        all_votes[int(j)].append((fold,(old,new)))
                        enabled[k]+=1
        model_sha[f"S446_fold{fold}"]=sha(fname)
        model_sha[f"S416_fold{fold}"]=sha(cp)
    chosen=base.copy()
    transitions=[]
    for i,votes in all_votes.items():
        old=[int(p) for p in chosen[i] if p>=40]
        updated,action=consensus(votes,old)
        if action is not None:
            chosen[i,:]=-1
            chosen[i,:len(updated)]=updated
            transitions.append(dict(row_id=str(ids[i]),predicted_K=int(original_k[i]),
                old_notes=[note_name(p) for p in old],
                new_notes=[note_name(p) for p in updated],votes=action["votes"]))
    need(np.array_equal((chosen>=0).sum(1),original_k)
         and np.array_equal(chosen[~np.isin(original_k,(2,3,4))],
                            base[~np.isin(original_k,(2,3,4))]),
         "S450 CRITICAL original S80 K0..6 or protected notes modified")
    frozen=out/"S450_IDMT_EXTERNAL_FROZEN_NOTE_NAMES.jsonl"
    with frozen.open("w") as fp:
        for row,k,pitches,m,st,en in zip(ids,original_k,chosen,members,starts,ends):
            midi=[int(p) for p in pitches if p>=0]
            fp.write(json.dumps(dict(row_id=str(row),member=str(m),
                start_samples=int(st),end_samples=int(en),
                s80_K=int(k),notes=[note_name(p) for p in midi],
                midi_pitch=midi),sort_keys=True)+"\n")
    artifact=out/"S450_IDMT_EXTERNAL_FROZEN_NOTE_NAMES.npz"
    np.savez_compressed(artifact,row_id=ids,member=members,start_samples=starts,
        end_samples=ends,s80_K=original_k,
        S430_pitch_slots=base,S450_pitch_slots=chosen)
    reread=[json.loads(l) for l in frozen.read_text().splitlines()]
    need(len(reread)==N and
         all(len(r["notes"])==r["s80_K"] for r in reread) and
         sum(len(r["notes"]) for r in reread)==int(original_k.sum()),
         "S450 published frozen musical names violate actual S80 K")
    rep=dict(status="S450_ORIGINAL_S80_EXTERNAL_IDMT_506_NOTE_NAMES_FROZEN",
        genuine_external_IDMT_dataset3=True,recordings=5,original_events=N,
        original_native_original_S80_K_used=True,
        original_S80_predictions_unchanged=True,exactK_regressions=0,
        original_S430_four_head_baseline_not_original_S428_GuitarSet_OOF=True,
        S446_external_consensus_minimum_independent_heads=AGREEMENT,
        note_set_changes=len(transitions),
        per_predicted_K_votes={str(k):int(v) for k,v in enabled.items()},
        note_set_transitions=transitions,
        actual_notes_sum=int(original_k.sum()),
        no_retraining_no_external_threshold_selection=True,
        no_XML_MIDI_truth_opened=True,source_S80_GuitarSet_training_overlap_not_external=True,
        external_recordings_IDMT_dataset3_were_previously_annotation_scored=True,
        no_new_blind_cohort=True,original_instrument_identity_unverified=True,
        not_untouched_other_instrument_generalization=True,
        no_Live_Offset_changes=True,
        full_raw_WAV_to_unchanged_S80_already_replayed_in_S277_S297_not_single_executable=True,
        experimental_not_promoted=True,
        checksums=dict(source_S283=sha(source),S80_S297=sha(s80),
            S446_report=sha(s446_report),S430_frozen=sha(baseline_dir/"notes_from_new_sound.npz"),
            frozen_notes=sha(frozen),prediction_npz=sha(artifact),models=model_sha))
    (out/"report.json").write_text(json.dumps(rep,indent=2)+"\n")
    print("S450 EXTERNAL FREEZE",json.dumps(dict(events=N,
          sum_S80_K=int(original_k.sum()),external_note_set_changes=len(transitions),
          original_exactK_regressions=0,per_K_pred=np.bincount(original_k,minlength=7).tolist())),
          flush=True)

def selftest():
    baseline=[52,55,59]
    out,_=consensus([(0,(55,67)),(2,(55,67))],baseline)
    need(out==[52,67,59],"S450 two different folds must agree on same swap")
    out,_=consensus([(0,(55,67)),(2,(59,71))],baseline)
    need(out==baseline,"S450 no consensus must abstain")
    out,_=consensus([(0,(55,67))],baseline)
    need(out==baseline,"S450 one fold must abstain")
    print("S450 SELFTEST PASS same-fold consensus, preserve K, abstention")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for arg in ("source","source-report","s80","s80-report",
                "s402","s416","s428","s419-report","s428-report",
                "s446","s446-report","out"):
        p.add_argument("--"+arg,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        fields=("source","source_report","s80","s80_report","s402","s416","s428",
                "s419_report","s428_report","s446","s446_report","out")
        for field in fields:need(getattr(a,field)is not None,"S450 missing "+field)
        run(*(getattr(a,field) for field in fields))
