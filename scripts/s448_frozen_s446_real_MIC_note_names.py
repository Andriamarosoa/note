"""S448 -- S446 saved crossfit models -> real GuitarSet MIC WAV notes.

No training, no retuning, no truth/annotation. Original S437/Source S80 K
and real acoustic sample alignment are immutably SHA checked. One held-
composition S416/S446 model per song (not an oracle by true K).
This is a retrospective MIC domain-transfer diagnostic, NOT an independent
unseen-song benchmark: S80 has historical GuitarSet exposure, and S437
baseline used a 4-model ensemble with prior training overlap.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import joblib,numpy as np,torch

from scripts.s439_klocked_real_MIC_joint_pitch_challengers import (
    source_audio_events,note_midi,
)
from scripts.s446_per_predicted_k_joint_note_songheld import proposals,REQUIRED
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    composition_fold,composition_key,need,
)
from scripts.s416_native_Kfirst_temporal_MIDI_note_decoder import NativeKTimePitchNet
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import note_name,sha

def run(frozen_names,s437_report,mic_zip,s416_models,
        s446_models,s446_report,out):
    need(not out.exists(),"S448 cannot change an existing MIC note freeze")
    source=json.loads(s437_report.read_text())
    trained=json.loads(s446_report.read_text())
    need(source["status"]=="S437_FROZEN_MULTI_GUITAR_REAL_MIC_WAV_S80_SOURCE_NOTE_NAMES"
         and source["number_distinct_original_compositions"]==12
         and source["true_K_labels_and_JAMS_NEVER_loaded_for_sampling_or_inference"]
         and sha(frozen_names)==source["output_SHA256"]
         and sha(mic_zip)==source["original_GuitarSet_MIC_audio_zip_SHA256"],
         "S448 source S437 real MIC frozen references no longer valid")
    need(trained["status"]=="S446_LEARNED_JOINT_NOTE_SET_SONGHELD_K_LOCKED_DEV"
         and trained["composition_disjoint_S446_folds"]
         and trained["threshold_gates_separate_by_predicted_K"]
         and trained["original_S80_K_regressions"]==0 and trained["no_promotion"],
         "S448 pretrained S446 model provenance does not match")
    rows=[json.loads(line) for line in frozen_names.read_text().splitlines()]
    need(len(rows)==174 and len({r["original_row_id"] for r in rows})==174,
         "S448 174 original S437 events altered")
    K=np.asarray([r["original_S80_K"] for r in rows],np.int64)
    need(sum(map(int,K))==340 and np.isin(K,range(7)).all(),
         "S448 real S437 original 340 note-count invariant failed")
    folds=np.asarray([composition_fold(composition_key(r["member"])) for r in rows])
    audio=source_audio_events(mic_zip,rows)
    seq=np.asarray(audio,np.float32)
    need(seq.shape==(174,42,49) and np.isfinite(seq).all()
         and np.min(seq)>=0,"S448 real microphone acoustic feature contract")
    original=[]
    for row in rows:
        p=[note_midi(v) for v in row["notes"]]
        need(len(p)==int(row["original_S80_K"]) and len(set(p))==len(p),
             "S448 original S437 real-note names malformed")
        original.append(p)
    chosen=[list(r) for r in original]
    decisions=[]
    checks={}
    torch.set_num_threads(2)
    for fold in range(4):
        s416_path=s416_models/f"S416_native_Kfirst_temporal_note_fold{fold}.pt"
        x=torch.load(s416_path,map_location="cpu",weights_only=True)
        need(x.get("architecture")=="NativeKTimePitchNet"
             and x.get("K_model_input") is True and
             x.get("K_output_locked") is True and
             x.get("training_uses_YourMT3") is False and
             int(x["fold"])==fold,
             "S448 crossfit S416 model architecture/provenance invalid")
        net=NativeKTimePitchNet().eval()
        net.load_state_dict(x["model"],strict=True)
        s446_path=s446_models/f"S446_joint_note_swap_fold{fold}.joblib"
        expected=trained["per_fold"][fold]
        need(int(expected["fold"])==fold and
             sha(s446_path)==expected["model_sha256"],
             "S448 pretrained note-swap model SHA verification failed")
        fit=joblib.load(s446_path)
        need(fit["held_fold"]==fold and fit["uses_S80_predicted_K"] is True,
             "S448 S446 held song model mismatch")
        gates={int(p):float(v) for p,v in fit["thresholds_by_predicted_K"].items()}
        need(all(k in REQUIRED and 0.0<=v<=1.0 for k,v in gates.items()),
             "S448 unexpected S446 predicted-K correction gate")
        ids=np.flatnonzero(folds==fold)
        if len(ids)==0:continue
        # Only the held-composition S416 head feeds this fold's correction
        # features. S80 predicted K is provided directly, never true K.
        with torch.no_grad():
            scores=net(torch.from_numpy(seq[ids]),
                       torch.from_numpy(K[ids])).sigmoid().numpy()
        need(scores.shape==(len(ids),49) and np.isfinite(scores).all(),
             "S448 genuine S416 frame/pitch scores malformed")
        for idx,q in zip(ids,scores):
            kk=int(K[idx])
            rec=dict(row_id=rows[idx]["original_row_id"],
                     original_predicted_K=kk,fold=fold,
                     corrected=False,predicted_confidence=None)
            if kk in gates:
                feat,swap=proposals(seq[idx],q,
                      np.asarray(original[idx]+[-1]*(6-kk),np.int16),kk)
                preds=fit["model"].predict_proba(feat)[:,1]
                winner=int(np.argmax(preds))
                confidence=float(preds[winner])
                rec["predicted_confidence"]=confidence
                if confidence>=gates[kk]:
                    old,new=map(int,swap[winner])
                    updated=list(chosen[idx])
                    need(old in updated and new not in updated,
                         "S448 S446 illegal note-set substitution")
                    updated[updated.index(old)]=new
                    chosen[idx]=updated
                    rec["corrected"]=True
            decisions.append(rec)
        checks[f"fold{fold}"]=dict(
            S416_SHA256=sha(s416_path),
            S446_SHA256=sha(s446_path),
            held_composition_events=len(ids),
            enabled_predicted_K=gates)
    need(len(decisions)==len(rows) and
         all(len(p)==int(k) and len(set(p))==len(p)
             for p,k in zip(chosen,K)),
         "S448 predicted S80 cardinality was changed")
    out.mkdir(parents=True)
    name=out/"S448_FROZEN_REAL_MIC_NOTE_NAMES.jsonl"
    with name.open("w") as fp:
        for row,midi in zip(rows,chosen):
            fp.write(json.dumps(dict(
                original_row_id=row["original_row_id"],
                member=row["member"],start_sample=row["start_sample"],
                end_sample=row["end_sample"],
                original_S80_K=row["original_S80_K"],
                notes=[note_name(p) for p in midi]),sort_keys=True)+"\n")
    noterows=[json.loads(s) for s in name.read_text().splitlines()]
    need(len(noterows)==174 and sum(len(r["notes"]) for r in noterows)==340,
         "S448 frozen note-name JSONL does not conserve true original K")
    changes=sum(collections.Counter(a)!=collections.Counter(b)
                for a,b in zip(chosen,original))
    rep=dict(status="S448_S446_CROSSFIT_REAL_GUITAR_MIC_WAV_NOTE_NAMES_FROZEN",
        original_12_MIC_compositions=True,n_events=174,
        original_predicted_K_total=340,
        source_S80_original_K_unchanged=True,exactK_regressions=0,
        no_truth_or_original_JAMS_loaded=True,no_training_or_retuning=True,
        no_YourMT_teacher=True,no_MIDI_file_export=True,
        composition_held_fold_models=True,
        original_S437_4fold_runtime_baseline_not_identical_to_train_OOF=True,
        GuitarSet_historical_training_overlap=True,
        source_MIC_not_pickup_mix=True,
        S438_scores_previously_known_retrospective_diagnostic=True,
        no_independent_unseen_music_validation=True,
        candidate_note_sets_changed=changes,
        note_identity_accuracy_not_yet_measured=True,
        no_promotion=True,
        historical_S80_Live_Offset_unchanged=True,
        original_frozen_S437_SHA256=sha(frozen_names),
        original_MIC_SHA256=sha(mic_zip),
        pretrained_S446_report_SHA256=sha(s446_report),
        checkpoint_SHA256=checks,
        output_SHA256=sha(name))
    (out/"report.json").write_text(json.dumps(rep,indent=2)+"\n")
    print("S448 FROZEN",json.dumps(dict(
        events=174,total_original_K=340,modified_note_sets=changes,
        by_predicted_K={str(kk):sum(r["original_predicted_K"]==kk and r["corrected"]
                                    for r in decisions) for kk in range(7)})),flush=True)
    print("S448 SUCCESS true real MIC sound to musical names; annotations unopened",flush=True)

def selftest():
    sample=[dict(notes=["E3","G3"],original_S80_K=2),
            dict(notes=[],original_S80_K=0)]
    assert [note_midi(p) for p in sample[0]["notes"]]==[52,55]
    assert sum(len(x["notes"]) for x in sample)==2
    print("S448 SELFTEST PASS true note names and original S80 K locked")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    for k in ("frozen-names","s437-report","mic-zip","s416-models",
              "s446-models","s446-report","out"):
        ap.add_argument("--"+k,type=Path)
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        for field in ("frozen_names","s437_report","mic_zip","s416_models",
                      "s446_models","s446_report","out"):
            need(getattr(a,field)is not None,"S448 missing "+field)
        run(a.frozen_names,a.s437_report,a.mic_zip,
            a.s416_models,a.s446_models,a.s446_report,a.out)
