"""S430 — native direct musical note NAMES from new acoustic event features.

INPUT (not WAV): sound event sequence42x49 and PRECOMPUTED S80 K0..K6,
aligned on source row IDs. OUTPUT: e.g. E3 or [E3,G3,B3] without MIDI file.
No labels, JAMS, weights from YourMT3 or training at inference time.

Three trained native components are reused: S402 K-conditioned MLP
probabilities, S416 K-conditioned temporal CNN probabilities and S428
harmonic ranker. Four-fold checkpoints are ENSEMBLED for NEW inputs.
This ensemble was never evaluated composition-held as a system; DO NOT
claim the S428 40.96% benchmark for the ensemble or replay old GuitarSet
to manufacture a higher metric. Raw WAV -> the original 42x49/S80
preprocessing remains a separate upstream requirement.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import joblib,numpy as np
import torch
from scripts.s402_k_input_first_supervised_midi_pitch_network import feature_bank,need
from scripts.s416_native_Kfirst_temporal_MIDI_note_decoder import NativeKTimePitchNet
from scripts.s428_train_sound_to_note_names_harmonic_ranker import build_features,FEATURES
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import (
    choose_pitches,note_name,sha)

N_MODELS=4

def load_models(d402,d416,d428,s419_report,s428_report):
    p402=[];p416=[];p428=[]
    checks={}
    for f in range(4):
        name=d402/f"S402_pitch_head_held_song_fold{f}.joblib"
        need(name.is_file(),"Missing S402 checkpoint "+str(name))
        obj=joblib.load(name)
        need(obj.get("S80_K_fed_to_model_input") is True and
             obj.get("feature_width")==352 and int(obj["held_fold"])==f,
             "S430 S402 checkpoint provenance mismatched")
        p402.append(obj)
        checks[f"S402_fold{f}"]=sha(name)

        name=d416/f"S416_native_Kfirst_temporal_note_fold{f}.pt"
        need(name.is_file(),"Missing S416 checkpoint "+str(name))
        obj=torch.load(name,map_location="cpu",weights_only=True)
        need(obj.get("architecture")=="NativeKTimePitchNet" and
             obj.get("K_model_input") is True and
             obj.get("K_output_locked") is True and
             obj.get("training_uses_YourMT3") is False and
             int(obj["fold"])==f,
             "S430 S416 native K input or weights provenance mismatched")
        net=NativeKTimePitchNet().eval()
        net.load_state_dict(obj["model"],strict=True)
        p416.append(net)
        checks[f"S416_fold{f}"]=sha(name)

        name=d428/f"S428_acoustic_harmonic_pitch_ranker_fold{f}.joblib"
        need(name.is_file(),"Missing original harmonic pitch model "+str(name))
        obj=joblib.load(name)
        need(obj.get("feature_width")==FEATURES
            and obj.get("original_S80_K_required") is True
            and obj.get("no_YourMT3_at_train_or_test") is True
            and int(obj["held_composition_fold"])==f,
            "S430 S428 trained musical pitch model provenance mismatch")
        p428.append(obj["model"])
        checks[f"S428_fold{f}"]=sha(name)
    gate419=json.loads(s419_report.read_text())["fold_gates"]
    gate428=json.loads(s428_report.read_text())["training_folds"]
    decisions={}
    for K in range(1,7):
        old_votes=sum(bool(gate419[str(f)][str(K)]["use_S416"]) for f in range(4))
        new_votes=sum(bool(row["note_head_gate"][str(K)]["active"]) for row in gate428)
        decisions[str(K)]=dict(
             S419_use_CNN=old_votes>=2,
             S428_use_harmonic_ranker=new_votes>=2,
             S419_votes=old_votes,S428_votes=new_votes)
    return p402,p416,p428,decisions,checks

def run(audio,s80,d402,d416,d428,report419,report428,out):
    need(not out.exists(),"S430 immutable new acoustic inference output exists")
    with np.load(audio,allow_pickle=False) as z:
        need({"row_id","sequence42x49"}.issubset(z.files),
             "S430 input sound events need original acoustic source features")
        ids=z["row_id"].astype(str)
        seq=z["sequence42x49"].astype(np.float32)
    with np.load(s80,allow_pickle=False) as z:
        need({"row_id","s80_K"}.issubset(z.files),
             "S430 original predicted S80 K source is mandatory")
        ref=z["row_id"].astype(str);K=z["s80_K"].astype(np.int8)
    n=len(ids)
    need(n>0 and len(set(ids))==n and np.array_equal(ids,ref)
        and seq.shape==(n,42,49)
        and np.isfinite(seq).all() and np.min(seq)>=0
        and np.isin(K,range(7)).all(),
        "S430 sound feature/K event schema invalid")
    torch.set_num_threads(2)
    head402,head416,head428,decisions,hashes=load_models(
         d402,d416,d428,report419,report428)
    chosen=np.full((n,6),-1,np.int16)
    provenance={"S402":0,"S416":0,"S428":0}
    for off in range(0,n,256):
        end=min(off+256,n)
        s=seq[off:end];kk=K[off:end]
        ft=feature_bank(s,kk)
        prob402=np.zeros((len(kk),49),np.float32)
        for net in head402:
            prob402+=net["network"].predict_proba(
                net["transform"].transform(ft)).astype(np.float32)/4
        prob416=np.zeros_like(prob402)
        with torch.no_grad():
            ac=torch.from_numpy(s)
            kv=torch.from_numpy(kk.astype(np.int64))
            for net in head416:
                prob416+=net(ac,kv).sigmoid().numpy()/4
        old,_=choose_pitches(prob402,kk)
        cnn,_=choose_pitches(prob416,kk)
        reference=old.copy()
        for i,k in enumerate(kk):
            if k>0 and decisions[str(int(k))]["S419_use_CNN"]:
                reference[i]=cnn[i]
        physical=build_features(s,kk,prob416,reference,old)
        confidence=np.zeros_like(prob402)
        for net in head428:
            confidence+=net.predict_proba(
                 physical.reshape(-1,FEATURES))[:,1].reshape(-1,49).astype(np.float32)/4
        acoustic,_=choose_pitches(confidence,kk)
        for i,k in enumerate(kk):
            if k==0:
                chosen[off+i]=reference[i]
                continue
            method="S428" if decisions[str(int(k))]["S428_use_harmonic_ranker"] else (
                "S416" if decisions[str(int(k))]["S419_use_CNN"] else "S402")
            provenance[method]+=1
            chosen[off+i]=acoustic[i] if method=="S428" else reference[i]
    need(np.array_equal((chosen>=0).sum(1),K),
         "S430 CRITICAL new note name count differs from S80 K")
    out.mkdir(parents=True)
    fname=out/"notes_from_new_sound.jsonl"
    with fname.open("w",encoding="utf8") as h:
        for row,k,pitches in zip(ids,K,chosen):
            note_list=[int(p) for p in pitches if p>=0]
            h.write(json.dumps(dict(source_row_id=str(row),K=int(k),
                notes=[note_name(p) for p in note_list],
                MIDI_pitches=note_list),ensure_ascii=False)+"\n")
    replayed=0
    with fname.open(encoding="utf8") as h:
        for i,line in enumerate(h):
            p=json.loads(line)
            need(p["K"]==int(K[i]) and p["source_row_id"]==ids[i]
             and len(p["notes"])==int(K[i])
             and p["notes"]==[note_name(int(x)) for x in p["MIDI_pitches"]],
             "S430 exported musical names do not conserve predicted K")
            replayed+=1
    need(replayed==n,"S430 actual sound event row count mismatch")
    pred=out/"notes_from_new_sound.npz"
    np.savez_compressed(pred,row_id=ids,s80_K=K,midi_pitch_slots=chosen)
    report=dict(status="S430_NATIVE_SOUND_EVENT_TO_MUSICAL_NOTE_NAMES",
        original_S80_K_supplied_upstream=True,
        original_S80_K_unchanged_every_event=True,
        original_source_events=n,
        actual_emitted_note_count=int(K.sum()),
        exactK_regressions=0,
        actual_emitted_note_names_replayed_from_file=True,
        no_YourMT3_at_inference=True,
        no_JAMS_or_true_K_loaded=True,
        no_model_fitting_or_calibration_at_inference=True,
        output_is_note_names_not_MIDI_file=True,
        raw_WAV_to_42x49_and_S80_still_required_upstream=True,
        new_audio_4model_ensemble_not_S428_song_OOF_benchmark=True,
        new_audio_note_accuracy_not_yet_measured=True,
        model_votes_for_unseen_music_not_independently_validated=True,
        gate_vote_policy_for_new_music=decisions,
        event_model_usage=provenance,
        original_S80_sha256=sha(s80),acoustic_sha256=sha(audio),
        model_sha256=hashes,
        names_jsonl_sha256=sha(fname),predictions_sha256=sha(pred),
        official_promotion=False)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S430 NOTES",json.dumps(dict(events=n,note_count=int(K.sum()),
                                   per_K_policy=decisions)),flush=True)
    print("S430 PASS NEW sound event features -> musical names E3/G3/B3, zero K changes",flush=True)

def selftest():
    from tempfile import TemporaryDirectory
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    torch.manual_seed(430)
    with TemporaryDirectory() as path:
        root=Path(path)
        for name in ("s402","s416","s428"):
            (root/name).mkdir()
        ids=np.array(["a","b","c","d"])
        x=np.random.default_rng(430).random((4,42,49)).astype(np.float32)*.03
        k=np.array([0,1,2,3],np.int8)
        x[:,15:22,12]=np.log1p(3.0)
        np.savez_compressed(root/"acoustic.npz",row_id=ids,sequence42x49=x)
        np.savez_compressed(root/"s80.npz",row_id=ids,s80_K=k)
        train_x=feature_bank(x,k)
        y=np.zeros((4,49),np.uint8)
        y[0,0]=1;y[1,12]=1;y[2,15]=1;y[3,19]=1
        for fold in range(4):
            std=StandardScaler().fit(train_x)
            clf=MLPClassifier(hidden_layer_sizes=(6,),max_iter=2,
                random_state=fold,batch_size=4)
            clf.fit(std.transform(train_x),y)
            joblib.dump(dict(transform=std,network=clf,
                S80_K_fed_to_model_input=True,feature_width=352,held_fold=fold),
                root/"s402"/f"S402_pitch_head_held_song_fold{fold}.joblib")
            cnn=NativeKTimePitchNet()
            torch.save(dict(model=cnn.state_dict(),architecture="NativeKTimePitchNet",
                K_model_input=True,K_output_locked=True,
                training_uses_YourMT3=False,fold=fold),
                root/"s416"/f"S416_native_Kfirst_temporal_note_fold{fold}.pt")
            rng=np.random.default_rng(430+fold)
            feature_rows=rng.random((90,FEATURES)).astype(np.float32)
            targets=np.arange(90)%2
            tree=HistGradientBoostingClassifier(max_iter=3,min_samples_leaf=3)
            tree.fit(feature_rows,targets)
            joblib.dump(dict(model=tree,feature_width=FEATURES,
                original_S80_K_required=True,
                no_YourMT3_at_train_or_test=True,held_composition_fold=fold),
                root/"s428"/f"S428_acoustic_harmonic_pitch_ranker_fold{fold}.joblib")
        gates={str(f):{str(i):dict(use_S416=i==1)
                       for i in range(7)} for f in range(4)}
        (root/"g419.json").write_text(json.dumps(dict(fold_gates=gates)))
        (root/"g428.json").write_text(json.dumps(dict(training_folds=[
           dict(note_head_gate={str(i):dict(active=i in (1,2,3))
                                    for i in range(1,7)})
           for f in range(4)])))
        run(root/"acoustic.npz",root/"s80.npz",
            root/"s402",root/"s416",root/"s428",
            root/"g419.json",root/"g428.json",root/"out")
        with np.load(root/"out/notes_from_new_sound.npz") as z:
            need(np.array_equal((z["midi_pitch_slots"]>=0).sum(1),k),
                 "S430 new input note names cardinality differs from K")
        actual=[json.loads(x) for x in
                (root/"out/notes_from_new_sound.jsonl").read_text().splitlines()]
        need(actual[0]["notes"]==[] and len(actual[3]["notes"])==3,
             "S430 note sound example failed")
    print("S430 SELFTEST PASS new S80 acoustic events K0-3 -> real note names, no teacher")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for flag in ("audio","s80","s402-models","s416-models",
                 "s428-models","s419-report","s428-report","out"):
        p.add_argument("--"+flag,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for field in ("audio","s80","s402_models","s416_models",
                      "s428_models","s419_report","s428_report","out"):
            need(getattr(a,field)is not None,"missing --"+field)
        run(a.audio,a.s80,a.s402_models,a.s416_models,a.s428_models,
            a.s419_report,a.s428_report,a.out)
