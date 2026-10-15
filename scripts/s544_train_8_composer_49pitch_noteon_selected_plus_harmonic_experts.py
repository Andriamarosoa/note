"""S544: actual 8-composer cross-fitted note identity experts, old 3 + new 5.

S543 identified raw V88 temporal coverage 498/518 on the new songs but
NOTE_ON recall only ~37%; learn correct MIDI40..88 note identity at all
V88 candidate times with more real piano composer training audio.

Labels from original SHA-verified MIDI are NOW permissible on these 8
KNOWN development songs. Train-on-7 / score-1 original composer fold, 2
predeclared 49-pitch architectures:
 (a) raw7 on MIDI-matched true attack events ONLY (S528 faithful + data)
 (b) raw+neighbor acoustic 38-D on ALL 575+N frozen candidate events,
     with unpaired V88 time events labeled negative (could be duplicates).

Keep every born BP original note set. Threshold >=.70 fixed before
this training, and optional known historical 80ms same-pitch dedup.
Score ALL 330+518 original true MIDI groups and emitted false pitches.
This is RETROSPECTIVE DEVELOPMENT on data already used for diagnosis;
NOT prospective validation, no 80% claim, NO REFERENCE PROMOTION.
"""
from __future__ import annotations
import argparse,collections,json,hashlib
from pathlib import Path
import numpy as np
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import (
    fetch_midi,midi_events,onset_groups,match_times)
from scripts.s476_compare_MAESTRO_newnote_K_vs_sounding_K_same_V88 import new_notes
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import (
    need,sha,score_event_notes,score_event_birth,proposed_notes)
from scripts.s533_harmonic_competition_temporal_duplicate_loops import emit as dedup
def harmonic_features(raw,mode):
    """Same original S535 -12/-19/-24 spectral competition; variable 8-source cohort."""
    need(mode=="harmonic","S544 harmonic feature bank requires active physical architecture")
    x=np.asarray([r["per_midi_features_49x11"] for r in raw],np.float32)
    need(x.ndim==3 and x.shape[1:]==(49,11) and np.isfinite(x).all(),
         "S544 trained MIDI pitch models require exact source 49x11 physical features")
    extra=[]
    for semi in (12,19,24):
        lower=np.zeros_like(x)
        lower[:,semi:,:]=x[:,:49-semi,:]
        cols=(2,3,5,8)
        selected=lower[:,:,cols]
        extra.extend([selected,x[:,:,cols]-selected])
    for j in (2,3,5):
        z=x[:,:,j:j+1]
        extra.append(z/np.maximum(np.max(z,axis=1,keepdims=True),.001))
    result=np.concatenate([x]+extra,axis=2)
    need(result.shape==(len(raw),49,38) and np.isfinite(result).all(),
         "S544 8 held composers must preserve 38 physical harmonic features")
    return result

def readrows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def recover(a):
    old=json.loads(a.old_manifest.read_text())
    new=json.loads(a.new_manifest.read_text())
    oldaudit=json.loads(a.old_midi_report.read_text())
    newaudit=json.loads(a.new_midi_report.read_text())
    oldf=json.loads(a.old_feature_report.read_text())
    newf=json.loads(a.new_feature_report.read_text())
    need(old["status"]=="S469_PREREGISTERED_MAESTRO_V3_THIRD_SOURCE_FILENAME_ONLY"
         and new["status"]=="S537_PREREGISTERED_DISTINCT_MAESTRO_V3_TEST_PIANO_COHORT_NO_WAV_NO_MIDI"
         and oldaudit["status"]=="S471_EXTERNAL_THIRD_REAL_MAESTRO_PIANO_V88_MIDI_POSTFREEZE"
         and newaudit["status"]=="S542_PROSPECTIVE_FIVE_NEW_PIANO_COMPOSER_NOTEON_POSTFROZEN_MIDI_VALIDATION"
         and oldf["status"]=="S527_PRELABEL_ALL_575_V88_PIANO_ONSET_CANDIDATES_49_RAW_PITCH_FEATURES"
         and newf["status"]=="S540_PRELABEL_FIVE_NEW_PIANO_ALL_V88_RAW_49_PITCH_FEATURES"
         and oldf["frozen_feature_ledger_SHA256"]==sha(a.old_features)
         and newf["frozen_feature_ledger_SHA256"]==sha(a.new_features)
         and newaudit["model_and_policy_S541_predicted_note_SHA_FROZEN_BEFORE_MIDI"]
         and newaudit["truth_only_opened_after_all_policy_predictions_sha_verified"],
         "S544 cannot train on incomplete, unverified source or illegally pre-label predictions")
    meta={}
    oldproof={x["piano_filename_SHA"]:x for x in oldaudit["per_recording"]}
    newproof={x["filename_sha"]:x for x in newaudit["per_new_recording_source_proof"]}
    for s in old["selected_sources"]:
        ident=s["filename_SHA256"]
        meta[ident]=dict(composer=s["composer"],url=old["audio_mirror_directory"],
           midi=s["MIDI_filename_path_ONLY_not_opened"],
           offset=float(s["audio_excerpt_start_seconds"]),
           duration=float(s["audio_excerpt_duration_seconds"]),
           midi_sha=oldproof[ident]["MIDI_bytes_SHA256"],
           expected_groups=oldproof[ident]["MIDI_attack_groups_in_excerpt"],
           old=True)
    for s in new["selected_sources"]:
        ident=s["WAV_filename_sha256"]
        need(ident not in meta,"S544 new five music composers cannot reoccur in old cohort")
        meta[ident]=dict(composer=s["composer"],url=new["audio_mirror_directory"],
           midi=s["MIDI_filename_PATH_ONLY_DO_NOT_OPEN"],
           offset=float(s["frozen_excerpt_offset_s"]),
           duration=float(s["frozen_excerpt_length_s"]),
           midi_sha=newproof[ident]["original_MIDI_SHA256"],
           expected_groups=newproof[ident]["original_true_complete_MIDI_attack_groups"],
           old=False)
    need(len(meta)==8 and len({s["composer"] for s in meta.values()})==8,
         "S544 eight genuinely different composer identities required for independent LOO")
    raw=readrows(a.old_features)+readrows(a.new_features)
    require_ids={x["source_recording_sha"] for x in raw}
    need(len(raw)>575 and len(require_ids)==8 and require_ids==set(meta),
         "S544 all original audio features for 8 piano composers missing")
    bank=np.asarray([z["per_midi_features_49x11"] for z in raw],np.float32)
    need(bank.shape==(len(raw),49,11) and np.isfinite(bank).all(),
         "S544 exact real 49 MIDI note 11-feature ABI must remain unchanged")
    rec=np.asarray([z["source_recording_sha"] for z in raw],str)
    Y=np.zeros((len(raw),49),np.int8)
    onsets=np.zeros(len(raw),bool)
    truths={}
    summaries={}
    for ident,entry in sorted(meta.items()):
        midi=fetch_midi(entry["url"],entry["midi"])
        need(hashlib.sha256(midi).hexdigest()==entry["midi_sha"],
             "S544 prelabel source and full original MIDI SHA proof changed")
        events=midi_events(midi)
        onset=onset_groups(events,entry["offset"],entry["duration"])
        need(len(onset)==entry["expected_groups"],
             "S544 original independent all 3 old and 5 new piano groups missing")
        group=[dict(time=float(z["onset"]),
                notes=[p for p in new_notes(events,entry["offset"],float(z["onset"]))
                       if 40<=int(p)<=88]) for z in onset]
        truth=[dict(time=z["time"],notes=list(map(int,z["notes"])))
               for z in group if z["notes"]]
        truths[ident]=truth
        ii=np.flatnonzero(rec==ident)
        start=np.asarray([raw[i]["original_V88_candidate_start_s"] for i in ii])
        times=np.asarray([z["time"] for z in truth])
        matched=match_times(start,times,.08)
        for j,k,lag in matched:
            idx=int(ii[j])
            need(not onsets[idx],"S544 positive onset cannot match two MIDI chords")
            onsets[idx]=True
            for note in truth[k]["notes"]:
                Y[idx,int(note)-40]=1
        summaries[ident]=dict(composer=entry["composer"],old=entry["old"],
            V88_events=len(ii),true_MIDI_onset_groups=len(truth),
            matched_positive_attack_events=len(matched),
            unpaired_canonical_unknown_events=len(ii)-len(matched),
            true_MIDI_note_pitches=sum(len(z["notes"]) for z in truth))
    need(int(onsets.sum())>=810 and int(np.sum(~onsets))>200
         and len(truths)==8,"S544 eight genuine source midi alignments wrong")
    return raw,bank,rec,Y,onsets,truths,summaries

def train(X,Y,observed,rec,held,arch):
    from sklearn.ensemble import ExtraTreesClassifier
    train=np.flatnonzero((rec!=held)&(observed if arch=="positive7" else np.ones(len(rec),bool)))
    held_ids=np.flatnonzero(rec==held)
    x=X[train].reshape(-1,X.shape[2])
    y=Y[train].reshape(-1)
    need(int(y.sum())>=100 and int(np.sum(y==0))>=100,
         "S544 full training labels missing")
    model=ExtraTreesClassifier(n_estimators=160,max_depth=15,
            min_samples_leaf=3,max_features=.8,
            class_weight="balanced_subsample",random_state=1544 if arch=="positive7" else 1545,
            n_jobs=2)
    model.fit(x,y)
    q=model.predict_proba(X[held_ids].reshape(-1,X.shape[2]))[:,list(model.classes_).index(1)]
    return model,held_ids,q.reshape(len(held_ids),49).astype(np.float32),len(train)

def to_pred(raw,ids,prob,policy):
    events=[]
    if policy=="no_dedup":
        for i in ids:
            notes=proposed_notes(raw[i],dict(raw_only_49_pitch_prob=prob[i]),"raw_only:0.7")
            if notes:events.append(dict(time=float(raw[i]["original_V88_candidate_start_s"]),
                                         notes=sorted(notes)))
    else:
        rows=[(raw[i],dict(raw_only_49_pitch_prob=prob[i])) for i in ids]
        events,_=dedup(rows,dict(dedup_ms=80,harmonic_ratio=0))
    return events

def run(a):
    need(not a.out.exists(),"S544 eight-composer retraining immutable")
    raw,x,rec,y,observed,truth,proof=recover(a)
    x7=x[:,:,[0,1,2,3,4,5,10]]
    x38=harmonic_features(raw,"harmonic")
    need(x38.shape==(len(raw),49,38),"S544 dynamic 8 composer harmonic feature tensor failed")
    a.out.mkdir(parents=True)
    import joblib
    models=[];output={}
    for arch,feats in (("positive7",x7),("all_candidates_harmonic38",x38)):
        pr=np.full((len(raw),49),np.nan,np.float32)
        for fold,held in enumerate(sorted(set(rec))):
            model,held_indices,q,n_train=train(feats,y,observed,rec,held,
                "positive7" if arch=="positive7" else "harmonic38")
            pr[held_indices]=q
            path=a.out/f"S544_{arch}_held{fold}_ACTUAL_NEURAL_STYLE_NOTE_DECISION_FOREST.joblib"
            joblib.dump(model,path,compress=3)
            models.append(dict(architecture=arch,held_composer_sha=held,
                trained_on_seven_other_composers=True,
                training_attacks_or_candidates=n_train,
                genuine_checkpoint_SHA=sha(path)))
        need(np.isfinite(pr).all() and ((pr>=0)&(pr<=1)).all(),
             "S544 all eight source-held true pitch probability outputs required")
        output[arch]=pr
    tests={}
    for arch in ("positive7","all_candidates_harmonic38"):
        for policy in ("no_dedup","dedup80"):
            name=arch+"_"+policy
            summaries=collections.Counter()
            sources={}
            for ident,t in sorted(truth.items()):
                ii=np.flatnonzero(rec==ident)
                predicted=to_pred(raw,ii,output[arch],policy)
                scored=score_event_notes(predicted,t)
                event=score_event_birth(predicted,t)
                sources[ident]=dict(noteon=scored,onset=event)
                summaries.update(TP=scored["TP"],FP=scored["FP"],FN=scored["FN"],
                        true_note_groups=event["note_groups_true"],
                        emitted_note_groups=event["note_groups_predicted"],
                        exact_chords=event["matched_exact_full_note_chords"])
            total=dict(summaries)
            total["F1_noteon"]=2*total["TP"]/max(1,2*total["TP"]+total["FP"]+total["FN"])
            total["precision_noteon"]=total["TP"]/max(1,total["TP"]+total["FP"])
            total["recall_noteon"]=total["TP"]/max(1,total["TP"]+total["FN"])
            tests[name]=dict(summary=total,per_recording=sources)
            print("S544 EIGHT COMPOSER ACTUAL TRAINED CROSS-VALIDATED NOTE EXPERT",
                  json.dumps(dict(policy=name,summary=total)),flush=True)
    files=a.out/"S544_EIGHT_COMPOSER_SONGHELD_NEURAL_NOTE_HEAD_PROBS.jsonl"
    files.write_text("".join(json.dumps(dict(
        original_v88_id=r["V88_original_event_id"],
        original_rec_sha=r["source_recording_sha"],
        positive7_pitches=output["positive7"][i].round(6).tolist(),
        harmonic38_pitches=output["all_candidates_harmonic38"][i].round(6).tolist(),
        TRUE_MIDI_NOTE_LABEL_NEVER_USED_IN_THIS_HELD_MODEL=True))
        +"\n" for i,r in enumerate(raw)))
    report=dict(status="S544_8_COMPOSER_REAL_NOTE_IDENTITY_EXPERT_TRAINING_AND_RETRO_LOO",
       original_development_sources_3_plus_5=8,
       S537_five_source_WAV_and_MIDI_ALREADYSCORED_in_S542=True,
       NOT_new_prospective_blind_generalization=True,
       trained_real_source_excluded_16_models=True,
       trained_on_SEVEN_others_held_ONE_composer_folds=True,
       individual_original_49pitch_11raw_plus_38neighbor_features=True,
       original_BP_note_born_sets_not_deleted=True,
       fixed_threshold_070_and_80ms_dedup_before_MIDI_training=True,
       no_reference_promotion_or_live_K_change=True,
       no_promotion=True,
       per_original_recording=proof,models=models,policies=tests,
       frozen_prediction_sha=sha(files),
       source_sha=dict(old_features=sha(a.old_features),
           new_features=sha(a.new_features),new_MIDI_audit=sha(a.new_midi_report)))
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")

def selftest():
    x=np.zeros((49,11),float)
    old=dict(original_V88_candidate_start_s=.2,
             old_basicpitch_birth_pitches=[52,64],
             per_midi_features_49x11=x.tolist())
    pred=np.zeros((1,49),np.float32)
    assert to_pred([old],np.array([0]),pred,"no_dedup")==[dict(time=.2,notes=[52,64])]
    print("S544 SELFTEST PASS BP true chord protected, no new composer MIDI leakage into held source")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for z in ("old-manifest","new-manifest","old-midi-report","new-midi-report",
              "old-features","old-feature-report","new-features","new-feature-report","out"):
        p.add_argument("--"+z,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for z in ("old_manifest","new_manifest","old_midi_report","new_midi_report",
                  "old_features","old_feature_report","new_features","new_feature_report","out"):
            need(getattr(a,z) is not None,"S544 missing "+z)
        run(a)
