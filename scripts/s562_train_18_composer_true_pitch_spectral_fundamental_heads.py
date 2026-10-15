"""S562 18-source REAL MIDI-trained pitch identity and lower-harmonic physical expert.

13? Actually 3 S469 +5 S537+5 S545+5 S554 = 18 EXPOSED MUSIC
RECORDINGS, treated as development; NOT new prospective validation.

Two preregistered tree architectures:
- raw7 (S546-compatible original physical note acoustic features)
- competing_fundamental31 (raw7, acoustic lower-pitch support per actual
  harmonic intervals -12,-19,-24, differential onset/novelty/strength,
  local competition and physical transient).
Every train feature is derived from original 49x11 WAV only, no test MIDI.
Train 18 actual composer-component held folds; score true MIDI note+onset
one-to-one with all emitted FP, every missed true NoteOn FN. Train 2
additional FULL-18 source deployment models for next S561 holdout, SHA
sealed; NO MIDI from S561 touched, no new reference or Live.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s546_train_8_source_folds_excluding_SHARED_COMPOSER_COMPONENTS import (
  recover, composer_components)
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import (
  fetch_midi,midi_events,onset_groups,match_times)
from scripts.s476_compare_MAESTRO_newnote_K_vs_sounding_K_same_V88 import new_notes
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import need,sha,score_event_notes,proposed_notes
from scripts.s536_onset_burst_adaptive_restrike_dedup_loops import emit as dedup100

def readrows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def acoustic(X,architecture):
    need(X.ndim==3 and X.shape[1:]==(49,11) and np.isfinite(X).all(),
         "S562 physical feature input must be 49 MIDI pitches times 11 acoustic indices")
    basic=X[:,:,[0,1,2,3,4,5,10]]
    if architecture=="raw7":return basic
    need(architecture=="competing_fundamental34","S562 only 2 preregistered heads")
    extras=[]
    idx=[2,3,5] # strength, birth burst and genuine novelty
    for semis in (12,19,24):
        lower=np.zeros_like(X)
        lower[:,semis:,:]=X[:,:49-semis,:]
        extras.extend([lower[:,:,idx],X[:,:,idx]-lower[:,:,idx]])
    for j in idx:
        x=X[:,:,j:j+1]
        extras.append(x/np.maximum(x.max(axis=1,keepdims=True),.001))
    for j in idx:
        note=X[:,:,j:j+1]
        lo=np.zeros_like(note);hi=np.zeros_like(note)
        lo[:,1:,:]=note[:,:-1,:]
        hi[:,:-1,:]=note[:,1:,:]
        extras.append(note-np.maximum(lo,hi))
    # Spectral dynamics separating fresh strike from an already sustained harmonic.
    extras.extend([X[:,:,4:5],X[:,:,3:4]-X[:,:,4:5],
                   X[:,:,5:6]-X[:,:,4:5]])
    result=np.concatenate([basic]+extras,axis=2).astype(np.float32)
    need(result.shape==(len(X),49,34) and np.isfinite(result).all(),
         "S562 waveform pitch lower-harmonic acoustic head shape is 34 not 38")
    return result

def extension(a,meta,raw,rec,Y,onsets,truth,proof):
    for name in ("S545","S554"):
        man=json.loads(getattr(a,name.lower()+"_manifest").read_text())
        report=json.loads(getattr(a,name.lower()+"_midi_report").read_text())
        feat=json.loads(getattr(a,name.lower()+"_feature_report").read_text())
        path=getattr(a,name.lower()+"_features")
        expected_manifest={
           "S545":"S545_PREREGISTERED_FIVE_MORE_TRAIN_SPLIT_COMPOSERS_AFTER_S544_NO_WAV_MIDI",
           "S554":"S554_PREREGISTERED_FIVE_MORE_TRAIN_SPLIT_COMPOSERS_AFTER_S544_NO_WAV_MIDI"}[name]
        expected_midi={
           "S545":"S551_EXTERNAL_POSTFROZEN_FIVE_NEW_COMPOSER_FULL_MIDI_NOTEON_F1",
           "S554":"S559_EXTERNAL_POSTFROZEN_FIVE_NEW_COMPOSER_FULL_MIDI_NOTEON_F1"}[name]
        expected_feat={
           "S545":"S549_PRELABEL_FIVE_NEW_PIANO_ALL_V88_RAW_49_PITCH_FEATURES",
           "S554":"S557_PRELABEL_FIVE_NEW_PIANO_ALL_V88_RAW_49_PITCH_FEATURES"}[name]
        need(man["status"]==expected_manifest and
             report["status"]==expected_midi and
             feat["status"]==expected_feat and
             feat["frozen_feature_ledger_SHA256"]==sha(path) and
             report["S550_all_six_named_note_policies_SHA_VERIFIED_before_true_MIDI"] if name=="S545" else
             man["status"]==expected_manifest and report["status"]==expected_midi
             and feat["status"]==expected_feat and
             feat["frozen_feature_ledger_SHA256"]==sha(path)
             and report["S558_all_six_named_note_policies_SHA_VERIFIED_before_true_MIDI"],
             "S562 previously independently verified source MIDIs and audio features required")
        newrows=readrows(path)
        midi_proof={x["original_audio_filename_sha"]:x for x in report["true_MIDI_provenance"]}
        for row in man["chosen_metadata_only"]:
            ident=row["filename_SHA256"]
            need(ident not in meta and ident in midi_proof,
                 "S562 cannot reuse old trained source under another label")
            p=midi_proof[ident]
            meta[ident]=dict(composer=row["composer"],url=man["source_audio_mirror"],
                midi=row["MIDI_filename_path_ONLY_not_opened"],
                offset=float(row["audio_excerpt_start_seconds"]),
                duration=float(row["audio_excerpt_duration_seconds"]),
                midi_sha=p["original_MIDI_sha256"],
                expected_groups=p["original_MIDI_onset_groups"],old=False)
        raw.extend(newrows)
    need(len(raw)>18 and len(meta)==18 and len({r["source_recording_sha"] for r in raw})==18,
         "S562 original 3+5+5+5 audio sources must all be unique")
    X=np.asarray([r["per_midi_features_49x11"] for r in raw],np.float32)
    rr=np.asarray([r["source_recording_sha"] for r in raw],str)
    Ynew=np.zeros((len(raw),49),np.int8)
    Ynew[:len(Y)]=Y
    onsetnew=np.zeros(len(raw),bool)
    onsetnew[:len(onsets)]=onsets
    for ident,item in sorted(meta.items()):
        if ident in truth:continue
        midi=fetch_midi(item["url"],item["midi"])
        need(hashlib.sha256(midi).hexdigest()==item["midi_sha"],
             "S562 original previously evaluated MIDI SHA changed during new training")
        events=midi_events(midi)
        start=float(item["offset"]);dur=float(item["duration"])
        groups=onset_groups(events,start,dur)
        need(len(groups)==item["expected_groups"],
             "S562 complete MIDI attack-group count must match previous held evaluation")
        gt=[dict(time=float(g["onset"]),
             notes=[int(x) for x in new_notes(events,start,float(g["onset"]))
                    if 40<=int(x)<=88]) for g in groups]
        gt=[z for z in gt if z["notes"]]
        truth[ident]=gt
        ii=np.flatnonzero(rr==ident)
        native=np.asarray([float(raw[j]["original_V88_candidate_start_s"]) for j in ii])
        pairs=match_times(native,np.asarray([q["time"] for q in gt]),.08)
        for j,k,dt in pairs:
            idx=int(ii[j])
            need(not onsetnew[idx],"S562 time matching must be 1:1")
            onsetnew[idx]=True
            for note in gt[k]["notes"]:Ynew[idx,int(note)-40]=1
        proof[ident]=dict(composer=item["composer"],V88_events=len(ii),
          true_MIDI_onset_groups=len(gt),true_MIDI_note_pitches=sum(len(q["notes"]) for q in gt),
          matched_positive_attack_events=len(pairs),
          source_original_MIDI_sha=item["midi_sha"])
    need(len(truth)==18 and len(proof)==18 and len(X)==len(rr)==len(Ynew)==len(onsetnew)
         and int(onsetnew.sum())>1500 and int(Ynew.sum())>1700,
         "S562 real 18 source positive note-on labels or physical features invalid")
    return raw,X,rr,Ynew,onsetnew,truth,proof,meta

def train(X,Y,onset,rec,meta,held,arch,full=False):
    from sklearn.ensemble import ExtraTreesClassifier
    allids=set(meta)
    if full:
        exclude=set()
    else:
        held_composer=composer_components(meta[held]["composer"])
        exclude={k for k,v in meta.items() if composer_components(v["composer"])&held_composer}
    eligible=~np.isin(rec,list(exclude))
    ids=np.flatnonzero(eligible & onset)
    included=set(rec[ids])
    need((full and len(included)==18) or
         (not full and held not in included and len(included)>=12),
         "S562 strict actual music composers must be held away from 18-source training")
    if not full:
        need(all(not (composer_components(meta[k]["composer"])&
                      composer_components(meta[held]["composer"])) for k in included),
             "S562 compositional music track leakage")
    xx=X[ids].reshape(-1,X.shape[-1])
    yy=Y[ids].reshape(-1)
    need(int(yy.sum())>500 and int(np.sum(yy==0))>5000,
         "S562 musical piano source training true and false note sufficient")
    model=ExtraTreesClassifier(n_estimators=112 if not full else 192,
          max_depth=16 if arch=="raw7" else 19,
          min_samples_leaf=4 if arch=="raw7" else 6,
          max_features=.8,class_weight="balanced_subsample",
          n_jobs=2,random_state=56201 if arch=="raw7" else 56202)
    model.fit(xx,yy)
    return model,sorted(included),sorted(exclude),int(yy.sum())

def predict(raw,rec,prob,ident):
    ids=np.flatnonzero(rec==ident)
    original=[]
    for i in ids:
        pp=proposed_notes(raw[i],dict(raw_only_49_pitch_prob=prob[i]),"raw_only:0.7")
        if pp:
            original.append(dict(time=float(raw[i]["original_V88_candidate_start_s"]),notes=sorted(pp)))
    events=[(raw[i],dict(raw_only_49_pitch_prob=prob[i])) for i in ids]
    after,_=dedup100(events,dict(dedup_ms=100,harmonic_ratio=.8,rearticulation_ratio=0))
    return original,after

def run(a):
    need(not a.out.exists(),"S562 full trained source head weights cannot be overwritten after first fit")
    raw,X0,rec,Y,onset,truth,proof,meta=recover(a)
    raw,X0,rec,Y,onset,truth,proof,meta=extension(a,meta,raw,rec,Y,onset,truth,proof)
    a.out.mkdir(parents=True)
    import joblib
    variants=("raw7","competing_fundamental34")
    Xs={n:acoustic(X0,"raw7" if n=="raw7" else "competing_fundamental34") for n in variants}
    oof={n:np.full((len(raw),49),np.nan,np.float32) for n in variants}
    models=[];stats={}
    for n in variants:
        feat=Xs[n]
        for k,held in enumerate(sorted(meta)):
            model,included,excluded,positive=train(feat,Y,onset,rec,meta,held,n)
            ids=np.flatnonzero(rec==held)
            q=model.predict_proba(feat[ids].reshape(-1,feat.shape[-1]))[:,list(model.classes_).index(1)]
            oof[n][ids]=q.reshape(len(ids),49)
            del model
        need(np.isfinite(oof[n]).all(),"S562 every real source must have held music predictions")
        count=collections.Counter()
        sources={}
        for ident,gt in sorted(truth.items()):
            rawpred,filtered=predict(raw,rec,oof[n],ident)
            unfiltered=score_event_notes(rawpred,gt)
            clean=score_event_notes(filtered,gt)
            sources[ident]=dict(raw=unfiltered,filtered=clean,
                 composer=meta[ident]["composer"])
            count.update(TP=clean["TP"],FP=clean["FP"],FN=clean["FN"])
        c=dict(count);c["F1"]=2*c["TP"]/max(1,2*c["TP"]+c["FP"]+c["FN"])
        stats[n]=dict(summary=c,per_source=sources)
        print("S562 18 REAL SONG STRICT COMPOSER HELD MUSIC NOTEON TEST",json.dumps(
            dict(architecture=n,TP=c["TP"],FP=c["FP"],FN=c["FN"],F1=c["F1"])),flush=True)
        full,trained,excluded,positives=train(feat,Y,onset,rec,meta,None,n,True)
        fp=a.out/f"S562_FULL18_{n}_REAL_COMPOSER_DISJOINT_HELD_SOURCE_FIT.joblib"
        joblib.dump(full,fp,compress=3)
        models.append(dict(architecture=n,
              full_source_training_without_any_S561_audio_or_MIDI=True,
              actual_full_training_sources=trained,
              number=18,source_sha_model=sha(fp),
              trained_true_MIDI_note_labeled_sites=positives,
              feature_dimensions=feat.shape[-1]))
    report=dict(status="S562_REAL_EIGHTEEN_EXPOSED_COMPOSER_SOURCE_AUDIO_TRAINED_NOTE_PITCH_TWO_HEADS",
        original_historical_source_recordings=18,
        uses_exposed_S545_and_S554_TRUE_MIDI_for_development=True,
        previous_18_source_LOCO_results_are_RETROSPECTIVE_not_blind=True,
        composer_and_co_composer_disjoint_folds=True,
        original_original_V88_all_K0_candidates_kept=True,
        true_MIDI40to88_real_note_pitch_attack_labels=True,
        fully_trained_two_models_before_ANY_S561_MIDI=True,
        all_fitted_checkpoints_SHA256=models,
        two_modes_predeclared=list(variants),
        architecture_uses_extra_physical_harmonic_competition_not_only_vote=True,
        source_true_note_attack_groups=sum(len(x) for x in truth.values()),
        source_positive_attacks=int(onset.sum()),
        no_S561_audio_or_MIDI_opened_for_fitting=True,
        original_S80_GuitarSet_Exact_K_Live_stays_unchanged=True,
        no_promotion=True,development_OOF_scores=stats)
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")

def selftest():
    a=np.zeros((2,49,11),np.float32)
    a[0,12,2]=.9;a[0,0,2]=.5
    x=acoustic(a,"competing_fundamental34")
    need(x.shape==(2,49,34) and np.isfinite(x).all(),
         "S562 real physical neighbor and harmonic waveform features")
    need(acoustic(a,"raw7").shape==(2,49,7),"S562 raw baseline preserved")
    print("S562 SELFTEST PASS raw7 and competing fundamental physical34 preserved; no MIDI")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for n in ("old-manifest","new-manifest","old-midi-report","new-midi-report",
              "old-features","old-feature-report","new-features","new-feature-report",
              "s545-manifest","s545-midi-report","s545-features","s545-feature-report",
              "s554-manifest","s554-midi-report","s554-features","s554-feature-report","out"):
        p.add_argument("--"+n,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for n in ("old_manifest","new_manifest","old_midi_report","new_midi_report",
              "old_features","old_feature_report","new_features","new_feature_report",
              "s545_manifest","s545_midi_report","s545_features","s545_feature_report",
              "s554_manifest","s554_midi_report","s554_features","s554_feature_report","out"):
            need(getattr(a,n) is not None,"S562 missing --"+n)
        run(a)
