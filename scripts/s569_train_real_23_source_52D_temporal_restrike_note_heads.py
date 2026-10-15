"""S569 train 23-source real-restrike vs ghost-harmonic temporal pitch heads.

Only PREVIOUSLY evaluated real WAV/MIDI from S469/S537/S545/S554/S561
as supervised DEVELOPMENT labels. The next held-out cohort S570 is neither
selected nor opened while these 23 genuine MIDI sources are fitted.

The original 49x11 MIDI-blind spectral source observations and original
V88 onset candidates are kept. New 52D per-note head = 34D S562
physical harmonic/attack/sustain contrast + 18 genuine neighboring
time-indexed observations of same MIDI pitch at preceding V88 times.
All fields derived from WAV/BasicPitch frozen outputs, NEVER from true MIDI.
New note / repeated pitch at <100 ms remains a learnable target; never
label repeated genuine MIDI as a duplicate.

Two fixed architectures:
 A - positive-matched onset sites (S562 compatible)
 B - matched onset + audio candidate false-event hard negatives, downweighted
     to avoid assuming every extra source event was necessarily silent.
Both 23 full-source models and their SHA are made before any new S570 MIDI.
Actual leave-composer-and-co-composer-out cross-fitted full stream NoteOn F1,
no reference, no official S80/Live change.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s562_train_18_composer_true_pitch_spectral_fundamental_heads import (
    recover,extension,acoustic,readrows)
from scripts.s546_train_8_source_folds_excluding_SHARED_COMPOSER_COMPONENTS import composer_components
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import (
    fetch_midi,midi_events,onset_groups,match_times)
from scripts.s476_compare_MAESTRO_newnote_K_vs_sounding_K_same_V88 import new_notes
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import need,sha,score_event_notes
from scripts.s536_onset_burst_adaptive_restrike_dedup_loops import emit as dedup100

POLICIES=("attack_positive_sites","attack_and_ghost_events")
def temporal52(raw):
    """Deterministic per-source causal previous-V88-pitch acoustic temporal input."""
    X=np.asarray([r["per_midi_features_49x11"] for r in raw],np.float32)
    need(X.ndim==3 and X.shape[1:]==(49,11) and np.isfinite(X).all(),
         "S569 authentic real WAV note rows all 49x11 physical input required")
    n=len(raw);prev=np.zeros((n,49,4),np.float32)
    dif=np.zeros_like(prev);two=np.zeros((n,49,2),np.float32)
    lag=np.zeros((n,49,2),np.float32);ratio=np.zeros((n,49,2),np.float32)
    recent=collections.defaultdict(list)
    for i,r in enumerate(raw):
        k=r["source_recording_sha"]
        recent[k].append((float(r["original_V88_candidate_start_s"]),str(r["V88_original_event_id"]),i))
    for events in recent.values():
        events.sort(key=lambda z:(z[0],z[1]))
        for pos,(time,_,i) in enumerate(events):
            if pos==0:continue
            t,j=events[pos-1][0],events[pos-1][2]
            dt=time-t
            need(dt>=0,"S569 source local audio events not ordered")
            if dt>.350:continue
            prev[i]=X[j][:,[2,3,4,5]]
            dif[i]=X[i][:,[2,3,4,5]]-prev[i]
            lag[i,:,0]=np.exp(-dt/.080)
            lag[i,:,1]=np.exp(-dt/.250)
            ratio[i,:,0]=np.clip((X[i,:,2]+.01)/(X[j,:,2]+.01),0,8)
            ratio[i,:,1]=np.clip((X[i,:,5]+.01)/(X[j,:,5]+.01),0,8)
            if pos>=2 and time-events[pos-2][0]<=.350:
                jj=events[pos-2][2]
                two[i]=X[jj][:,[2,5]]
    out=np.concatenate((acoustic(X,"competing_fundamental34"),
           prev,dif,two,lag,ratio,X[:,:,[6,7,8,9]]),axis=2).astype(np.float32)
    need(out.shape==(n,49,52) and np.isfinite(out).all(),
         "S569 per MIDI note real wave 34+18 physical temporal features ABI changed")
    return out

def add_five(a,meta,raw,Y,onsets,truth,proof):
    man=json.loads(a.s561_manifest.read_text())
    verified=json.loads(a.s561_midi_report.read_text())
    frep=json.loads(a.s561_feature_report.read_text())
    need(man["status"]=="S561_PREREGISTERED_FIVE_MORE_TRAIN_SPLIT_COMPOSERS_AFTER_S544_NO_WAV_MIDI"
       and man["original_previous_S469_3_S537_5_S545_5_S554_5_component_excluded"]
       and verified["status"]=="S567_EXTERNAL_POSTFROZEN_FIVE_NEW_COMPOSER_FULL_MIDI_NOTEON_F1"
       and verified["S566_all_seven_named_note_policies_SHA_VERIFIED_before_true_MIDI"]
       and verified["all_five_original_MIDI_used_ONLY_in_S567_scoring"]
       and frep["status"]=="S565_PRELABEL_FIVE_NEW_PIANO_ALL_V88_RAW_49_PITCH_FEATURES"
       and frep["frozen_feature_ledger_SHA256"]==sha(a.s561_features),
       "S569 only genuine previously validated S561 source data may become supervised training")
    midi_sha={x["original_audio_filename_sha"]:x for x in verified["true_MIDI_provenance"]}
    nold=len(raw)
    newrows=readrows(a.s561_features)
    need(len(midi_sha)==5 and len({r["source_recording_sha"] for r in newrows})==5,
         "S569 must add five fully audio-frozen S561 recordings")
    for item in man["chosen_metadata_only"]:
        ident=item["filename_SHA256"]
        need(ident not in meta and ident in midi_sha,
             "S569 no S561 music may appear in original 18 composer training twice")
        p=midi_sha[ident]
        meta[ident]=dict(composer=item["composer"],url=man["source_audio_mirror"],
            midi=item["MIDI_filename_path_ONLY_not_opened"],
            offset=float(item["audio_excerpt_start_seconds"]),
            duration=float(item["audio_excerpt_duration_seconds"]),
            midi_sha=p["original_MIDI_sha256"],
            expected_groups=p["original_MIDI_onset_groups"],old=False)
    raw.extend(newrows)
    Ynew=np.zeros((len(raw),49),np.int8);Ynew[:len(Y)]=Y
    Onew=np.zeros(len(raw),bool);Onew[:len(onsets)]=onsets
    rec=np.asarray([x["source_recording_sha"] for x in raw],str)
    for ident in sorted(midi_sha):
        m=meta[ident]
        blob=fetch_midi(m["url"],m["midi"])
        need(hashlib.sha256(blob).hexdigest()==m["midi_sha"],
             "S569 true MIDI of previously prospectively evaluated source is not SHA-stable")
        events=midi_events(blob)
        g=onset_groups(events,m["offset"],m["duration"])
        need(len(g)==m["expected_groups"],"S569 genuine S561 all true NoteOns cannot be omitted")
        gt=[dict(time=float(q["onset"]),notes=[int(p) for p in new_notes(
            events,m["offset"],float(q["onset"])) if 40<=int(p)<=88]) for q in g]
        gt=[r for r in gt if r["notes"]]
        truth[ident]=gt
        ii=np.flatnonzero(rec==ident)
        candidate=np.asarray([float(raw[i]["original_V88_candidate_start_s"]) for i in ii])
        pair=match_times(candidate,np.asarray([q["time"] for q in gt]),.080)
        for i,k,dt in pair:
            idx=int(ii[i])
            need(not Onew[idx],"S569 one source musical onset must be aligned only once")
            Onew[idx]=True
            for p in gt[k]["notes"]:Ynew[idx,int(p)-40]=1
        proof[ident]=dict(composer=m["composer"],V88_events=len(ii),
                          true_MIDI_onset_groups=len(gt),
                          true_MIDI_note_pitches=sum(len(z["notes"]) for z in gt),
                          MIDI_SHA=m["midi_sha"],positive_aligned_events=len(pair))
    need(len(meta)==23 and len(truth)==23 and len(proof)==23
         and len({r["source_recording_sha"] for r in raw})==23 and len(raw)>nold
         and int(Onew.sum())>1600 and int(Ynew.sum())>1900,
         "S569 23 true source aligned MIDI note-on labels/physical rows missing")
    return rec,Ynew,Onew

def true_restrike_count(truth,sec=.100):
    count=0
    for gt in truth.values():
        last={}
        for g in sorted(gt,key=lambda z:z["time"]):
            for p in set(map(int,g["notes"])):
                if p in last and 0<float(g["time"])-last[p]<=sec:
                    count+=1
                last[p]=float(g["time"])
    return count

def fit(features,labels,onsets,rec,meta,held,policy):
    from sklearn.ensemble import ExtraTreesClassifier
    held_components=composer_components(meta[held]["composer"]) if held else set()
    banned={k for k,m in meta.items()
            if composer_components(m["composer"])&held_components}
    good=~np.isin(rec,list(banned))
    if policy=="attack_positive_sites":
        good&=onsets
    chosen=np.flatnonzero(good)
    sources=set(rec[chosen])
    need((held is None and len(sources)==23) or
         (held is not None and held not in sources and len(sources)>=19),
         "S569 held composer/co-composer cannot appear in training")
    if held:
        need(not any(composer_components(meta[k]["composer"])&held_components
                     for k in sources),"S569 composite composer identity leakage")
    xx=features[chosen].reshape(-1,52)
    yy=labels[chosen].reshape(-1)
    need(int(yy.sum())>600 and np.sum(yy==0)>5000,
         "S569 cannot train real pitches without MIDI positives and hard negatives")
    model=ExtraTreesClassifier(n_estimators=96 if held else 180,
         max_depth=19,min_samples_leaf=4,max_features=.8,
         class_weight="balanced_subsample",n_jobs=2,
         random_state=56901 if policy=="attack_positive_sites" else 56902)
    weights=np.repeat(np.where(onsets[chosen],1.,.3),49)
    model.fit(xx,yy,sample_weight=weights)
    return model,sorted(sources),sorted(banned),int(yy.sum())

def evaluate(raw,prob,rec,truth,metadata):
    sumall=collections.Counter();byrecord={}
    for ident,real in sorted(truth.items()):
        ii=np.flatnonzero(rec==ident)
        events=[(raw[j],dict(raw_only_49_pitch_prob=prob[j])) for j in ii]
        guess,_=dedup100(events,dict(dedup_ms=100,harmonic_ratio=.8,rearticulation_ratio=0))
        score=score_event_notes(guess,real)
        byrecord[ident]=dict(composer=metadata[ident]["composer"],result=score)
        sumall.update({k:score[k] for k in ("TP","FP","FN")})
    total=dict(sumall)
    total["F1"]=2*total["TP"]/max(1,2*total["TP"]+total["FP"]+total["FN"])
    return total,byrecord

def run(a):
    need(not a.out.exists(),"S569 immutable fitted MIDI/restrike source model directory")
    raw,X,rec,Y,O,truth,proof,meta=recover(a)
    raw,X,rec,Y,O,truth,proof,meta=extension(a,meta,raw,rec,Y,O,truth,proof)
    rec,Y,O=add_five(a,meta,raw,Y,O,truth,proof)
    Xt=temporal52(raw)
    need(Xt.shape==(len(raw),49,52),"S569 temporal pitch physical true wave bank failed")
    a.out.mkdir(parents=True)
    import joblib
    stats={};models=[]
    for policy in POLICIES:
        p=np.full((len(raw),49),np.nan,np.float32)
        for ident in sorted(meta):
            forest,included,banned,truecount=fit(Xt,Y,O,rec,meta,ident,policy)
            ii=np.flatnonzero(rec==ident)
            pos=list(forest.classes_).index(1)
            p[ii]=forest.predict_proba(Xt[ii].reshape(-1,52))[:,pos].reshape(len(ii),49)
            del forest
        need(np.isfinite(p).all() and np.all((p>=0)&(p<=1)),
             "S569 source-held all MIDI40..88 note probabilities must exist")
        total,song=evaluate(raw,p,rec,truth,meta)
        stats[policy]=dict(summary=total,by_song=song)
        print("S569 CAUSAL TEMPORAL RESONANCE AND REAL RESTRIKE NOTE MODEL",json.dumps(
            dict(architecture=policy,**total,
                 real_same_pitch_restrikes_under_100ms=true_restrike_count(truth))),flush=True)
        full,trainIDs,excl,positives=fit(Xt,Y,O,rec,meta,None,policy)
        file=a.out/f"S569_TWENTYTHREE_REAL_SOURCE_TRAINED_{policy}_52D.joblib"
        joblib.dump(full,file,compress=3)
        models.append(dict(architecture=policy,checkpoint_filename=file.name,
            checkpoint_SHA=sha(file),trained_on_exact_23_exposed_MAESTRO_real_MIDI_sources=True,
            no_future_MIDI_used_in_checkpoint_training=True,training_sources=trainIDs,
            original_features_per_MIDI_pitch=52,observed_positive_notes=positives))
    report=dict(status="S569_TWENTYTHREE_SOURCE_RESTRIKE_HARMONIC_GHOST_TEMPORAL_HEADS_REAL_MIDI_TRAINED",
        real_23_music_source_train_and_source_composer_held_folds=True,
        temporal_52_acoustic_features_from_prior_actual_WAV_events_ONLY=True,
        original_34_physical_onset_harmonic_head_plus_18_temporal_strike_features=True,
        two_predeclared_models=list(POLICIES),
        authentic_MIDI40_88_1to1_positive_labels=True,
        original_49x11_still_frozen_from_WAV_before_corresponding_MIDI=True,
        true_genuine_same_pitch_fast_restrikes_100ms=true_restrike_count(truth),
        no_exposed_MIDI_note_was_blind_in_23_source_crossvalidation=True,
        metrics_are_retrospective_not_new_source_test=True,
        future_new_composers_NEVER_consulted_for_training=True,
        no_reference_promotion=True,
        original_S80_ExactK_freeze_local_combo_Live_unchanged=True,
        checkpoints_SHA=models,crossfit_fullstream_noteon_F1=stats,
        source_counts=dict(number=23,events=len(raw),positive_attack_candidates=int(O.sum()),
                           MIDI_true_note_onsets=sum(sum(len(x["notes"]) for x in gt) for gt in truth.values())))
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
def selftest():
    def row(rec,t,s,nov,birth=0):
        x=np.zeros((49,11),np.float32)
        x[20,2]=s;x[20,3]=s*.7;x[20,5]=nov;x[20,4]=.4
        x[20,6]=birth
        return dict(source_recording_sha=rec,original_V88_candidate_start_s=t,
          V88_original_event_id=f"{rec}{t}",per_midi_features_49x11=x.tolist())
    a=[row("one",.1,.8,.5),row("one",.15,.95,.7,1),row("two",.14,.3,.2)]
    X=temporal52(a)
    need(X.shape==(3,49,52) and X[1,20,34]>.0 and X[2,20,34]==0.
         and X[1,20,38]>0 and X[1,20,48]==1.,
         "S569 strictly source-separated causal temporal acoustic and BP reattack features")
    rev=temporal52(list(reversed(a)))
    need(np.allclose(X[0],rev[2]) and np.allclose(X[1],rev[1])
         and np.allclose(X[2],rev[0]),
         "S569 audio previous-note temporal features cannot depend on row order")
    print("S569 SELFTEST PASS temporal 52 acoustic pitch real reattack candidate features without MIDI")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for name in ("old-manifest","new-manifest","old-midi-report","new-midi-report",
       "old-features","old-feature-report","new-features","new-feature-report",
       "s545-manifest","s545-midi-report","s545-features","s545-feature-report",
       "s554-manifest","s554-midi-report","s554-features","s554-feature-report",
       "s561-manifest","s561-midi-report","s561-features","s561-feature-report","out"):
        p.add_argument("--"+name,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for key in ("old_manifest","new_manifest","old_midi_report","new_midi_report",
            "old_features","old_feature_report","new_features","new_feature_report",
            "s545_manifest","s545_midi_report","s545_features","s545_feature_report",
            "s554_manifest","s554_midi_report","s554_features","s554_feature_report",
            "s561_manifest","s561_midi_report","s561_features","s561_feature_report","out"):
            need(getattr(a,key)is not None,"S569 missing --"+key)
        run(a)
