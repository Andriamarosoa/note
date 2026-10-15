"""S578: 28-source genuine middle-register NoteOn specialist vs 28-source global control.

Source 28 previously evaluated MAESTRO tracks (S469,537,545,554,561,570).
No held S577 WAV/MIDI read or used to choose weights. True source-held
COMPOSER BLOCKED 5-fold evaluation, same exact full NoteOn F1 for each head.

Pitches 60-71 get a binary specialist physical temporal 52D model trained
on true MIDI attacks, with baseline full 49-pitch global 52D forest trained
on identical 28 sources and same source folds. Three fixed policies:
- all MIDI40..88 general model;
- replace only MIDI60..71 probability with specialist;
- mid specialist/global average 50:50.
Original BP-born notes protected; fixed threshold 0.7 and 100ms+harmony
unchanged. Score *every* real note, even if V88 missed onset, and all FP.
"""
from __future__ import annotations
import argparse,hashlib,json,collections
from pathlib import Path
import numpy as np
from scripts.s569_train_real_23_source_52D_temporal_restrike_note_heads import (
    recover,extension,add_five,temporal52)
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import (
    fetch_midi,midi_events,onset_groups,match_times)
from scripts.s476_compare_MAESTRO_newnote_K_vs_sounding_K_same_V88 import new_notes
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import (
    need,sha,score_event_notes)
from scripts.s536_onset_burst_adaptive_restrike_dedup_loops import emit as dedup
from scripts.s546_train_8_source_folds_excluding_SHARED_COMPOSER_COMPONENTS import composer_components

POLICIES=("global28","specialist_mid_replace","specialist_mid_blend50")
def load_fifth(a,meta,raw,rec,Y,O,truth,proof):
    m=json.loads(a.s570_manifest.read_text())
    validated=json.loads(a.s570_midi_report.read_text())
    f=json.loads(a.s570_feature_report.read_text())
    need(m["status"]=="S570_PREREGISTERED_FIVE_MORE_TRAIN_SPLIT_COMPOSERS_AFTER_S544_NO_WAV_MIDI"
         and m["original_previous_S469_3_S537_5_S545_5_S554_5_S561_5_component_excluded"]
         and m["five_next_distinct_composers"]==5
         and validated["status"]=="S575_EXTERNAL_POSTFROZEN_FIVE_NEW_COMPOSER_FULL_MIDI_NOTEON_F1"
         and validated["S574_all_seven_named_note_policies_SHA_VERIFIED_before_true_MIDI"]
         and f["status"]=="S573_PRELABEL_FIVE_NEW_PIANO_ALL_V88_RAW_49_PITCH_FEATURES"
         and f["frozen_feature_ledger_SHA256"]==sha(a.s570_features),
         "S578 only 5 previously validated original S570 sources may join MIDI supervised training")
    from scripts.s569_train_real_23_source_52D_temporal_restrike_note_heads import readrows
    newrows=readrows(a.s570_features)
    proofs={x["original_audio_filename_sha"]:x for x in validated["true_MIDI_provenance"]}
    need(len(proofs)==5 and len({r["source_recording_sha"] for r in newrows})==5,
         "S578 all 5 original S570 source physical note sets required")
    for item in m["chosen_metadata_only"]:
        key=item["filename_SHA256"]
        need(key not in meta and key in proofs,"S578 28 training sources unique")
        x=proofs[key]
        meta[key]=dict(composer=item["composer"],url=m["source_audio_mirror"],
            midi=item["MIDI_filename_path_ONLY_not_opened"],
            offset=float(item["audio_excerpt_start_seconds"]),
            duration=float(item["audio_excerpt_duration_seconds"]),
            midi_sha=x["original_MIDI_sha256"],
            expected_groups=x["original_MIDI_onset_groups"],old=False)
    old=len(raw)
    raw.extend(newrows)
    Ynext=np.zeros((len(raw),49),np.int8);Ynext[:len(Y)]=Y
    Onext=np.zeros(len(raw),bool);Onext[:len(O)]=O
    rec=np.asarray([r["source_recording_sha"] for r in raw],str)
    for src in sorted(proofs):
        data=meta[src]
        midi_data=fetch_midi(data["url"],data["midi"])
        need(hashlib.sha256(midi_data).hexdigest()==data["midi_sha"],
             "S578 frozen official S575 MIDI SHA must remain unchanged")
        midi=midi_events(midi_data)
        groups=onset_groups(midi,data["offset"],data["duration"])
        need(len(groups)==data["expected_groups"],"S578 100% original audio MIDI attacks")
        gt=[dict(time=float(q["onset"]),
                 notes=[int(p) for p in new_notes(midi,data["offset"],float(q["onset"]))
                        if 40<=int(p)<=88]) for q in groups]
        gt=[x for x in gt if x["notes"]]
        truth[src]=gt
        ii=np.flatnonzero(rec==src)
        site=np.asarray([float(raw[j]["original_V88_candidate_start_s"]) for j in ii])
        matches=match_times(site,np.asarray([v["time"] for v in gt]),.08)
        for i,k,_ in matches:
            ix=int(ii[i])
            need(not Onext[ix],"S578 one V88 onset site cannot match two true MIDI onsets")
            Onext[ix]=True
            for pitch in gt[k]["notes"]:Ynext[ix,pitch-40]=1
        proof[src]=dict(composer=data["composer"],V88_events=len(ii),
            true_MIDI_onset_groups=len(gt),
            true_MIDI_note_pitches=sum(len(z["notes"]) for z in gt),
            matched_attack_events=len(matches),midi_sha=data["midi_sha"])
    need(len(meta)==28 and len(truth)==28 and len(proof)==28 and len(raw)>old
         and len({r["source_recording_sha"] for r in raw})==28,
         "S578 all 28 original music sources + labels valid")
    return rec,Ynext,Onext

def component_folds(meta,folds=5):
    # Exact connected components by composer / co-composer; never split
    # any connected artist into training and holdout across folds.
    keys=sorted(meta)
    groups=[];unused=set(keys)
    while unused:
        seed=min(unused);current={seed};artist=composer_components(meta[seed]["composer"])
        changed=True
        while changed:
            changed=False
            for k in sorted(unused-current):
                a=composer_components(meta[k]["composer"])
                if a&artist:current.add(k);artist.update(a);changed=True
        unused-=current;groups.append(sorted(current))
    # deterministic assignment without consulting any notes/results
    groups.sort(key=lambda g:(-len(g),hashlib.sha256(";".join(g).encode()).hexdigest()))
    blocks=[[] for _ in range(folds)]
    for group in groups:
        i=min(range(folds),key=lambda z:(len(blocks[z]),z))
        blocks[i].extend(group)
    need(all(blocks) and sorted(x for b in blocks for x in b)==keys,
         "S578 all 28 source groups must have five composer-disjoint folds")
    for i,b in enumerate(blocks):
        comp=set().union(*(composer_components(meta[x]["composer"]) for x in b))
        other=set().union(*(composer_components(meta[x]["composer"]) for j,bb in enumerate(blocks) if j!=i for x in bb))
        need(not(comp&other),"S578 co-composer identity leaked across OOF folds")
    return blocks

def train_models(X,Y,O,rec,sources,held=None):
    from sklearn.ensemble import ExtraTreesClassifier
    allowed=np.isin(rec,sources)&O
    ids=np.flatnonzero(allowed)
    need(len(ids)>100 and int(Y[ids].sum())>1000,
         "S578 genuine MIDI positive attack sites must be present")
    g=ExtraTreesClassifier(n_estimators=135 if held else 210,
         max_depth=19,min_samples_leaf=4,max_features=.8,
         class_weight="balanced_subsample",random_state=57801,n_jobs=2)
    g.fit(X[ids].reshape(-1,52),Y[ids].reshape(-1))
    mid=ExtraTreesClassifier(n_estimators=130 if held else 200,
         max_depth=19,min_samples_leaf=3,max_features=.8,
         class_weight="balanced_subsample",random_state=57802,n_jobs=2)
    mid.fit(X[ids,20:32,:].reshape(-1,52),Y[ids,20:32].reshape(-1))
    need(1 in g.classes_ and 1 in mid.classes_,"S578 both true pitch heads have positives")
    return g,mid
def emission(raw,rec,prob,ident):
    ii=np.flatnonzero(rec==ident)
    events=[(raw[i],dict(raw_only_49_pitch_prob=prob[i].tolist())) for i in ii]
    out,_=dedup(events,dict(dedup_ms=100,harmonic_ratio=.8,rearticulation_ratio=0))
    return out
def policies(global49,mid12):
    need(global49.shape[1]==49 and mid12.shape==(len(global49),12),
         "S578 only middle MIDI60..71 can be routed through specialist")
    one=global49.copy();one[:,20:32]=mid12
    half=global49.copy();half[:,20:32]=.5*(global49[:,20:32]+mid12)
    return dict(zip(POLICIES,(global49,one,half)))
def run(a):
    need(not a.out.exists(),"S578 28 music source trained MIDI specialist artifact immutable")
    raw,X,rec,Y,O,truth,proof,meta=recover(a)
    raw,X,rec,Y,O,truth,proof,meta=extension(a,meta,raw,rec,Y,O,truth,proof)
    rec,Y,O=add_five(a,meta,raw,Y,O,truth,proof)
    rec,Y,O=load_fifth(a,meta,raw,rec,Y,O,truth,proof)
    X=temporal52(raw)
    blocks=component_folds(meta)
    preds={p:np.full((len(raw),49),np.nan,np.float32) for p in POLICIES}
    for i,held in enumerate(blocks):
        other=sorted(set(meta)-set(held))
        g,m=train_models(X,Y,O,rec,other,held=held)
        ii=np.flatnonzero(np.isin(rec,held))
        p=g.predict_proba(X[ii].reshape(-1,52))[:,list(g.classes_).index(1)].reshape(len(ii),49)
        pm=m.predict_proba(X[ii,20:32,:].reshape(-1,52))[:,list(m.classes_).index(1)].reshape(len(ii),12)
        for name,q in policies(p,pm).items():preds[name][ii]=q
        print("S578 HELD CO-COMPOSER BLOCK",json.dumps(dict(fold=i,held_music=len(held),
             train_music=len(other),candidate_audio_sites=len(ii),midi_mid_true=int(Y[ii,20:32].sum()))),flush=True)
    summary={}
    for name,p in preds.items():
        need(np.isfinite(p).all() and p.shape==(len(raw),49) and
             np.all((p>=0)&(p<=1)),"S578 actual source-held 49 pitch probabilities")
        agg=collections.Counter();songs={}
        for ident,gt in sorted(truth.items()):
            predicted=emission(raw,rec,p,ident)
            score=score_event_notes(predicted,gt)
            songs[ident]=dict(composer=meta[ident]["composer"],metrics=score)
            agg.update({key:score[key] for key in ("TP","FP","FN")})
        tot=dict(agg);tot["F1"]=2*tot["TP"]/max(1,2*tot["TP"]+tot["FP"]+tot["FN"])
        summary[name]=dict(metrics=tot,per_song=songs)
        print("S578 GENUINE 28-COMPOSER-BLOCK 5FOLD NOTEON",
              json.dumps(dict(architecture=name,**tot)),flush=True)
    a.out.mkdir(parents=True)
    import joblib
    g,m=train_models(X,Y,O,rec,sorted(meta))
    checkpoints=[]
    for label,model in (("general28",g),("mid12",m)):
        f=a.out/f"S578_FULL28_{label}_52D_TRUE_AUDIO_NOTE_MODEL.joblib"
        joblib.dump(model,f,compress=3)
        checkpoints.append(dict(architecture=label,filename=f.name,sha=sha(f),
             source_count=28,source_composers=sorted(meta),
             original_52D_audio_compatible=True,true_original_MIDI_note_targets=True))
    report=dict(status="S578_TWENTYEIGHT_TRUE_PIANO_SOURCE_MIDDLE_MIDI_60_71_EXPERT_VS_GLOBAL_NOTEON",
      previous_28_recordings_original_MIDI_known_for_training=True,
      original_source_WAV_49x11_and_temporal52_features_from_audio_ONLY=True,
      composer_and_co_composer_connected_components_five_held_folds=True,
      held_groups=blocks,
      one_real_source_training_28_original_music_models=True,
      actual_28_midi_pitch_sources=len(meta),
      sum_actual_true_MIDI_noteon=sum(sum(len(x["notes"]) for x in gt) for gt in truth.values()),
      middleregister_true_aligned_pitch_sites=int(Y[:,20:32].sum()),
      old_real_BP_born_notes_protected=True,
      source_MIDI_true_genuine_repeat_restrikes_not_labeled_negative=True,
      fixed_threshold_070_and_dedup100_harm080_no_tuning=True,
      three_predeclared_musical_policies=list(POLICIES),
      no_future_S577_MIDI_or_WAV_read_for_training=True,
      weights_before_S577_exposure=checkpoints,
      development_crossfit_is_NOT_NEW_PROSPECTIVE_TEST=True,
      original_reference_S80_Live_ExactK_unchanged=True,no_promotion=True,
      full_noteon_evaluations=summary)
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
def selftest():
    x=np.full((2,49),.75,np.float32)
    y=np.full((2,12),.11,np.float32)
    q=policies(x,y)
    need(np.array_equal(q["global28"],x) and
         np.allclose(q["specialist_mid_replace"][:,20:32],.11) and
         np.allclose(q["specialist_mid_blend50"][:,20:32],.43) and
         np.array_equal(q["specialist_mid_replace"][:,:20],x[:,:20]),
         "S578 middle expert cannot change any MIDI pitch outside 60..71")
    items={"a":dict(composer="Liszt / Schubert"),
           "b":dict(composer="Schubert"),"c":dict(composer="Bach"),
           "d":dict(composer="Haydn"),"e":dict(composer="Mozart"),
           "f":dict(composer="Chopin")}
    folds=component_folds(items)
    need(any(set(["a","b"]).issubset(set(f)) for f in folds),
         "S578 shared co-composers must remain in same source fold")
    print("S578 SELFTEST PASS true fixed 3 routes, blocked artists, no target MIDI")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for key in ("old-manifest","new-manifest","old-midi-report","new-midi-report",
       "old-features","old-feature-report","new-features","new-feature-report",
       "s545-manifest","s545-midi-report","s545-features","s545-feature-report",
       "s554-manifest","s554-midi-report","s554-features","s554-feature-report",
       "s561-manifest","s561-midi-report","s561-features","s561-feature-report",
       "s570-manifest","s570-midi-report","s570-features","s570-feature-report","out"):
        p.add_argument("--"+key,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for key in ("old_manifest","new_manifest","old_midi_report","new_midi_report",
          "old_features","old_feature_report","new_features","new_feature_report",
          "s545_manifest","s545_midi_report","s545_features","s545_feature_report",
          "s554_manifest","s554_midi_report","s554_features","s554_feature_report",
          "s561_manifest","s561_midi_report","s561_features","s561_feature_report",
          "s570_manifest","s570_midi_report","s570_features","s570_feature_report","out"):
            need(getattr(a,key) is not None,"S578 missing "+key)
        run(a)
