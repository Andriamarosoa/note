"""S700 train a NEW strictly causal K0..K6 shadow head with untouched NOTE events.

Native K classifier is frozen. The new head learns from true GT counts ONLY on
GuitarSet TRAIN composition groups, combining past/current native K logits,
native current/past sustain/attack evidence, physical fundamental spectra,
and native already-emitted MIDI count. It never reads future PCM or GT at
inference. First five TRAIN compositions train the head; four different
TRAIN compositions select from fixed candidate strategies; DEV is an
independent historically-exposed diagnostic (not promotional evidence).
All original MIDI NoteOn/NoteOff events and S675 model remain unchanged.

Two separate thresholds: (a) 80% total K frames, (b) 80% active
(non-silence) K frames. No majority-silence inflation is permitted.
"""
from __future__ import annotations
import argparse,json,tempfile,time,zipfile
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from causal_note.strict_live import Selection,load_heads,feature_track,HOP,SR
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import decode,real_notes,event_score
from scripts.s672_causal_offset_timing_grid import ROOT,MODEL,raw_guitar_audio,native_posteriors
from scripts.s696_direct_K_causal_audit import (
    native_count_timeline,gt_count_timeline,posterior_ema,score
)
POLICIES=("native_decoded","frozen_direct_K",
          "learned","fuse_0.30","fuse_0.55","fuse_0.75")
TRAIN_ROW_STRIDE=3
def expanded_train_specs(original, archive, annotation_zip):
    """Use additional authentic players 00..03 ONLY on preexisting TRAIN
    composition groups. Never use player 04 DEV or player 05 TEST.
    Reject accidental cross-composition exposure.
    """
    wav_names=set(archive.namelist())
    ann_names=set(annotation_zip.namelist())
    candidates={}
    for spec in original:
        stem=spec["name"][3:]
        for player in range(4):
            name=f"{player:02d}_"+stem
            audio=name+"_mic.wav"
            annotation=name+".jams"
            if audio not in wav_names or annotation not in ann_names:
                continue
            row=dict(spec,name=name,audio=audio,annotation=annotation)
            candidates[name]=row
    actual=list(candidates.values())
    if len(actual)<len(original) or len(actual)>len(original)*4:
        raise AssertionError("Player augmentation contract or archive paths failed")
    if {x["composition"] for x in actual}!={x["composition"] for x in original}:
        raise AssertionError("Player augmentation changed composition split")
    if any(x["name"].startswith(("04_","05_")) for x in actual):
        raise AssertionError("Unseen DEV/TEST player contaminated training")
    return sorted(actual,key=lambda x:(x["composition"],x["name"]))




def rank_cols(x,count):
    """Find strongest CURRENT pitch responses, ignoring ordering."""
    x=np.asarray(x,dtype=np.float32)
    return np.sort(x,axis=1)[:,-count:][:,::-1]


def shadow_features(feat,p,k,events_count):
    """Entire stream built from frame-local or earlier features only.

    No forward-looking filters, annotation fields or future-derived
    temporal statistics appear here. Counts are reconstructed from
    native live events only up to each frame.
    """
    n=len(k)
    if (k.shape!=(n,7) or p.shape!=(n,49,3) or
        feat.shape!=(n,49,126) or len(events_count)!=n):
        raise ValueError("S700 original NOTE ABI mismatch")
    if n==0:return np.empty((0,63),np.float32)
    ks=posterior_ema(k,.25)
    lag8=k[np.maximum(np.arange(n)-8,0)]
    lag32=k[np.maximum(np.arange(n)-32,0)]
    # 49 source notes, never annotated pitch or true K.
    onset=p[:,:,0];sus=p[:,:,1];off=p[:,:,2]
    top_on=rank_cols(onset,6)
    top_sus=rank_cols(sus,6)
    top_off=rank_cols(off,3)
    fundamentals=np.maximum(feat[:,:,[0,8,16]],0)
    harmonics=np.maximum(feat[:,:,[1,9,17]],0)
    # Aggregate current source pitch spectra; not a GT chord count.
    spectral=np.column_stack((
        np.max(fundamentals,axis=(1,2)),
        np.mean(fundamentals,axis=(1,2)),
        np.sum(fundamentals,axis=(1,2)),
        np.max(harmonics,axis=(1,2)),
        np.mean(harmonics,axis=(1,2)),
        np.sum(harmonics,axis=(1,2))))
    counts=np.asarray(events_count,dtype=np.int32)
    t=np.arange(n)
    count_previous=counts[np.maximum(t-4,0)]
    count_long=counts[np.maximum(t-20,0)]
    X=np.column_stack((
        k,ks,lag8,lag32,
        top_on,top_sus,top_off,
        np.sum(sus>=.25,axis=1),
        np.sum(sus>=.50,axis=1),
        np.sum(sus>=.75,axis=1),
        np.sum(onset>=.25,axis=1),
        np.sum(onset>=.55,axis=1),
        np.sum(off>=.60,axis=1),
        np.sum(sus,axis=1),
        np.max(sus,axis=1),
        np.sum(onset,axis=1),
        spectral,
        counts,count_previous,count_long,
        counts-count_previous,counts-count_long,
    )).astype(np.float32)
    if not np.isfinite(X).all():raise ValueError("S700 features contain NaN/Inf")
    return X


def chunked_predict(model,X):
    out=np.empty((len(X),7),np.float32)
    # Labels 0..6 might be absent on a small TRAIN sample;
    # align probability columns with original K0..K6 ABI.
    for a in range(0,len(X),8192):
        b=min(a+8192,len(X))
        pr=model.predict_proba(X[a:b])
        out[a:b]=0
        out[a:b,model.classes_]=pr
    return out


def predictions(item,model):
    x=item["features"];p=chunked_predict(model,x)
    k=item["k"];native=item["count"]
    original=np.eye(7,dtype=np.float32)[native]
    result={"native_decoded":native,"frozen_direct_K":k.argmax(axis=1),
            "learned":p.argmax(axis=1)}
    for weight in (.30,.55,.75):
        # Learned probabilities plus already-decoded live cardinality.
        result[f"fuse_{weight:.2f}"]=(weight*p+(1-weight)*original).argmax(axis=1)
    return result


def record(spec,root,archive,heads,selection):
    audio,sha=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(audio)/SR)
    ends,feat=feature_track(audio,block_size=4096)
    p,k=native_posteriors(feat,heads)
    source_sha=digest(p.tobytes()+k.tobytes())
    events=decode(ends,p,k,selection,len(audio))
    count=native_count_timeline(events,ends)
    actual=gt_count_timeline(truth,ends)
    X=shadow_features(feat,p,k,count)
    assert digest(p.tobytes()+k.tobytes())==source_sha
    return dict(name=spec["name"],composition=spec["composition"],
                split=spec["split"],audio_sha256=sha,
                y=actual,k=k,count=count,features=X,
                source_onsets=sum(e["type"]=="note_on" for e in events))


def train(examples):
    x=np.vstack([a["features"][::TRAIN_ROW_STRIDE] for a in examples])
    y=np.concatenate([a["y"][::TRAIN_ROW_STRIDE] for a in examples])
    if len(y)<20000 or len(np.unique(y))<3:
        raise AssertionError("S700 not enough truly annotated K classes")
    # Force repeatable training, do not tune model hyperparameters on DEV.
    model=HistGradientBoostingClassifier(
        max_iter=195,learning_rate=.060,max_leaf_nodes=31,
        min_samples_leaf=105,l2_regularization=25,early_stopping=False,
        random_state=700)
    # Polyphonic chord frames are the primary error class. Increase
    # TRAIN-only weight of true K2..K6; never read GT at inference.
    frame_weight=np.where(y>=4,2.5,np.where(y>=3,2.0,
                     np.where(y>=2,1.6,1.0))).astype(np.float32)
    model.fit(x,y,sample_weight=frame_weight)
    return model,dict(training_weights={"K0":1.0,"K1":1.0,
        "K2":1.6,"K3":2.0,"K4_K6":2.5},
        rows=len(y),train_classes=np.unique(y).tolist(),
                      class_hist=np.bincount(y,minlength=7).tolist(),
                      feature_dimensions=x.shape[1])


def aggregate(rows,policy):
    keys=("correct","total","active_correct","active_total",
          "transition_correct","transitions")
    counts={key:sum(row["metrics"][policy][key] for row in rows) for key in keys}
    return dict(**counts,
         accuracy=counts["correct"]/max(1,counts["total"]),
         active_accuracy=counts["active_correct"]/max(1,counts["active_total"]),
         transition_accuracy=counts["transition_correct"]/max(1,counts["transitions"]))


def choose(rows):
    totals={name:aggregate(rows,name) for name in POLICIES}
    base=totals["native_decoded"]
    accepted=[]
    for name in POLICIES[1:]:
        item=totals[name]
        if (item["accuracy"]>base["accuracy"] and
            item["active_accuracy"]>base["active_accuracy"] and
            item["transition_accuracy"]>=base["transition_accuracy"]):
            accepted.append(name)
    accepted.sort(key=lambda p:(totals[p]["active_accuracy"],
                                totals[p]["accuracy"]),reverse=True)
    return accepted[0] if accepted else "native_decoded",totals,accepted


def run(out):
    if out.exists():raise FileExistsError("S700 immutable report path exists")
    st=time.perf_counter()
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    proto=json.loads((ROOT/"protocol.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    tr=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dv=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    comps=sorted({s["composition"] for s in tr})
    assert len(tr)==36 and len(dv)==12 and len(comps)==9
    cal=set(comps);held=set(comps)
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    sel=Selection(**heads.metadata["selection"])
    report=dict(schema="S700_TRAINED_CAUSAL_K_SHADOW_NEW_HEAD",
        frozen_model_sha256=frozen["model_sha256"],
        original_NATIVE_K_0_to_6_logits_unchanged=True,
        all_original_live_NoteOn_and_NoteOff_unchanged=True,
        current_S675_release_model_unchanged=True,
        only_past_audio_native_K_and_native_active_counts=True,
        safety_is_shadow_only_no_MIDI_replacement=True,
        DEV_previously_exposed=True,TEST_labels_unopened=True,
        train=sorted(cal),held=sorted(held),
        training_players=["00","01","02"], heldout_players=["03"],
        held_out_player_different_from_all_training_players=True)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        for key,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            arc=proto["guitar_archives"][key]
            download(arc["url"],root/"data"/name,arc["size"],arc["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive, zipfile.ZipFile(
                root/"data/GuitarSet_annotations.zip") as annotations:
            train_original=list(tr)
            all_players=expanded_train_specs(train_original,archive,annotations)
            train_expanded=[s for s in all_players if s["name"][:2] in ("00","01","02")]
            held_expanded=[s for s in all_players if s["name"][:2]=="03"]
            if len(train_expanded)!=108 or len(held_expanded)!=36:
                raise AssertionError("S700 strict 108 TRAIN player / 36 held player composition protocol violated")
            report["authentic_training_players"]="00-02"
            report["authentic_training_sources"]=len(train_expanded)
            report["trained_only_on_preexisting_nine_TRAIN_composition_families"]=True
            print("S700 AUTHENTIC TRAIN",len(train_expanded),flush=True)
            learned=[]
            for i,spec in enumerate(train_expanded,1):
                learned.append(record(spec,root,archive,heads,sel))
                print("S700 TRAIN",i,"/",len(train_expanded),spec["name"],flush=True)
            model,train_stats=train(learned)
            report["training_statistics"]=train_stats
            del learned
            heldrows=[]
            for i,spec in enumerate(held_expanded,1):
                item=record(spec,root,archive,heads,sel)
                trial=predictions(item,model)
                heldrows.append(dict(name=spec["name"],composition=spec["composition"],
                           metrics={p:score(k,item["y"]) for p,k in trial.items()}))
                print("S700 HELD",i,"/36",spec["name"],flush=True)
            selected,hold_scores,qualified=choose(heldrows)
            report["TRAIN_held_frozen_selection"]=selected
            report["TRAIN_held_metrics"]=hold_scores
            report["TRAIN_held_qualified"]=qualified
            print("S700 TRAIN FROZEN",selected,json.dumps(hold_scores[selected]),flush=True)
            devrows=[]
            for i,spec in enumerate(dv,1):
                item=record(spec,root,archive,heads,sel)
                trial=predictions(item,model)
                devrows.append(dict(name=spec["name"],composition=spec["composition"],
                         metrics={p:score(k,item["y"]) for p,k in trial.items()}))
                print("S700 DEV",i,"/12",spec["name"],flush=True)
    dev_metrics={name:aggregate(devrows,name) for name in POLICIES}
    report["DEV_all_policies_diagnostic"]=dev_metrics
    report["DEV_chosen"]=dev_metrics[selected]
    report["DEV_original"]=dev_metrics["native_decoded"]
    report["DEV_detailed"]=devrows
    report["DEV_ge80_total"]=dev_metrics[selected]["accuracy"]>=.8
    report["DEV_ge80_active"]=dev_metrics[selected]["active_accuracy"]>=.8
    report["held_ge80_total"]=hold_scores[selected]["accuracy"]>=.8
    report["held_ge80_active"]=hold_scores[selected]["active_accuracy"]>=.8
    report["four_lock_note_benchmark_unchanged"]=True
    report["promotion"]="SHADOW_ONLY_NO_LIVE_EVENT_CHANGES"
    report["elapsed_s"]=round(time.perf_counter()-st,2)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S700 FINAL",json.dumps(dict(
       selected=selected,TRAIN_HELD=hold_scores[selected],
       DEV_original=dev_metrics["native_decoded"],
       DEV_shadow=dev_metrics[selected],
       DEV_ge80_total=report["DEV_ge80_total"],
       DEV_ge80_active=report["DEV_ge80_active"],
       existing_MIDI_Onset_pitch_Offset_S675_unchanged=True)),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    with threadpool_limits(limits=1):run(args.out)
