"""S645: real (not mixed) GuitarSet octave-duo classification on natural chords.

S643 learnt constructed duets with 82.92% F1 on constructed hold-outs,
but S644 proved its transfer to REAL microphone GuitarSet DEV chords
is only 40.74% F1 (28 positives, 333 negatives). This study fixes the
TRAIN distribution mismatch: learn only from natural original GuitarSet
TRAIN recordings, with simultaneous ±30ms true octave labels as in S644.

Nine TRAIN compositions; threshold/model chosen exclusively from 5-fold
COMPOSITION-disjoint out-of-fold scores. Then freeze all choices and
evaluate once on twelve GuitarSet DEV original microphone recordings.

No GuitarSet TEST, IDMT TEST or URMP TEST records used. No NOTE S614
or S641 model, K head, or strict microphone runtime is modified.
A 190ms AFTER-onset observation is used: causal, but not instant note
emission. This is a diagnostic binary classification, not global MIDI F1.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.s613_prepare_strict_live import download,digest
from scripts.s637_guitarset_three_view_groundtruth import ROOT
from scripts.s644_natural_octave_transfer import dev_source,standard_metrics

THRESHOLDS=(.08,.12,.16,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.95)
EVALUATION_TOLERANCE_MS=30
OBSERVATION_AFTER_NOTE_MS=190


def small_indices():
    """Independent octave harmonic physical ratios and temporal novelties.

    Raw signal is represented at 2 observation times, each at 3
    resolutions. Keep the 8 physical harmonic energies and the 5 key
    odd-vs-even / higher octave contrast ratios, plus their derivatives.
    No learned feature selection based on DEV labels.
    """
    one=1+3*19
    chosen=[0]
    for part in (0,1,2):
        base=0 if part==0 else one if part==1 else 2*one
        # Part 2 is the later-minus-early difference of all EXCEPT pitch,
        # so first harmonic is at 2*one+0, not +1.
        offset=1 if part<2 else 0
        for j in range(3):
            start=base+offset+j*19
            chosen.extend(start+ix for ix in (0,1,2,3,4,5,7,8,9,10,11,15,16,17,18))
    return tuple(chosen)


def models():
    return {
        "logistic_173D":make_pipeline(
            StandardScaler(),
            LogisticRegression(C=.20, class_weight="balanced",
                               max_iter=1800,random_state=645)),
        "logistic_harmonic_subset":make_pipeline(
            StandardScaler(),
            LogisticRegression(C=.30, class_weight="balanced",
                               max_iter=1800,random_state=645)),
        "small_gradient_ensemble":HistGradientBoostingClassifier(
            max_iter=100,max_leaf_nodes=7,max_depth=3,
            learning_rate=.07,min_samples_leaf=18,
            l2_regularization=5.,class_weight="balanced",
            random_state=645),
    }


def training_rows(records):
    flat=[(r,source["composition"],source["name"])
          for source in records for r in source["rows"]]
    if not flat:raise ValueError("No original GuitarSet annotated onsets")
    x=np.stack([r["feature"] for r,_,_ in flat]).astype(np.float32)
    y=np.asarray([r["target"] for r,_,_ in flat],np.int8)
    groups=np.asarray([g for _,g,_ in flat],dtype=object)
    if len(set(groups))!=9 or len(x)<150 or int(y.sum())<15:
        raise ValueError(f"Insufficient natural octave TRAIN data: {len(x)}, {y.sum()}, {set(groups)}")
    return x,y,groups


def project(x,method):
    if method=="logistic_harmonic_subset":
        return x[:,small_indices()]
    return x


def oof_select_model(x,y,groups):
    choices={}
    folds=[]
    splits=list(GroupKFold(n_splits=5).split(x,y,groups))
    for train_idx,val_idx in splits:
        trained=set(groups[train_idx]);validation=set(groups[val_idx])
        if trained&validation:
            raise AssertionError("GuitarSet composition leakage in TRAIN CV")
        if set(np.unique(y[train_idx]))!={0,1}:
            raise AssertionError("No true octave examples in CV training")
        folds.append(dict(train_compositions=sorted(trained),
                          held_compositions=sorted(validation),
                          true_train=int(y[train_idx].sum()),
                          true_held=int(y[val_idx].sum()),
                          n_train=len(train_idx),n_held=len(val_idx)))
    for method,baseline in models().items():
        features=project(x,method)
        predicted=np.zeros(len(y),np.float64)
        for train_idx,val_idx in splits:
            head=clone(baseline)
            head.fit(features[train_idx],y[train_idx])
            predicted[val_idx]=head.predict_proba(features[val_idx])[:,1]
        trials=[]
        for threshold in THRESHOLDS:
            adjusted=(predicted>=threshold).astype(np.float64)
            row=standard_metrics(y,adjusted)
            trials.append(dict(threshold=threshold,**row))
        # Entirely TRAIN-only selection by composition-out-of-fold F1.
        winner=max(trials,key=lambda r:(
            r["F1"] if r["F1"] is not None else -1,
            r["precision"] if r["precision"] is not None else 0,
            -r["FP"],r["threshold"]))
        choices[method]=dict(threshold=winner["threshold"],
                             OOF=winner,
                             trials=trials)
    winner=max(choices,key=lambda name:(
        choices[name]["OOF"]["F1"],
        choices[name]["OOF"]["precision"],
        -choices[name]["OOF"]["FP"],name))
    return winner,choices,folds


def evaluate_source_set(records,model,name,threshold):
    all_items=[]
    by_source=[]
    for source in records:
        candidates=source["rows"]
        if candidates:
            x=np.stack([r["feature"] for r in candidates]).astype(np.float32)
            p=model.predict_proba(project(x,name))[:,1]
            rows=[dict(
                time_s=r["at_s"],low_pitch=r["low_pitch"],
                upper_pitch=r["upper_pitch"],true_chord_K=r["true_chord_K"],
                target=int(r["target"]),probability=float(prob),
                predicted=int(prob>=threshold))
                for r,prob in zip(candidates,p)]
        else:
            rows=[]
        all_items.extend(rows)
        local=standard_metrics(
            [r["target"] for r in rows],
            [r["predicted"] for r in rows])
        by_source.append(dict(name=source["name"],composition=source["composition"],
                              metrics=local))
    y=[r["target"] for r in all_items]
    p=[r["predicted"] for r in all_items]
    by_pitch={}
    for pitch in range(40,65):
        filtered=[r for r in all_items if r["low_pitch"]==pitch]
        if not filtered:continue
        by_pitch[str(pitch)]=standard_metrics(
            [r["target"] for r in filtered],
            [r["predicted"] for r in filtered])
    by_K={}
    for group,predicate in (
        ("onset_single",lambda r:r["true_chord_K"]==1),
        ("onset_K2",lambda r:r["true_chord_K"]==2),
        ("onset_K3plus",lambda r:r["true_chord_K"]>=3)):
        filtered=[r for r in all_items if predicate(r)]
        by_K[group]=standard_metrics(
            [r["target"] for r in filtered],
            [r["predicted"] for r in filtered])
    return dict(model=name,threshold=threshold,
                overall=standard_metrics(y,p),
                by_K=by_K,by_pitch=by_pitch,
                per_source=by_source,
                events=all_items)


def run(out):
    if out.exists():raise FileExistsError("S645 evidence cannot be overwritten")
    protocol=json.loads((ROOT/"protocol.json").read_text())
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    train=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    test=[s for s in protocol["sources"] if s["split"]=="test"]
    assert len(train)==36 and len(dev)==12 and len({s["composition"] for s in train})==9
    assert not {(s["corpus"],s["composition"]) for s in train}&{
        (s["corpus"],s["composition"]) for s in dev+test}
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for name,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                              ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            item=protocol["guitar_archives"][name]
            download(item["url"],root/"data"/filename,item["size"],item["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            original_train=[]
            for i,s in enumerate(train,1):
                rec=dev_source(s,root,archive)
                original_train.append(rec)
                targets=[r["target"] for r in rec["rows"]]
                print("S645 TRAIN REAL",i,"/36",rec["name"],
                      "natural_low_onsets",len(targets),
                      "true_duo",sum(targets),flush=True)
            x,y,groups=training_rows(original_train)
            winner,choices,folds=oof_select_model(x,y,groups)
            print("S645 SELECT TRAIN ONLY",json.dumps(dict(
                train_n=len(y),train_duos=int(y.sum()),
                winner=winner,models={
                    k:dict(threshold=v["threshold"],oof=v["OOF"])
                    for k,v in choices.items()})),flush=True)
            finals={}
            for name,base in models().items():
                head=clone(base)
                head.fit(project(x,name),y)
                finals[name]=head
            original_dev=[]
            for i,s in enumerate(dev,1):
                rec=dev_source(s,root,archive)
                original_dev.append(rec)
                targets=[r["target"] for r in rec["rows"]]
                print("S645 DEV REAL",i,"/12",rec["name"],
                      "natural_low_onsets",len(targets),
                      "true_duo",sum(targets),flush=True)
    tested={}
    for name,head in finals.items():
        result=evaluate_source_set(original_dev,head,name,choices[name]["threshold"])
        tested[name]=result
        print("S645 DEV RESULT",name,json.dumps(result["overall"]),
              "K3plus",result["by_K"]["onset_K3plus"]["F1"],flush=True)
    chosen=tested[winner]
    report=dict(
        status="S645_REAL_GUITARSET_TRAINED_NATURAL_OCTAVE_DUO_COMPOSITION_GROUP_CV",
        fixed_native_NOTE_S614_sha256=frozen["model_sha256"],
        original_waveform_not_digital_duet=True,
        learned_only_GuitarSet_9_TRAIN_compositions=True,
        DEV_data_used_only_after_train_CV_selection=True,
        DEV_historically_exposed=True,TEST_never_opened=True,
        original_NOTE_heads_K_and_LIVE_unchanged=True,
        added_lookahead_from_true_note_onset_ms=OBSERVATION_AFTER_NOTE_MS,
        not_yet_compatible_with_very_low_latency_NoteOn=True,
        natural_onset_annotation_tolerance_ms=EVALUATION_TOLERANCE_MS,
        natural_train_n=len(y),natural_train_true_duos=int(y.sum()),
        frozen_model_choice_by_TRAIN_OOF=winner,
        train_CV_models=choices,train_CV_group_folds=folds,
        evaluation=tested,
        chosen_DEV_result=chosen["overall"],
        chosen_DEV_K3plus=chosen["by_K"]["onset_K3plus"],
        transfer_S644_natural_F1_reference=.4074074074074074,
        no_automatic_promotion=True,
        reminder="This scores single-versus-octave-duet classification at annotated onsets, NOT global MIDI note transcription F1.",
    )
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S645 FINAL",json.dumps(dict(
        TRAIN=dict(total=len(y),octave=int(y.sum()),winner=winner),
        DEV={name:v["overall"] for name,v in tested.items()},
        chosen=chosen["overall"],
    )),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(args.out)
