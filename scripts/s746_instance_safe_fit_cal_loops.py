"""S746: FIT80 trained event-instance selector; 12 real CAL32 full-MIDI loops.

Original S708/S707 outputs are immutable. Distinct S743 heads and 32
per-fragment subsets propose supplemental pitches; a separate FIT-trained
classifier learns whether their complete note instances are correct.
The classifier's *inputs at inference* are strictly onset-time causal.
S746 always scores the actual merged NoteOn/NoteOff stream, not head K.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import tempfile
import zipfile

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from causal_note.strict_live import SR
from causal_note.s707_causal_release import prepare_hazards, render_release
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s743_fundamental_router import MultiHeadBirthPolicy, decode_s743
from causal_note.s746_instance_guard import (
    InstanceGate, extra_pairs, onset_features, fit_event_labels, merge_events)
from scripts.s613_prepare_strict_live import download, labels, digest
from scripts.s613_train_strict_live import real_notes, event_score
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import SCORES, metric, aggregate
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import (
    FIT_COMPS, CAL_COMPS, load_model)
from scripts.s735_audit_32_onset_combinations import load_s734
from scripts.s745_calibrated_fit_cal_loops import track_sources, save

BIRTH = MultiHeadBirthPolicy(
    k_tail=.20, min_onset=.12, min_sustain=.35,
    mask_delta=.04, mask_strategy="adaptive", min_rise=.008,
    max_births=1).validate()
MODELS = {
    "depth3": dict(max_depth=3, max_iter=110, learning_rate=.07),
    "depth5": dict(max_depth=5, max_iter=140, learning_rate=.05),
}
CHOICES = {
    f"{name}_risk{int(risk*100)}_tail{int(tail*100)}":
    (name, InstanceGate(risk=risk, k_tail=tail))
    for name, risk, tail in product(
        tuple(MODELS), (.55, .75, .90), (.20, .55))
}
assert len(CHOICES) == 12


def proposals(track, policy, offset):
    """Generate all native causal candidate instances, without FIT/CAL truth."""
    item = track["item"]
    raw, diag = decode_s743(
        item["decision_ends"], track["post"], track["k"],
        track["physical_heads"], track["context"], policy,
        BIRTH, item["samples"])
    completed, skip = render_release(
        prepare_hazards(item, raw, offset),
        item["ends"], item["samples"],
        gate=.8, persistence=1, grace=24)
    pairs = extra_pairs(completed)
    feats, ids = onset_features(
        pairs, item["decision_ends"], track["birth_input"],
        track["physical_heads"], track["k"])
    return pairs, feats, ids, {**diag, "s707_budget_skipped":int(skip)}


def emit_result(out, report):
    save(out / "report.json", report)


def run(args, *, proposal_builder=None, schema=None, proposal_metadata=None):
    if proposal_builder is None:
        proposal_builder = proposals
    if args.out.exists():
        raise FileExistsError("S746 requires an immutable fresh result folder")
    args.out.mkdir(parents=True)
    source = json.loads((args.s708_dir / "report.json").read_text())
    sha = digest((args.s708_dir / "s708_selected.pt").read_bytes())
    if sha != source["checkpoint_sha256"]:
        raise AssertionError("Frozen S708 checkpoint signature mismatch")
    h5report = json.loads(args.s734_report.read_text())
    if h5report["base_sha256"] != sha or not h5report["complete"]:
        raise AssertionError("Signed S734 H0 acoustic source mismatch")
    if source["frozen_candidate"]["choice"]["release"] != dict(
            gate=.8, persistence=1, grace=24):
        raise AssertionError("S707 original release policy changed")
    policy = GuidedPolicy(**source["frozen_candidate"]["choice"]["policy"])
    torch.set_num_threads(2)
    network = load_model(args.s708_dir / "s708_selected.pt")[0]
    h5 = load_s734(args.s734_checkpoint, sha)
    report = dict(
        schema=schema or "S746_REAL_FIT80_EVENT_INSTANCE_SAFE_CAL32",
        signed_sha256=sha, split="FIT80 CAL32 composition-disjoint",
        complete=False, promoted=False, LIVE_changed=False,
        TEST_opened=False, selected="S708_frozen", loops=len(CHOICES),
        event_truth_used_at_inference=False,
        original_release_unchanged=True,
        FIT={}, CAL={}, CAL_candidate=None,
        source_birth_policy=proposal_metadata if proposal_metadata is not None else asdict(BIRTH),
        proposal_generator=proposal_builder.__module__ + "." + proposal_builder.__name__,
        param_grid={name:dict(model=model, policy=asdict(gate))
                    for name,(model, gate) in CHOICES.items()})
    emit_result(args.out, report)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "data").mkdir()
        proto, heads, selection, offset = setup(root, args.offset_checkpoint)
        for kind, name in (
            ("GuitarSet_audio", "GuitarSet_audio.zip"),
            ("GuitarSet_annotations", "GuitarSet_annotations.zip")
        ):
            archive = proto["guitar_archives"][kind]
            download(archive["url"], root / "data" / name,
                     archive["size"], archive["sha256"])
        train = [spec for spec in proto["sources"]
                 if spec["corpus"] == "GuitarSet" and spec["split"] == "train"]
        with zipfile.ZipFile(root / "data/GuitarSet_audio.zip") as audio, \
             zipfile.ZipFile(root / "data/GuitarSet_annotations.zip") as annotations:
            expanded = expanded_train_specs(train, audio, annotations)
            groups = {
                "FIT":[s for s in expanded if s["composition"] in FIT_COMPS],
                "CAL":[s for s in expanded if s["composition"] in CAL_COMPS],
            }
            if [len(groups[z]) for z in ("FIT","CAL")] != [80,32]:
                raise AssertionError("S746 invalid disjoint corpus split")
            train_x, train_y, counters = [], [], Counter()
            for i, spec in enumerate(groups["FIT"]):
                tr = track_sources(spec,audio,root,heads,selection,offset,
                                   network,policy,source["frozen_candidate"]
                                   ["choice"]["release"],h5)
                pairs, feat, ids, diag = proposal_builder(tr,policy,offset)
                truth = real_notes(labels(spec,root)["notes"],
                                   tr["item"]["samples"] / SR)
                baseline_pairs = event_score(tr["baseline"],truth)[1]
                y = fit_event_labels(pairs,truth,{j for _,j in baseline_pairs})
                if len(feat) != len(y):
                    raise AssertionError("S746 FIT event supervision misaligned")
                train_x.append(feat)
                train_y.append(y)
                counters.update(proposed=len(y), complete_true=int(y.sum()))
                print("S746 FIT",i+1,80,"proposals",len(y),
                      "complete_new",int(y.sum()),flush=True)
            X=np.concatenate(train_x,axis=0)
            y=np.concatenate(train_y)
            if len(y) < 100 or len(set(y.tolist())) < 2:
                raise AssertionError("S746 no valid FIT event-instance learning basis")
            report["FIT"] = dict(
                rows=len(y), positives=int(y.sum()),
                natural_prevalence=float(y.mean()),
                source_counter=dict(counters),
                compositions=sorted({s["composition"] for s in groups["FIT"]}))
            emit_result(args.out,report)
            trained={}
            for name,hyper in MODELS.items():
                model=HistGradientBoostingClassifier(
                    random_state=746 if name=="depth3" else 747,
                    l2_regularization=.3, early_stopping=False,
                    **hyper)
                # Small weighting: minimize the many false additional births.
                sample_w=np.where(y==1,2.,1.)
                model.fit(X,y,sample_weight=sample_w)
                joblib.dump(dict(
                    model=model, fit_only=True, source_sha256=sha,
                    features=X.shape[1], kind="complete_event_instance"),
                    args.out / f"s746_{name}.joblib")
                trained[name]=model
            names=["S708_frozen",*CHOICES]
            all_metrics={name:[] for name in names}
            lost={name:Counter() for name in CHOICES}
            details={name:Counter() for name in CHOICES}
            mask_hist=Counter()
            for i,spec in enumerate(groups["CAL"]):
                tr=track_sources(spec,audio,root,heads,selection,offset,
                                 network,policy,source["frozen_candidate"]
                                 ["choice"]["release"],h5)
                item=tr["item"]
                original_source=tuple(
                    (e["type"],int(e["pitch"]),int(e["decision_sample"]),
                     int(e["available_sample"]),str(e["reason"]))
                    for e in tr["baseline"])
                pairs,feat,indices,diag=proposal_builder(tr,policy,offset)
                after_proposal=tuple(
                    (e["type"],int(e["pitch"]),int(e["decision_sample"]),
                     int(e["available_sample"]),str(e["reason"]))
                    for e in tr["baseline"])
                if original_source!=after_proposal:
                    raise AssertionError("S747 candidate generator mutated signed S708 event objects")
                mask_hist.update(diag["masks"])
                probabilities={
                    name:trained[name].predict_proba(feat)[:,1]
                    if len(feat) else np.empty(0,np.float32)
                    for name in MODELS
                }
                events={"S708_frozen":tr["baseline"]}
                for name,(family,gate) in CHOICES.items():
                    stream, mux=merge_events(
                        tr["baseline"],pairs,probabilities[family],
                        indices,tr["k"],gate)
                    events[name]=stream
                    details[name].update(mux)
                # Only AFTER all inference streams are frozen, read CAL labels.
                truth=real_notes(labels(spec,root)["notes"],
                                 item["samples"]/SR)
                measured={name:metric(stream,truth,item["ends"])
                          for name,stream in events.items()}
                base=measured["S708_frozen"]
                for name in names:
                    all_metrics[name].append(measured[name])
                for name in CHOICES:
                    now=measured[name]
                    lost[name]["correct_offsets"]+=len(
                        set(base["correct_offset_truth_ids"])-
                        set(now["correct_offset_truth_ids"]))
                    lost[name]["long_onsets"]+=len(
                        set(base["long_onset_truth_ids"])-
                        set(now["long_onset_truth_ids"]))
                    lost[name]["onset_tp"]+=max(0,base["tp"]-now["tp"])
                print("S746 CAL",i+1,32,spec["name"],
                      "extra",len(pairs),
                      "best_activeK",round(max(measured[name]["active_exact_k"]
                                               for name in CHOICES),4),flush=True)
                report["CAL_recordings_done"]=i+1
                report.setdefault("source_per_recording",[]).append(dict(
                    name=spec["name"],
                    source_events=len(original_source),
                    scores={k:float(base[k]) for k in SCORES},
                    tp=int(base["tp"]),fp=int(base["fp"]),fn=int(base["fn"]),
                    k_correct=int(base["k_correct"]),frames=int(base["frames"])))
                emit_result(args.out,report)
            sums={name:aggregate(records)
                  for name,records in all_metrics.items()}
            base=sums["S708_frozen"]
            original_signed=source["frozen_candidate"]["CAL"]
            report["baseline_parity"]=dict(
                original={name:float(original_signed[name]) for name in SCORES},
                current={name:float(base[name]) for name in SCORES},
                diff={name:float(base[name]-original_signed[name]) for name in SCORES},
                valid=bool(all(abs(base[name]-original_signed[name])<=1e-8
                               for name in SCORES)))
            report["reference_scores"]={name:float(base[name]) for name in SCORES}
            emit_result(args.out,report)
            report["candidate_masks"]=dict(mask_hist)
            eligible=[]
            for name in CHOICES:
                scores=sums[name]
                passed=bool(report["baseline_parity"]["valid"]
                    and all(scores[k]>=base[k]-1e-12 for k in SCORES)
                    and lost[name]["correct_offsets"]==0
                    and lost[name]["long_onsets"]==0
                    and lost[name]["onset_tp"]==0
                    and scores["tp"]>=base["tp"])
                d=dict(
                    scores={k:float(scores[k]) for k in SCORES},
                    delta={k:float(scores[k]-base[k]) for k in SCORES},
                    baseline_events_preserved=True,
                    safe=passed,
                    six_gt80=bool(passed and all(scores[k]>.80 for k in SCORES)),
                    lost=dict(lost[name]),
                    event_counts=dict(details[name]))
                report["CAL"][name]=d
                if passed and any(d["delta"][k]>1e-12 for k in SCORES):
                    eligible.append(name)
            report["CAL_candidate"]=max(
                eligible, key=lambda n:(min(report["CAL"][n]["scores"].values()),
                                        sum(report["CAL"][n]["delta"].values())),
                default=None)
            report["complete"]=bool(report["baseline_parity"]["valid"])
            emit_result(args.out,report)
            if not report["baseline_parity"]["valid"]:
                raise AssertionError("S747 signed S708 metric mismatch; see "
                                     "report.json baseline_parity and source_per_recording")
            print("S746 COMPLETE",json.dumps(dict(
                selected=report["selected"],
                candidate=report["CAL_candidate"],
                baseline=report["reference_scores"],
                best_activeK=max((name,report["CAL"][name]["scores"]["active_exact_k"])
                                 for name in CHOICES),
                safe_candidates=eligible)),flush=True)


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--s708-dir",type=Path,required=True)
    parser.add_argument("--s734-checkpoint",type=Path,required=True)
    parser.add_argument("--s734-report",type=Path,required=True)
    parser.add_argument("--offset-checkpoint",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    return parser.parse_args()


if __name__=="__main__":
    run(parse_args())
