"""S616 LIVE: preregistered URMP-development variants replayed on GuitarSet dev.

The choices here are *fixed before* loading GuitarSet notes:
- S614 control
- harmonic guard 0.6
- rising-attack rescue 0.6
- combined harmonic 0.5 / rescue 0.4
- release four frames

These are the 12 original S614 GuitarSet development recordings (three
distinct arrangements families), NOT the player-05 test and NOT a new
independent validation. Original S80/S614 weights remain untouched.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
from scipy.io import wavfile

from causal_note.strict_live import SR, feature_track, load_heads, Selection
from causal_note.s616_live_guard import GuardParameters, replay
from scripts.s613_prepare_strict_live import download, digest, labels
from scripts.s613_train_strict_live import real_notes, event_score, offset_audit, decode, summary

EVIDENCE=Path("analysis/evidence/v273-s613-strict-live")
MODEL=Path("model/live/strict_note/NOTE_strict_live.npz")
CONTROL={"TP":998,"FP":585,"FN":582}
POLICIES=(
    ("S614_baseline",GuardParameters(),5),
    ("harmonic_0.6",GuardParameters(harmonic_penalty=.6),5),
    ("rescue_0.6",GuardParameters(attack_rescue=.6),5),
    ("joint_0.5_0.4",GuardParameters(.5,.4),5),
    ("release_4",GuardParameters(),4),
)


def run(out):
    if out.exists():
        raise FileExistsError("Immutable cross-corpus development evidence")
    out.mkdir(parents=True)
    protocol=json.loads((EVIDENCE/"protocol.json").read_text())
    frozen=json.loads((EVIDENCE/"selection_frozen.json").read_text())
    assert digest((EVIDENCE/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest()==frozen["model_sha256"]
    selected=[s for s in protocol["sources"] if s["split"]=="dev" and s["corpus"]=="GuitarSet"]
    assert len(selected)==12 and len({s["composition"] for s in selected})==3
    assert all(s["name"].startswith("04_") for s in selected)
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    cfg=Selection(**heads.metadata["selection"])
    scores={name:[] for name,_,_ in POLICIES}
    sources=[]
    begin=time.perf_counter()
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        (root/"data").mkdir()
        for section,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                             ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            item=protocol["guitar_archives"][section]
            download(item["url"],root/"data"/name,item["size"],item["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as wav_zip:
            for spec in selected:
                audio_blob=wav_zip.read(spec["audio"])
                rate,pcm=wavfile.read(io.BytesIO(audio_blob))
                assert rate==SR and pcm.dtype in (np.int16,np.int32)
                signal=pcm.astype(np.float32)/(32768 if pcm.dtype==np.int16 else 2147483648)
                if signal.ndim==2:
                    signal=signal.mean(axis=1)
                signal=signal[:int(spec["seconds"]*SR)].copy()
                ends,x=feature_track(signal)
                pred,k=heads.predict(x)
                truth_data=labels(spec,root)
                truth=real_notes(truth_data["notes"],len(signal)/SR)
                direct=decode(ends,pred,k,cfg,len(signal))
                noop=replay(ends,pred,k,x,cfg,GuardParameters(),len(signal))
                assert noop==direct
                for name,param,release in POLICIES:
                    policy=replace(cfg,release_frames=release)
                    events=replay(ends,pred,k,x,policy,param,len(signal))
                    scored,pairs=event_score(events,truth)
                    offs=offset_audit(events,truth,pairs,len(signal))
                    scores[name].append(dict(source=spec["name"],score=scored,offset=offs))
                sources.append(dict(name=spec["name"],composition=spec["composition"],
                                    true_notes=len(truth),
                                    source_sha256=digest(audio_blob),
                                    annotation_sha256=truth_data["sha256"]))
                print("S616 GUITAR",spec["name"],"notes",len(truth),
                      "elapsed_s",round(time.perf_counter()-begin,1),flush=True)
    baseline=summary([r["score"] for r in scores["S614_baseline"]])
    assert all(baseline[k]==v for k,v in CONTROL.items()),"Failed to reproduce unchanged original S614"
    variants=[]
    for name,parameters,release in POLICIES:
        rows=scores[name]
        total=summary([r["score"] for r in rows])
        offs=Counter()
        for row in rows:
            offs.update({k:row["offset"][k] for k in ("matched_noteons_with_observed_reference_offset",
                        "offsets_within_max_50ms_or_20pct_duration","forced_or_missing_offsets")})
        correct=offs["offsets_within_max_50ms_or_20pct_duration"]
        eligible=offs["matched_noteons_with_observed_reference_offset"]
        variants.append(dict(name=name,
            F1=total["F1"],TP=total["TP"],FP=total["FP"],FN=total["FN"],
            delta_F1_points=100*(total["F1"]-baseline["F1"]),
            offsets=dict(offs,accuracy=correct/eligible if eligible else 0.),
            by_source={r["source"]:{q:r["score"][q] for q in ("F1","TP","FP","FN")} for r in rows}))
    report=dict(status="S616_FROZEN_CROSS_DEV_TRANSFER_NO_PROMOTION",
        reference_checkpoint_sha256=frozen["model_sha256"],
        original_S80_and_S614_unchanged=True, fixed_variants_before_opening_guitar_dev=True,
        guitarset_dev_already_used_in_S614_selection=True,
        development_only_no_unseen_song_heldout_claim=True,
        historical_test_not_loaded=True, sources=sources,
        baseline=baseline,variants=variants,elapsed_s=time.perf_counter()-begin)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S616 GUITAR DEV SUMMARY",json.dumps(
        [{k:row[k] for k in ("name","F1","TP","FP","FN","delta_F1_points")} for row in variants],indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    run(args.out)
