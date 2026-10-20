"""S800 immutable FIT80 OOF per-frame/per-pitch evidence snapshots.

Read-only observer of original S795 trained models and MIDI emissions.
Does not alter source notes, offsets, policies, features, or predictions.
Held-out and CAL data are not accepted; FIT composition must be OOF.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from causal_note.s795_activek_experts import HEADS,FEATURES,PITCHES
from scripts.s704_joint_metrics import metric,SCORES
from scripts.s795_train_activek_midi256 import COUNT_KEYS,score_record

SCHEMA="S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1"
EVENT_FIELDS=("type","pitch","decision_sample","available_sample","time",
              "reason","velocity","first_eligible_sample")
def canonical_json(obj):
    return json.dumps(obj,ensure_ascii=False,sort_keys=True,
                      separators=(",",":"),allow_nan=False).encode("utf-8")
def sha256(data):
    return hashlib.sha256(data).hexdigest()

def snapshot(index,record,predictions,training,fold,outdir):
    """Store complete current acoustic features, all OOF heads and raw FIT labels.

    This function NEVER runs model inference or MIDI decoding. The identity
    control is *verified* by the original S795 function, not changed.
    """
    if not 0<=index<80 or record["composition"]!=fold:
        raise AssertionError("FIT80 OOF fold/identity mismatch")
    X=np.asarray(record["X"])
    y=np.asarray(record["y"])
    d=np.asarray(record["decision_ends"],np.int64)
    ends=np.asarray(record["ends"],np.int64)
    if X.ndim!=3 or X.shape[1:]!=(PITCHES,FEATURES) or X.dtype!=np.float16:
        raise AssertionError("S795 inference float16 feature ABI changed")
    if y.shape!=X.shape[:2] or len(d)!=len(X) or len(ends)<len(d):
        raise AssertionError("S800 sample/decision/feature misalignment")
    if not len(X) or np.any(np.diff(d)<=0) or np.any(np.diff(ends)<=0):
        raise AssertionError("S800 non-monotone/empty source clock")
    if set(predictions)!=set(HEADS):
        raise AssertionError("S800 missing OOF specialist")
    for name in HEADS:
        z=np.asarray(predictions[name])
        if z.shape!=X.shape[:2] or z.dtype!=np.float32 or not np.isfinite(z).all():
            raise AssertionError("OOF specialist dimensions or finite values")
        if (z<0).any() or (z>1).any():
            raise AssertionError("OOF probability outside [0,1]")
        spec=training[name]
        if spec.get("trained") and fold in spec.get("compositions",[]):
            raise AssertionError("OOF leakage: head trained on target fold")
    source_score=metric(record["events"],record["truth"],ends)
    identity,delta=score_record(record,predictions,0,0.)
    if any(source_score[k]!=identity[k] for k in COUNT_KEYS):
        raise AssertionError("S795 shadow identity differs from source MIDI counts")
    if any(abs(source_score[k]-identity[k])>1e-12 for k in SCORES):
        raise AssertionError("S795 shadow identity differs from source MIDI metrics")
    if delta["new_onsets"] or delta["new_offsets"] or delta["max_added_native_delay_samples"]:
        raise AssertionError("S800 introduced a MIDI event")
    events=[{k:e[k] for k in EVENT_FIELDS if k in e}
            for e in record["events"]]
    for e in events:
        if e["available_sample"]<e["decision_sample"]:
            raise AssertionError("source event emitted before availability")
    truth=[{k:n[k] for k in ("pitch","time","duration")} for n in record["truth"]]
    meta=dict(schema=SCHEMA,index=index,name=record["name"],
              composition=fold,texture=record["texture"],source_events=events,
              fit_truth=truth,head_order=list(HEADS),frames=int(len(ends)),
              fragment_frames=int(len(d)),samples=int(record["samples"]),
              source_counts={k:int(source_score[k]) for k in COUNT_KEYS},
              source_scores={k:float(source_score[k]) for k in SCORES},
              no_model_retraining=True,
              production_live_modified=False,cal_annotations_opened=False,
              policy_selection_on_cal=False,head_training_fold_excluded=True,
              source_events_unmodified=True)
    arrays={"features":X,"onset_targets_fit":y.astype(np.uint8),
            "decision_ends":d,"metric_ends":ends}
    for h in HEADS:
        arrays["head_"+h]=np.asarray(predictions[h],np.float32)
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    stem=f"fit_{index:03d}"
    target_npz=outdir/(stem+".npz")
    target_meta=outdir/(stem+".json")
    if target_npz.exists() or target_meta.exists():
        raise FileExistsError("S800 immutable OOF trace already present")
    # np.savez_compressed writes only deterministic numeric arrays, no pickle.
    np.savez_compressed(target_npz,**arrays)
    m=canonical_json(meta)
    target_meta.write_bytes(m+b"\n")
    return dict(index=index,name=record["name"],composition=fold,
       fold_held_out=True,frames=int(len(ends)),
       fragment_frames=int(len(d)),
       source_counts=meta["source_counts"],
       npz_sha256=sha256(target_npz.read_bytes()),
       json_sha256=sha256(target_meta.read_bytes()),
       npz_bytes=target_npz.stat().st_size)

def verify_snapshot(item,outdir):
    """Refuse tampered, missing, incomplete, shifted or mislabeled traces."""
    stem=f"fit_{item['index']:03d}"
    path=Path(outdir)
    zpath=path/(stem+".npz");mpath=path/(stem+".json")
    if not zpath.is_file() or not mpath.is_file():
        raise AssertionError("missing FIT OOF evidence")
    if sha256(zpath.read_bytes())!=item["npz_sha256"] or sha256(
            mpath.read_bytes())!=item["json_sha256"]:
        raise AssertionError("S800 trace SHA256 mismatch")
    m=json.loads(mpath.read_text(encoding="utf-8"))
    if (m["schema"]!=SCHEMA or m["index"]!=item["index"]
            or m["composition"]!=item["composition"]
            or m["cal_annotations_opened"] or not m["head_training_fold_excluded"]):
        raise AssertionError("OOF source provenance changed")
    with np.load(zpath,allow_pickle=False) as z:
        if set(z.files)!={"features","onset_targets_fit","decision_ends",
                           "metric_ends"}|{"head_"+x for x in HEADS}:
            raise AssertionError("missing/foreign trace arrays")
        X=z["features"]; d=z["decision_ends"]; ends=z["metric_ends"]
        if X.shape!=(item["fragment_frames"],PITCHES,FEATURES) or (
                len(ends)!=item["frames"]):
            raise AssertionError("OOF shapes drift")
        if X.dtype!=np.float16 or np.any(np.diff(d)<=0) or np.any(np.diff(ends)<=0):
            raise AssertionError("OOF types/time drift")
        for h in HEADS:
            P=z["head_"+h]
            if P.shape!=X.shape[:2] or P.dtype!=np.float32:
                raise AssertionError("OOF specialist ABI drift")
    return m
