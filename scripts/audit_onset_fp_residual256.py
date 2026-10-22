"""Auditable residual false-Onset audit, native256, FROZEN learned weights.

Never train on this audit or claim it is a true live policy result. It only
reconstructs the SHA256-verified three-head ErrorPrediction 69D accepted
NoteOn events from a previous full genuine 80-WAV PCM256 causal replay.
All labels came from GT matching AFTER the track's audio-only inference.

Causal evidence: previous native Main NoteOn time at -12/-19/-24/-28 semitones
only. Features are measured from previous 256 boundaries, before the current
native NoteOn is incorporated. A future supervised Onset-only specialist may
use them, but this audit does NOT change models/selection/MIDI.
"""
import argparse,hashlib,json
from collections import Counter,defaultdict
from itertools import groupby
from pathlib import Path
import joblib
import numpy as np
from onset256.acoustics import HOP,SR

COMPS=("BN1","Funk2","Jazz1","Rock3","SS1")
INTERVALS=(12,19,24,28)
REGISTERS=("E2_B2","C3_B3","C4_E6")
EXPECTED_RAW=dict(tp=8099,fp=4600)
EXPECTED_ACCEPTED=dict(tp=8054,fp=4411)
MAX_PAST_SAMPLES=int(.080*SR)  # exact 80 ms (3,528 samples), not 14*256.
SCHEMA="ONSET_ERROR_MLP69_RESIDUAL_FP_CAUSAL_PARENT_V1"
HYPOTHESIS_ISSUE=142

def original_main_events(path):
    tracks=defaultdict(list)
    totals=Counter()
    keyset=set()
    with Path(path).open() as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line)
            if row.get("policy")!="native" or row.get("label") not in ("TP","FP"):
                continue
            sample=row["sample"]
            pitch=row["pitch"]
            vector=np.asarray(row["hard_predecision_vector"],np.float32)
            if (type(sample) is not int or sample%HOP!=0 or sample<=0 or
                type(pitch) is not int or not 40<=pitch<=88 or
                vector.shape!=(69,) or not np.isfinite(vector).all()
                or row["composition"] not in COMPS
                or row["texture"] not in ("solo","comp")):
                raise ValueError("Invalid signed native256 FP forensic record")
            key=(row["track"],sample,pitch)
            if key in keyset:raise AssertionError("duplicate signed Main attack")
            keyset.add(key)
            totals[row["label"]]+=1
            tracks[row["track"]].append(row)
    if totals!=Counter({"TP":8099,"FP":4600}) or len(tracks)!=80:
        raise AssertionError("S613 native256 frozen signed baseline changed")
    for track,records in tracks.items():
        records.sort(key=lambda a:(a["sample"],a["pitch"]))
        if len({r["composition"] for r in records})!=1:
            raise AssertionError("composition leakage across a track")
    return tracks

def prior_parent_ages(last_native_on,sample,pitch):
    """Strictly earlier Main NoteOn only, for four harmonic parent intervals.

    Outputs are normalized [0,1] age in 32 hops (≈186 ms);
    missing/older parents = 1. No current/future frame or labels.
    """
    idx=pitch-40
    out=np.ones(len(INTERVALS),np.float32)
    for i,delta in enumerate(INTERVALS):
        if idx>=delta:
            before=int(last_native_on[idx-delta])
            if before>=0 and before<sample:
                out[i]=min(1.,(sample-before)/(32*HOP))
    return out

def load_frozen(checkpoints):
    outputs={}
    for comp in COMPS:
        stem=Path(checkpoints)/f"onset_error_outer_{comp}"
        path=stem.with_suffix(".joblib")
        metadata=json.loads(stem.with_suffix(".json").read_text())
        if (metadata["outer_composition"]!=comp or
            comp in metadata["train_compositions"] or
            comp==metadata["cal_composition"] or
            metadata["schema"]!="ONSET_ERROR_HEAD_HARDNEGATIVE_NATIVE256_69D_V1" or
            metadata["feature_dim"]!=69 or
            metadata["training_hop_samples"]!=256):
            raise ValueError("Untrusted ErrorPrediction risk train/split ABI")
        h=hashlib.sha256(path.read_bytes()).hexdigest()
        if h!=metadata["checkpoint_sha256"]:
            raise ValueError("Onset 69D risk MLP checkpoint hash mismatch")
        model=joblib.load(path) # SHA pinned project-owned original joblib.
        if model.n_features_in_!=69:raise ValueError("expected true 256 trained 69D")
        outputs[comp]=(model,float(metadata["calibrated_cut"]),h)
    return outputs

def per_pitch_register(pitch):
    return "E2_B2" if pitch<48 else "C3_B3" if pitch<60 else "C4_E6"

def report_ratio(records):
    n=len(records)
    tp=sum(x["label"]=="TP" for x in records)
    fp=n-tp
    return dict(proposals=n,tp=tp,fp=fp,
                false_rate=fp/n if n else 0.)

def run(events,models,out):
    if Path(out).exists():
        raise FileExistsError("never overwrite immutable FP evidence")
    tracks=original_main_events(events)
    trained=load_frozen(Path(models)/"checkpoints")
    accepted=[]
    all_computed=[]
    for track in sorted(tracks):
        events=tracks[track]
        comp=events[0]["composition"]
        model,cut,checkpoint=trained[comp]
        X=np.stack([e["hard_predecision_vector"] for e in events]).astype(np.float32)
        confidence=np.asarray(model.predict_proba(X)[:,1],np.float32)
        if (confidence.shape!=(len(events),) or
            not np.isfinite(confidence).all() or
            np.any(confidence<0.) or np.any(confidence>1.)):
            raise AssertionError("frozen MLP69 risk inference invalid")
        prev=np.full(49,-10**12,np.int64)
        for sample,g in groupby(zip(events,confidence),key=lambda x:x[0]["sample"]):
            snapshot=list(g)
            # Only past native Main events are in prev, never same-frame.
            for row,correct in snapshot:
                pitch=row["pitch"];parent=prior_parent_ages(prev,sample,pitch)
                parents_recent=bool(any(
                    idx>=delta and
                    0<sample-prev[idx-delta]<=MAX_PAST_SAMPLES
                    for delta in INTERVALS
                    for idx in [pitch-40]))
                same_recent=bool(0<sample-prev[pitch-40]<=MAX_PAST_SAMPLES)
                if float(correct)>=cut:
                    data=dict(
                        track=track,composition=comp,texture=row["texture"],
                        pitch=pitch,sample=sample,
                        label=row["label"],
                        register=per_pitch_register(pitch),
                        correctness_probability=float(correct),
                        original_69d=row["hard_predecision_vector"],
                        causal_parent_ages_4d=parent.astype(float).tolist(),
                        learned_candidate_74d=(
                            [float(x) for x in row["hard_predecision_vector"]]+
                            [float(x) for x in parent]+[float(correct)]),
                        causal_prior_native_parent_80ms=parents_recent,
                        causal_prior_native_same_pitch_80ms=same_recent,
                        stronger_lower_sustain=bool(
                            row["context"]["strongest_lower_sustain"]>.5),
                        stronger_lower_fundamental=bool(
                            row["harmonic_parent_stronger"]),
                        # TRUE GT only post-inference; NEVER a model input.
                        harmonic_lower_pitch_in_gt_80ms_POSTHOC=bool(
                            row["harmonic_lower_pitch_in_truth_80ms"]),
                    )
                    if len(data["learned_candidate_74d"])!=74:
                        raise AssertionError("Onset FP specialist 74D ABI drift")
                    accepted.append(data)
                all_computed.append((row,float(correct),cut))
            for row,_ in snapshot:
                prev[row["pitch"]-40]=sample
    baseline=report_ratio(accepted)
    if (baseline["tp"]!=EXPECTED_ACCEPTED["tp"] or
        baseline["fp"]!=EXPECTED_ACCEPTED["fp"]):
        raise AssertionError("frozen ErrorPrediction 69D accepted events changed")
    def grouped(field):
        return {str(v):report_ratio([r for r in accepted if r[field]==v])
                for v in sorted(set(r[field] for r in accepted))}
    parent=[r for r in accepted if r["causal_prior_native_parent_80ms"]]
    no_parent=[r for r in accepted if not r["causal_prior_native_parent_80ms"]]
    removed=report_ratio(parent)
    if len(removed)!=4:raise AssertionError("ratio report schema drift")
    # This is a descriptive event-level counterfactual, NOT real WAV replay.
    retained=report_ratio(no_parent)
    gt_total=11267
    naive_f1=2*retained["tp"]/(retained["tp"]+retained["fp"]+gt_total)
    baseline_f1=2*baseline["tp"]/(baseline["tp"]+baseline["fp"]+gt_total)
    result=dict(schema=SCHEMA,complete=True,
        issue=HYPOTHESIS_ISSUE,
        source_s613_S701_training_samples=256,
        event_inference_decision_samples=256,
        source_native_events=len(all_computed),
        known_s613_original=EXPECTED_RAW,
        frozen_error69_accepted=baseline,
        parent_neighbor_80ms=dict(near=report_ratio(parent),
                                  other=report_ratio(no_parent)),
        by_texture=grouped("texture"),
        by_register=grouped("register"),
        by_composition=grouped("composition"),
        by_parent_fundamental_dominant=grouped("stronger_lower_fundamental"),
        by_lower_parent_sustain=grouped("stronger_lower_sustain"),
        by_harmonic_ground_truth_POSTHOC=grouped(
            "harmonic_lower_pitch_in_gt_80ms_POSTHOC"),
        by_recent_same_pitch=grouped("causal_prior_native_same_pitch_80ms"),
        naive_parent_veto_counterfactual_ONLY=dict(
            tp=retained["tp"],fp=retained["fp"],
            fn=gt_total-retained["tp"],event_only_f1=naive_f1,
            removes_tp=removed["tp"],removes_fp=removed["fp"]),
        original_three_head_f1=baseline_f1,
        original_four_head_f1=0.6790945406125166,
        label_used_only_POST_inference=True,
        acoustic_future_features_used=False,
        GT_not_in_training_vector=True,
        causal_intervention_confirmed=False,
        hypotheses_not_proven=True,
        source_full_audio_replay_previously_verified=True,
        this_is_NOT_a_new_live_replay=True,
        this_is_NOT_a_model_improvement=True,
        other_axes_unchanged=True,
        official_midi_unchanged=True,
        promoted=False,
        frozen_checkpoint_shas={k:v[2] for k,v in trained.items()})
    Path(out).mkdir(parents=True)
    (Path(out)/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    with (Path(out)/"fp_signals74.jsonl").open("w") as f:
        for record in accepted:
            f.write(json.dumps(record,sort_keys=True)+"\n")
    print("ONSET_FP_MLP69_REAL_EVENT_FORENSIC_256_FINAL",json.dumps({
        k:result[k] for k in (
            "source_native_events","frozen_error69_accepted","by_texture",
            "by_register","parent_neighbor_80ms",
            "naive_parent_veto_counterfactual_ONLY")
        }),flush=True)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--events",type=Path,required=True)
    p.add_argument("--models",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.events,a.models,a.out)
if __name__=="__main__":main()
