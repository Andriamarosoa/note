"""S514 — external-source piano forensic of S513 real raw onset generator.

Compare truly original 322 independent MIDI-onset groups on three MAESTRO
excerpts to the already SHA-frozen 19 physical onset groups from S513.
This is SOURCE-DIRECTION failure diagnosis. Do not confuse 322 new-note
onsets with IDMT active-note semantics. All missing onset groups count
as false NEGATIVE on the known positive MIDI attack cohort; unpaired
raw model groups are UNKNOWN until negative onset annotations exist.
No model thresholds calibrated using the exposed MAESTRO labels.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
def need(c,m):
    if not c:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def match(real,pred):
    if not real or not pred:return {}
    a=np.array(real,float);b=np.array(pred,float)
    d=np.abs(a[:,None]-b[None,:])
    matrix=np.concatenate((np.where(d<=.080,d/.08*.5,1e6),
                           np.ones((len(a),len(a)))),axis=1)
    i,j=linear_sum_assignment(matrix)
    return {int(x):int(y) for x,y in zip(i,j) if y<len(b) and d[x,y]<=.080}
def run(args):
    need(not args.out.exists(),"S514 immutable source diagnostic already exists")
    pre=json.loads(args.raw_report.read_text())
    original=json.loads(args.piano_report.read_text())
    need(pre["status"]=="S511_NEW_INDEPENDENT_RAW_HARMONIC_PITCH_ONSET_CANDIDATES_PREMIDI"
         and pre["original_WAVs"]==8 and pre["candidates_sha256"]==sha(args.raw)
         and pre["no_XML_or_MIDI_ground_truth_read_in_training_or_inference"]
         and original["status"]=="S476_THIRD_MAESTRO_PIANO_NEW_ONSET_K_VS_SUSTAINING_K_READONLY"
         and original["SHA256"]["paired_label_comparison"]==sha(args.piano),
         "S514 raw source piano onsets MUST be frozen before real MIDI")
    allraw=[json.loads(x) for x in args.raw.read_text().splitlines()]
    guitar=[x for x in allraw if x["source"]=="guitar_S80"]
    piano=[x for x in allraw if x["source"]=="piano_V88"]
    labels=[json.loads(x) for x in args.piano.read_text().splitlines()]
    need(len(labels)==322 and sum(len(x["true_new_onset_MIDI_pitches"])>=2
                                  for x in labels)==73,
         "S514 true 322 piano MIDI onsets and 73 new-note poly unchanged")
    byrec=collections.defaultdict(list)
    for x in labels:byrec[x["row_id"].split(":")[0]].append(x)
    rows=collections.defaultdict(list)
    for x in piano:
        if x["policy"]=="top6":rows[x["WAV_filename_SHA256"]].append(x)
    reports={}
    total=collections.Counter()
    for source,truth in byrec.items():
        raw=sorted(rows[source],key=lambda r:r["independent_physical_onset_s"])
        pairs=match([v["onset_at_excerpt_s"] for v in truth],
                    [v["independent_physical_onset_s"] for v in raw])
        tp_pitch=sum(len(set(map(int,truth[i]["true_new_onset_MIDI_pitches"]))&
                      set(map(int,raw[j]["candidate_MIDI"])))
                     for i,j in pairs.items())
        pitch_total=sum(len(t["true_new_onset_MIDI_pitches"]) for t in truth)
        d=dict(real_annotated_MIDI_onset_groups=len(truth),
            raw_physical_onset_candidates=len(raw),
            physical_attacks_matched_80ms=len(pairs),
            genuinely_annotated_attack_groups_missed=len(truth)-len(pairs),
            annotated_true_note_pitches=pitch_total,
            genuine_pitch_notes_in_same_matched_physical_top6=tp_pitch,
            unpaired_raw_attacks_truth_UNKNOWN=len(raw)-len(pairs))
        reports[source]=d
        for key,v in d.items():total[key]+=v
    need(sum(x["raw_physical_onset_candidates"] for x in reports.values())==19
         and len(reports)==3,
         "S514 new source 19 piano physical detection groups drifted")
    args.out.mkdir(parents=True)
    result=dict(status="S514_RAW_HARMONIC_ONSET_GENERATOR_PIANO_DOMAIN_FAILURE_AUDIT",
        guitar_original_full_S80_not_part_of_this_piano_scoring=True,
        piano_original_V88_only_not_promoted=True,
        true_piano_label_target_IS_new_note_birth_not_active_sustain=True,
        physical_global_flux_gate_not_robust_across_instruments=True,
        notes_onset_matching_one_to_one_max80ms=True,
        all_322_true_piano_positive_events_in_denominator=True,
        unpaired_model_onsets_NOT_false_negatives_or_false_positives_proven=True,
        exposed_MAESTRO_tracks_not_new_blind=True,
        no_promotion=True,
        by_recording=reports,totals=dict(total),
        sourceSHA=dict(original_S513_raw=sha(args.raw),
                       original_S476_MIDI=sha(args.piano)))
    (args.out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S514 PIANO SOURCE ONSET FAILURE",json.dumps(dict(total)),flush=True)
def selftest():
    a=match([.1,.12],[.11])
    need(len(a)==1,"S514 one physical onset cannot justify multiple true MIDI attacks")
    print("S514 SELFTEST PASS no duplicate true onset attribution")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for k in ("raw","raw-report","piano","piano-report","out"):p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("raw","raw_report","piano","piano_report","out"):
            need(getattr(a,k)is not None,"S514 missing "+k)
        run(a)
