"""S493 forensic: why cautious S491 K0 note-birth routing misses all 7 true IDMT K0.

Use only immutable S277/S297 original source times, frozen independent
pretrained S487 notes, fixed S491 one-to-one matching, S488 original truth.
Correlate true XML onset vs native event timestamp and Basic Pitch estimated
note-on times + active durations, without inventing new K0 ground truth
on 459 unmatched events. No model retraining, router selection or promotion.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
def need(condition,msg):
    if not condition:raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def active(notes,t):
    return sorted({int(v["pitch_MIDI"]) for v in notes if
        float(v["predicted_onset_s"])<=t<float(v["predicted_offset_s"])})
def nearest_onset(notes,t,pitches=None):
    if pitches is not None:
        notes=[n for n in notes if int(n["pitch_MIDI"]) in pitches]
    if not notes:return None
    best=min(notes,key=lambda n:abs(float(n["predicted_onset_s"])-t))
    return dict(delta_ms=round((float(best["predicted_onset_s"])-t)*1000,3),
        pitch_MIDI=int(best["pitch_MIDI"]),
        absolute_onset_s=float(best["predicted_onset_s"]),
        abs_difference_ms=round(abs(float(best["predicted_onset_s"])-t)*1000,3))
def run(s277,s297,s487,s487_report,s491,s491_report,s488,s488_report,out):
    need(not out.exists(),"S493 forensic original matched source output already written")
    vr=json.loads(s487_report.read_text())
    sr=json.loads(s491_report.read_text())
    tr=json.loads(s488_report.read_text())
    need(vr["status"]=="S487_FIVE_GUITAR_IDMT_REAL_WAV_PRETRAINED_BASIC_PITCH_FROZEN_NOXML"
         and vr["all_output_note_events_sha256"]==sha(s487)
         and sr["status"]=="S491_TWO_SOURCE_K0_ONLY_ONE_TO_ONE_EXTERNAL_PITCH_BIRTH_FROZEN_NO_TRUTH"
         and sr["source_output_SHA256"]["guitar_S80"]["sha256"]==sha(s491)
         and tr["status"]=="S488_PAIRED_IDMT_45_TRUE_GUITAR_ATTACKS_MULTISOURCE_BASIC_PITCH"
         and tr["matched_rows_SHA256"]==sha(s488),
         "S493 source original guitar byte-level postfreeze provenance mismatch")
    with np.load(s277,allow_pickle=False) as z:
        ids=z["row_id"].astype(str)
        members=z["member"].astype(str)
        times=z["start_samples"].astype(np.int64)/44100.
    with np.load(s297,allow_pickle=False) as z:
        K=z["s80_K"].astype(int)
        need(np.array_equal(ids,z["row_id"].astype(str)),
             "S493 genuine original trained S80 identity drifted")
    with np.load(s491,allow_pickle=False) as z:
        need(np.array_equal(ids,z["row_id"].astype(str))
             and np.array_equal(K,z["source_K"]),
             "S493 source K0 shadow comparisons not same original V88 events")
        gated=z["independent_BP_note_onset_group_matched"].astype(bool)
        birth=z["proposed_only_K0_birth"].astype(int)
        active30=z["proposed_only_K0_active30"].astype(int)
    notes=[json.loads(v) for v in s487.read_text().splitlines()]
    annotations=[json.loads(v) for v in s488.read_text().splitlines()]
    byaudio=collections.defaultdict(list)
    for x in notes:byaudio[str(x["WAV_filename_SHA256"])].append(x)
    rows={k:i for i,k in enumerate(ids)}
    true_K0=[x for x in annotations if int(x["original_S80_K"])==0]
    need(len(true_K0)==7 and len(ids)==506,
         "S493 original seven truly annotated wrong-S80-K0 events disappeared")
    details=[];flags=collections.Counter()
    for item in true_K0:
        i=rows[item["row_id"]]
        need(K[i]==0 and members[i]==item["source_wav"] and
             len(item["true_active"])>0,
             "S493 original IDMT wrong K0 acoustic source event mislabeled")
        wave=item["source_wav"]
        sound=byaudio[wave]
        native=float(times[i])
        xml=float(item["original_XML_anchor"])
        dxml=round((native-xml)*1000.,3)
        pxml=active(sound,xml+.030)
        pnative=active(sound,native+.030)
        needed=set(int(k) for k in item["true_active"])
        matches=[v for v in sound if int(v["pitch_MIDI"]) in needed]
        prior=[v for v in sound if float(v["predicted_onset_s"])<native-.080
               and float(v["predicted_offset_s"])>native+.030]
        neighbors=np.flatnonzero((members==wave)&
                 (np.abs(times-native)<=.080))
        otherBP=[int(K[j]) for j in neighbors if j!=i]
        nearest=nearest_onset(matches,native)
        nearestxml=nearest_onset(matches,xml)
        event=dict(row_id=item["row_id"],recording_SHA=wave,
            true_active_K=len(needed),true_MIDI=sorted(needed),
            basicpitch_active_at_original_XML_plus30=pxml,
            basicpitch_active_at_original_V88_plus30=pnative,
            source_K0_to_positive_birth_proposal=int(birth[i]),
            source_K0_to_positive_active_proposal=int(active30[i]),
            native_event_time_s=round(native,6),
            original_XML_attack_time_s=round(xml,6),
            native_event_minus_XML_onset_ms=dxml,
            nearest_BP_true_pitch_birth_to_native=nearest,
            nearest_BP_true_pitch_birth_to_XML=nearestxml,
            original_native_V88_events_within_80ms_including_self=int(len(neighbors)),
            other_source_S80_K_nearby=otherBP,
            potential_SUSTAINED_BP_note_started_more_than80ms_before_native=
                len(prior)>0,
            V88_native_exact_BP_birth_group_paired=bool(gated[i]))
        flags.update(total=1,
            BP_truth_pitch_active_at_native=int(needed.issubset(set(pnative))),
            BP_truth_pitch_active_at_XML=int(needed.issubset(set(pxml))),
            BP_truth_pitch_new_onset_within80ms_native=int(
                nearest is not None and nearest["abs_difference_ms"]<=80),
            BP_truth_pitch_new_onset_within80ms_XML=int(
                nearestxml is not None and nearestxml["abs_difference_ms"]<=80),
            not_matched_independent_BP_birth=int(not gated[i]),
            native_time_misaligned_over30ms=int(abs(dxml)>30),
            native_time_misaligned_over60ms=int(abs(dxml)>60),
            adjacent_competing_native_events=int(len(neighbors)>1),
            BP_pitch_had_already_started_sustaining=int(len(prior)>0))
        details.append(event)
        print("S493 TRUE K0 EXPLANATION",json.dumps(event),flush=True)
    need(flags["total"]==7 and flags["not_matched_independent_BP_birth"]==7,
         "S493 source SHA original K0 mismatch should prove S491 missed 7/7")
    out.mkdir(parents=True)
    filepath=out/"S493_7_real_wrong_S80_K0_pitch_onset_alignment_details.jsonl"
    filepath.write_text("".join(json.dumps(o)+"\n" for o in details))
    result=dict(status="S493_POSTFREEZE_ORIGINAL_IDMT_TRUE_K0_ONSET_TEMPORAL_MISMATCH_FORENSIC",
        source_original_S80_K0_real_XML_attacks=7,
        original_native_candidate_events=506,
        original_IDMT_unmatched_459_NOT_true_negative=True,
        full_original_S80_reference_modified=False,
        no_external_threshold_selection_or_retraining=True,
        no_promotion=True,
        cause_observation_counters=dict(flags),
        seven_original_XML_matched_note_attacks=details,
        SHA256=dict(original_V88=sha(s277),original_S80=sha(s297),
            independent_pitch_notes=sha(s487),
            prelabel_shadow_gating=sha(s491),
            real_annotations=sha(s488),
            detailed_ledger=sha(filepath)))
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S493 CAUSAL ONSET DIAGNOSIS",json.dumps(dict(flags)),flush=True)
def selftest():
    n=[dict(pitch_MIDI=52,predicted_onset_s=.1,predicted_offset_s=.5)]
    need(active(n,.2)==[52] and
         nearest_onset(n,.35)["abs_difference_ms"]==250,
         "S493 an acoustically sustained note is not a freshly born attack")
    print("S493 SELFTEST PASS distinguish sustained pitch from new note onset")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for k in ("s277","s297","s487","s487-report","s491","s491-report","s488","s488-report","out"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("s277","s297","s487","s487_report","s491","s491_report","s488","s488_report","out"):
            need(getattr(a,k)is not None,"S493 missing --"+k)
        run(a.s277,a.s297,a.s487,a.s487_report,a.s491,a.s491_report,a.s488,a.s488_report,a.out)
