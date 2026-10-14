"""S515 — genuinely NEW multi-label pitch FEATURES from raw audio at each
pre-frozen canonical note birth, not an ad hoc top12 detector.

For each of the 294 S498 canonical BP birth groups on 5 original IDMT
guitar + 3 original MAESTRO piano WAV clips, verify WAV SHA against
historical S277/S470 and compute pitch-wise spectral/harmonic/onset
observations (MIDI40..88). Preserve independent raw S511 evidence;
add observable existing BP notes as one head, NOT MIDI annotations.
This is the candidate-generator feature bank for real held-song learned
multi-label *note* heads inspired by midi frame/onset/harmonic separation
and YourMT3+ mixture experts. It is NOT YourMT3+ pretrained.

Every event retains a separate score for every MIDI pitch, including
previously ABSENT BP notes. The classifier must train only on OTHER
recordings and will be tested later. Original S80/Live not modified.
"""
from __future__ import annotations
import argparse,json,hashlib,collections
from pathlib import Path
import numpy as np
from scripts.s511_new_raw_harmonic_pitch_onset_generators import (
    parse_sources,per_pitch_harmonic_scores,SR,MIDIS,sha)
def need(c,m):
    if not c:raise ValueError(m)
def filesha(p):return sha(Path(p).read_bytes())
def features(sound,onsets,notes):
    times,raw,rel,rise,physical=per_pitch_harmonic_scores(sound)
    X=[]
    onset_note_map=collections.defaultdict(list)
    for n in notes:onset_note_map[int(n["pitch_MIDI"])].append(n)
    for g in onsets:
        onset=float(g["event_time_seconds"])
        near=int(np.argmin(np.abs(times-onset)))
        lo=max(0,near-2);hi=min(raw.shape[1],near+3)
        start=max(0,near-13);end=max(1,near-4)
        energy=raw[:,lo:hi].mean(axis=1)
        spread=np.quantile(energy,.9)-np.quantile(energy,.1)
        absolute=np.clip((energy-np.quantile(energy,.1))/max(.05,spread),0,2.5)
        strength=rel[:,lo:hi].max(axis=1)
        burst=rise[:,lo:hi].max(axis=1)
        sustain=rel[:,start:end].mean(axis=1)
        novelty=np.maximum(strength-sustain,0)
        birth_set=set(map(int,g["bp_note_pitches"]))
        active_set=set(map(int,g["bp_active_pitches_at_birth_plus30"]))
        active_at_onset_plus30=float(onset)+.030
        row=[]
        for j,pitch in enumerate(MIDIS):
            p=int(pitch)
            predicted=[n for n in onset_note_map.get(p,[])
                       if onset-.0002<=float(n["predicted_onset_s"])<=onset+.0202]
            support=max([float(n.get("pretrained_note_score",0)) for n in predicted],
                        default=0.)
            # Could the current pitch be a harmonic of another observed
            # FUNDAMENTAL? Evidence only, NOT a hard octave removal.
            harmonic=any((p-base) in (12,19,24)
                         for base in active_set if base<p)
            # Tonic octave alone is NOT proof of a duplicate note.
            row.append([p/127.,float(absolute[j]),float(strength[j]),
                float(burst[j]),float(sustain[j]),float(novelty[j]),
                float(p in birth_set),float(p in active_set),
                float(support),float(harmonic),
                float(np.clip(physical[near]/5.,0,2.))])
        features=np.asarray(row,np.float32)
        need(features.shape==(49,11) and np.isfinite(features).all(),
             "S515 49 observed MIDI pitches times 11 audio-only features required")
        X.append(features)
    return X

def run(a):
    need(not a.out.exists(),"S515 before-MIDI source spectral note bank already frozen")
    s498=json.loads(a.canonical_report.read_text())
    bpG=json.loads(a.guitar_notes_report.read_text())
    bpP=json.loads(a.piano_notes_report.read_text())
    need(s498["status"]=="S498_TWO_SOURCE_CANONICAL_NOTE_ONSET_GRAPH_SOURCE_K_AS_SOFT_PRIOR_PRELABEL"
         and s498["no_original_MIDI_XML_annotations_read_in_canonical_model"]
         and bpG["status"]=="S487_FIVE_GUITAR_IDMT_REAL_WAV_PRETRAINED_BASIC_PITCH_FROZEN_NOXML"
         and bpP["status"]=="S485_PRETRAINED_BASIC_PITCH_NOTE_EVENTS_ON_THIRD_SOURCE_MAESTRO_PREMIDI"
         and bpG["all_output_note_events_sha256"]==filesha(a.guitar_notes)
         and bpP["frozen_notes_SHA256"]==filesha(a.piano_notes),
         "S515 original stereo-to-mono notes must be frozen BEFORE XML/MIDI")
    srcs=parse_sources(a.s276,a.s277,a.s277_report,a.s469,a.s470,True)
    audio={ident:(kind,wave,whole) for ident,kind,wave,whole in srcs}
    need(len(audio)==8,"S515 expected genuine 5 guitar and 3 piano recordings")
    base_notes={}
    for k,file in (("guitar_S80",a.guitar_notes),("piano_V88",a.piano_notes)):
        q=[json.loads(s) for s in file.read_text().splitlines()]
        cache=collections.defaultdict(list)
        for e in q:cache[e["WAV_filename_SHA256"]].append(e)
        base_notes[k]=cache
    output=[];stats={}
    for kind in ("guitar_S80","piano_V88"):
        original=a.canonical/f"S498_{kind}_CANONICAL_MUSICAL_EVENTS_PRELABEL.jsonl"
        need(s498["frozen_group_sha256"][kind]==filesha(original),
             "S515 original canonical music births changed")
        items=[json.loads(s) for s in original.read_text().splitlines()]
        group=collections.defaultdict(list)
        for e in items:group[e["source_recording_sha"]].append(e)
        count=0
        for ident,glist in group.items():
            need(ident in audio and audio[ident][0]==kind,
                 "S515 actual original WAV source ID not paired")
            computed=features(audio[ident][1],glist,base_notes[kind][ident])
            for e,feat in zip(glist,computed):
                output.append(dict(canonical_event_id=e["canonical_event_id"],
                    original_audio_sha=audio[ident][2],
                    source=kind,source_recording_sha=ident,
                    frozen_BP_note_pitches=e["bp_note_pitches"],
                    independent_audio_MIDI_range=[40,88],
                    per_midi_features_49x11=feat.round(6).tolist(),
                    true_XML_or_MIDI_not_used=True))
                count+=1
        need(count==s498["source_summaries"][kind]["canonical_musical_onset_groups"],
             "S515 source original 60/234 music birth count changed")
        stats[kind]=dict(recordings=len(group),events=count,
            per_pitch_features=11,MIDI_pitch_candidates=49,
            time_domain="pretrained-canonical-note-onset-observed",
            no_groundtruth_used=True)
        print("S515 TRUE SOUND FRAME ONSET PER PITCH",json.dumps(
              dict(corpus=kind,**stats[kind])),flush=True)
    a.out.mkdir(parents=True)
    fp=a.out/"S515_TWO_INSTRUMENT_49_PITCH_REAL_AUDIO_FEATURES_PREMIDI.jsonl"
    fp.write_text("".join(json.dumps(o)+"\n" for o in output))
    rep=dict(status="S515_TRUE_EIGHT_SOURCE_WAV_INDEPENDENT_MULTILABEL_NOTE_FEATURES_PRETRUTH",
        source_number=8,source_original_complete_guitar_S80=True,
        source_original_piano_V88_only=True,
        source_wav_bytes_sha_verified=True,
        all_predicted_pitches_40to88_not_just_BP_candidates=True,
        no_original_MIDI_XML_labels_read_to_generate_features=True,
        separate_11_pitch_temporal_harmonic_sustain_BP_features=True,
        not_a_pretrained_YourMT3_model=True,
        offline_noncausal_not_a_live_latency_score=True,
        no_trained_pitch_note_classifier_yet=True,no_promotion=True,
        original_S80_Live_onsets_offsets_unchanged=True,
        by_source=stats,real_audio_feature_SHA256=filesha(fp),
        source_SHA256=dict(original_S498=filesha(a.canonical_report),
             original_BP_guitar=filesha(a.guitar_notes),
             original_BP_piano=filesha(a.piano_notes)))
    (a.out/"report.json").write_text(json.dumps(rep,indent=2)+"\n")
def selftest():
    # Real audio-derived note spectral profiles, synthetic fixture is
    # ONLY shape/finite test and not a claimed note-detection score.
    sr=SR;wave=np.zeros(2*sr,np.float32)
    t=np.arange(sr)/sr
    wave[int(.2*sr):int(1.2*sr)]=(.3*np.exp(-2*t)*np.sin(2*np.pi*164.81*t)).astype(np.float32)
    event=dict(event_time_seconds=.22,bp_note_pitches=[52],
               bp_active_pitches_at_birth_plus30=[52])
    observed=[dict(pitch_MIDI=52,predicted_onset_s=.22,
                   predicted_offset_s=.8,pretrained_note_score=.9)]
    f=features(wave,[event],observed)
    need(len(f)==1 and f[0].shape==(49,11)
          and float(f[0][12,6])==1.
          and np.isfinite(f[0]).all(),
          "S515 must preserve real E3 source evidence and all 49 note candidates")
    print("S515 SELFTEST PASS raw sound 49-pitch 11-head stereo source feature contract")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for key in ("s276","s277","s277-report","s469","s470",
                "canonical","canonical-report","guitar-notes",
                "guitar-notes-report","piano-notes","piano-notes-report","out"):
        p.add_argument("--"+key,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for key in ("s276","s277","s277_report","s469","s470",
                    "canonical","canonical_report","guitar_notes",
                    "guitar_notes_report","piano_notes","piano_notes_report","out"):
            need(getattr(a,key)is not None,"S515 missing --"+key)
        run(a)
