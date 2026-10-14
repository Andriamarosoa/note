"""S489 — REAL sound (.wav/.mp3) -> polyphonic note names E3 G3 B3.

Independent, OFFLINE experimental decoder using pretrained Basic Pitch
0.4.0. Importantly, this is not a MIDI FILE generator: it prints actual
note names and their onset/offset and K per simultaneous birth group.
It is not the original S80 reference and is NOT promoted into Live.
The legacy V88/S80/offsets code is not imported or modified.
"""
from __future__ import annotations
import argparse,json,math,sys,contextlib
from pathlib import Path
MIN_MIDI=40;MAX_MIDI=88
WINDOW_SECONDS=.020
def note_name(pitch):
    names=("C","C#","D","D#","E","F","F#","G","G#","A","A#","B")
    midi=int(pitch)
    if not (0<=midi<=127):raise ValueError("invalid MIDI pitch")
    return names[midi%12]+str(midi//12-1)
def new_note_groups(events,low=MIN_MIDI,high=MAX_MIDI):
    """Only pitch onset groups, distinct from sustained/active-note count."""
    if not (0<=low<=high<=127):
        raise ValueError("MIDI target range must be within 0..127")
    selected=[]
    for event in events:
        # Basic Pitch event tuple: (onset_sec,offset_sec,pitch,confidence,bend).
        st,en,pitch,prob,*_=event
        midi=int(pitch)
        if not (low<=midi<=high and float(en)>float(st) and
                0<=float(st) and math.isfinite(float(st)) and
                math.isfinite(float(en)) and math.isfinite(float(prob))):
            continue
        selected.append((float(st),float(en),midi,float(prob)))
    selected.sort(key=lambda a:(a[0],a[2],-a[3]))
    groups=[]
    for st,en,pitch,prob in selected:
        if not groups or st-groups[-1]["onset_seconds"]>WINDOW_SECONDS:
            groups.append(dict(onset_seconds=round(st,4),notes={}))
        item=dict(name=note_name(pitch),pitch_MIDI=pitch,
                  offset_seconds=round(en,4),
                  confidence=round(prob,4))
        old=groups[-1]["notes"].get(pitch)
        if old is None or item["confidence"]>old["confidence"]:
            groups[-1]["notes"][pitch]=item
    for row in groups:
        row["notes"]=[row["notes"][k] for k in sorted(row["notes"])]
        row["K_new_notes"]=len(row["notes"])
        row["note_names"]=[x["name"] for x in row["notes"]]
        row["outside_original_ExactK0to6_range"]=row["K_new_notes"]>6
    return groups
def transcribe(audio:Path,low=MIN_MIDI,high=MAX_MIDI):
    if not audio.is_file():
        raise FileNotFoundError("Audio source file not found: "+str(audio))
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model,predict
    # Model progress messages must not corrupt machine-readable JSON stdout.
    with contextlib.redirect_stdout(sys.stderr):
        model=Model(ICASSP_2022_MODEL_PATH)
        # These remain BasicPitch's factory defaults, no source tuning.
        _,_,notes=predict(str(audio),model,
             onset_threshold=.5,frame_threshold=.3,
             minimum_note_length=127.7)
    groups=new_note_groups(notes,low,high)
    return dict(status="S489_EXPERIMENTAL_PRETRAINED_NOTE_NAME_DETECTOR",
        source_filename=audio.name,source_bytes=audio.stat().st_size,
        pretrained_model="Spotify Basic Pitch 0.4.0",
        original_trained_S80_reference_NOT_used=True,
        source_Live_and_onset_offset_unchanged=True,
        candidate_supports_polyphonic_new_note_groups=True,
        confidence_is_not_calibrated_probability_of_exact_note=True,
        MIDI_file_not_saved=True,
        pitch_range_MIDI=[low,high],
        total_predicted_note_events=sum(g["K_new_notes"] for g in groups),
        birth_groups=len(groups),
        note_events=groups,
        unpromoted_experimental_only=True)
def selftest():
    assert note_name(52)=="E3" and note_name(55)=="G3" and note_name(59)=="B3"
    example=[
        (.10,.40,52,.95,None),
        (.11,.46,55,.92,None),
        (.114,.51,59,.88,None),
        (.39,.70,64,.73,None)]
    z=new_note_groups(example)
    assert len(z)==2 and z[0]["note_names"]==["E3","G3","B3"]
    assert z[0]["K_new_notes"]==3 and z[1]["note_names"]==["E4"]
    assert not z[0]["outside_original_ExactK0to6_range"]
    print("S489 SELFTEST PASS example note event schema (not audio inference)")
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio",type=Path)
    parser.add_argument("--min-midi",type=int,default=MIN_MIDI)
    parser.add_argument("--max-midi",type=int,default=MAX_MIDI)
    parser.add_argument("--pretty",action="store_true")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:selftest();return
    if args.audio is None:parser.error("--audio required unless --self-test")
    result=transcribe(args.audio,args.min_midi,args.max_midi)
    print(json.dumps(result,ensure_ascii=False,indent=2 if args.pretty else None))
if __name__=="__main__":main()
