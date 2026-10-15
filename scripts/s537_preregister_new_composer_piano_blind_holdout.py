"""S537 preregister never-before-used-in-S469 MAESTRO v3 TEST piano songs.

Download OFFICIAL MAESTRO metadata CSV only. From official test split,
choose five *additional* composers and recordings with no overlap with
original S469 3 composers or filenames, using fixed SHA order.

Do NOT open or download any WAV, MIDI, labels, audio embeddings, or
model predictions. This only preselects a future independent test cohort.
Its suitability and ultimate blindness need independent provenance
checks against OTHER historical project sources, and its MIDI labels
must remain sealed until S538/S539 fixed audio inference is frozen.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,urllib.request
from pathlib import Path
from scripts.s469_preregister_MAESTRO_v3_third_source_filename_only import (
    URL,AUDIO_MIRROR,OFFSET,SECONDS)

N=5
# Broaden metadata-only duration bounds for 5 unseen composers after source
# eligibility check; WAV/MIDI remained unopened. Full WAV download must
# still be byte-size checked at the future source-inference stage.
MIN_DURATION=40.0
MAX_DURATION=240.0
def need(x,msg):
    if not x:raise ValueError(msg)
def digest(x):return hashlib.sha256(x).hexdigest()

def select(rows,old):
    previous_files={q["audio_filename"] for q in old}
    previous_composers={q["composer"] for q in old}
    selected=[]
    choices=[]
    for row in rows:
        try:duration=float(row["duration"])
        except (ValueError,KeyError):continue
        wav=row.get("audio_filename","")
        mid=row.get("midi_filename","")
        artist=row.get("canonical_composer","")
        if (row.get("split")=="test" and
            MIN_DURATION<=duration<=MAX_DURATION
            and wav.endswith(".wav") and mid.endswith(".midi")
            and wav.rsplit(".",1)[0]==mid.rsplit(".",1)[0]
            and ".." not in wav and ".." not in mid
            and not wav.startswith("/") and not mid.startswith("/")
            and wav not in previous_files and artist not in previous_composers
            and artist):
            choices.append(row)
    choices.sort(key=lambda row:digest(
        ("S537-independent-holdout-filename-only/"+row["audio_filename"]).encode()))
    artists=set()
    for row in choices:
        artist=row["canonical_composer"]
        if artist in artists:continue
        artists.add(artist)
        wav=row["audio_filename"]
        selected.append(dict(
          composer=artist,original_MAESTRO_split="test",
          audio_filename=wav,
          MIDI_filename_PATH_ONLY_DO_NOT_OPEN=row["midi_filename"],
          composition=row["canonical_title"],
          WAV_filename_sha256=digest(wav.encode()),
          metadata_duration_s=float(row["duration"]),
          frozen_excerpt_offset_s=OFFSET,
          frozen_excerpt_length_s=SECONDS))
        if len(selected)==N:break
    need(len(selected)==N and len(artists)==N and
         all(x["audio_filename"] not in previous_files and
             x["composer"] not in previous_composers for x in selected),
         "S537 held composers and filenames must be different from all S469 sources")
    return selected,len(choices)

def run(a):
    need(not a.out.exists(),"S537 preregistration immutable")
    old=json.loads(a.s469.read_text())
    need(old["status"]=="S469_PREREGISTERED_MAESTRO_V3_THIRD_SOURCE_FILENAME_ONLY"
         and not old["WAV_file_contents_read"]
         and not old["MIDI_note_contents_read"]
         and old["selected_piano_recordings"]==3,
         "S537 original independent S469 manifest proof not valid")
    request=urllib.request.Request(URL,headers={
        "User-Agent":"S537-metadata-only-new-cohort/1.0",
        "Accept":"text/csv"})
    with urllib.request.urlopen(request,timeout=90) as r:
        b=r.read(1_000_001)
        need(r.status==200 and b"audio_filename" in b[:500]
             and 150000<len(b)<900000,
             "S537 official MAESTRO CSV unavailable or changed unexpectedly")
    parsed=list(csv.DictReader(io.StringIO(b.decode("utf-8-sig"))))
    need(1200<=len(parsed)<=1350,
         "S537 official MAESTRO v3 metadata row count changed")
    chosen,count=select(parsed,old["selected_sources"])
    a.out.mkdir(parents=True)
    report=dict(status="S537_PREREGISTERED_DISTINCT_MAESTRO_V3_TEST_PIANO_COHORT_NO_WAV_NO_MIDI",
       original_metadata_URL=URL,official_csv_sha256=digest(b),
       old_S469_manifest_sha256=digest(a.s469.read_bytes()),
       selected_recordings=N,
       selected_different_composers_from_S469=True,
       selected_different_audio_filenames_from_S469=True,
       prospective_exclusion_verified_ONLY_AGAINST_S469_not_all_repo_history=True,
       MAESTRO_original_source_official_TEST_split=True,
       metadata_candidate_eligible_tracks_after_exclusion=count,
       preregistration_fixed_before_audio_or_labels=True,
       WAV_contents_downloaded_or_inspected=False,
       MIDI_contents_downloaded_or_inspected=False,
       model_predictions_or_weights_used_to_choose=False,
       original_S80_ExactK_and_Live_unchanged=True,
       no_promotion=True,
       audio_mirror_directory=AUDIO_MIRROR,
       selected_sources=chosen)
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S537 FIVE SEPARATE NEW MAESTRO COMPOSERS PRELABEL",
           json.dumps(dict(new_audio_filenames=[q["audio_filename"] for q in chosen],
                           candidate_count=count,original_csv_sha=digest(b))),flush=True)

def selftest():
    r=[dict(split="test",duration="80",audio_filename=f"a/{p}.wav",
            midi_filename=f"a/{p}.midi",
            canonical_composer=p,canonical_title="demo") for p in "ABCDEFGH"]
    old=[dict(audio_filename="a/A.wav",composer="A"),
         dict(audio_filename="a/B.wav",composer="B"),
         dict(audio_filename="a/C.wav",composer="C")]
    a,_=select(r,old)
    b,_=select(list(reversed(r)),old)
    need(a==b and len(a)==5 and all(z["composer"] not in ("A","B","C") for z in a),
         "S537 independent holdout metadata cannot be chosen using ordering or exposed songs")
    print("S537 SELFTEST PASS five new source composers preregistered without MIDI")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    p.add_argument("--s469",type=Path);p.add_argument("--out",type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(a.s469 is not None and a.out is not None,"S537 needs original S469 and output")
        run(a)
