"""S469: PREREGISTER third corpus MAESTRO v3.0.0 *metadata only*.

Choose 3 distinct-composer real acoustic piano tracks from the official v3
TEST split, using only public CSV metadata and a fixed sha256 ordering;
do not download or open WAV/MIDI notes at this stage. Subsequent S470 must
freeze original model predictions before S471 may read MIDI.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,urllib.request
from pathlib import Path
URL="https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv"
AUDIO_MIRROR="https://huggingface.co/datasets/ddPn08/maestro-v3.0.0/resolve/main/"
N=3
MIN_DURATION=55.
MAX_DURATION=145.
OFFSET=15.
SECONDS=12.
def need(x,m):
    if not x:raise ValueError(m)
def digest(blob):
    return hashlib.sha256(blob).hexdigest()
def choose(rows):
    allowed=[]
    for row in rows:
        try:duration=float(row["duration"])
        except (KeyError,ValueError):continue
        path=row.get("audio_filename","")
        midi=row.get("midi_filename","")
        if (row.get("split")=="test" and
            MIN_DURATION<=duration<=MAX_DURATION and
            path.endswith(".wav") and midi.endswith(".midi") and
            path.rsplit(".",1)[0]==midi.rsplit(".",1)[0] and
            ".." not in path and ".." not in midi and
            not path.startswith("/") and not midi.startswith("/")):
            allowed.append(row)
    allowed.sort(key=lambda row:digest(("S469-unaltered-maestro-test-"+row["audio_filename"]).encode()))
    selected=[];used_composers=set()
    for row in allowed:
        composer=row["canonical_composer"]
        if composer in used_composers:continue
        used_composers.add(composer)
        selected.append(dict(audio_filename=row["audio_filename"],
            MIDI_filename_path_ONLY_not_opened=row["midi_filename"],
            composer=row["canonical_composer"],title=row["canonical_title"],
            duration_seconds=float(row["duration"]),split="test",
            filename_SHA256=digest(row["audio_filename"].encode()),
            audio_excerpt_start_seconds=OFFSET,
            audio_excerpt_duration_seconds=SECONDS))
        if len(selected)==N:break
    need(len(selected)==N and len(used_composers)==N,
         "S469 official MAESTRO test split lacks 3 valid distinct composers")
    return selected,len(allowed)

def run(out):
    need(not out.exists(),"S469 metadata-only third source selection immutable")
    req=urllib.request.Request(URL,headers={"User-Agent":"S469-prospective-MAESTRO3-filename-only",
                                              "Accept":"text/csv"})
    with urllib.request.urlopen(req,timeout=75) as r:
        b=r.read(1_000_000)
        need(r.status==200 and len(b)>150_000 and len(b)<900_000,
             "S469 original MAESTRO v3 CSV metadata unavailable")
    source=list(csv.DictReader(io.StringIO(b.decode("utf-8-sig"))))
    need(1200<=len(source)<=1350,
         "S469 original MAESTRO version3 metadata row count changed")
    selected,n=choose(source)
    report=dict(status="S469_PREREGISTERED_MAESTRO_V3_THIRD_SOURCE_FILENAME_ONLY",
        official_metadata_URL=URL,official_metadata_SHA256=digest(b),
        original_metadata_rows=len(source),candidate_test_short_tracks=n,
        selected_piano_recordings=N,selected_sources=selected,
        verified_different_corpus_from_GuitarSet_and_IDMT=True,
        instrument_piano_not_guitar=True,
        official_MAESTRO_3ms_MIDI_audio_alignment_expected_not_verified=True,
        audio_mirror_directory=AUDIO_MIRROR,
        WAV_file_contents_read=False,MIDI_note_contents_read=False,
        original_weights_or_prediction_scores_read=False,
        selection_independent_of_any_audio_or_labels=True,
        prospective_subsequent_prediction_freeze_required=True,
        source_does_not_by_itself_confirm_original_S80_ExactK=True,
        no_promotion=True)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S469 THIRDSOURCE MANIFEST",json.dumps(dict(
        official_rows=len(source),test_tracks=n,
        chosen=[dict(composer=x["composer"],duration=x["duration_seconds"],
            audio_filename=x["audio_filename"]) for x in selected],
        MIDI_ANNOTATIONS_READ=False)),flush=True)

def selftest():
    q=[dict(canonical_composer=s,canonical_title="x",split="test",
      duration="65",audio_filename=f"2018/{s}.wav",midi_filename=f"2018/{s}.midi")
       for s in ["A","B","C"]]
    a,_=choose(q);b,_=choose(list(reversed(q)))
    need(a==b,"S469 filenames selection must not depend on CSV ordering")
    print("S469 SELFTEST PASS deterministic source selection, no annotation read")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:need(a.out is not None,"S469 missing out");run(a.out)
