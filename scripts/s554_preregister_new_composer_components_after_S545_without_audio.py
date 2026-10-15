"""S554 next completely source-disjoint final evaluation cohort preregistration.

Only MAESTRO v3 public CSV metadata. Choose 5 fixed-sha distinct composers
from official TRAIN split (not earlier old S469 three / held S537 five)
with 40-240s durations. This keeps a completely NEW source domain after
the 8-composer S544 development; split is called MAESTRO TRAIN and cannot
be described as official MAESTRO test score. No WAV or MIDI labels read
or model outcomes consulted before choosing these future validation files.

Future post-label validation must check entire repo historical exposure
and freeze all five source NoteOn outputs before opening annotations.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,urllib.request
from pathlib import Path
from scripts.s469_preregister_MAESTRO_v3_third_source_filename_only import (
    URL,AUDIO_MIRROR,OFFSET,SECONDS)
def digest(b):return hashlib.sha256(b).hexdigest()
def need(x,msg):
    if not x:raise ValueError(msg)
N=5

def components(artist):
    return {x.strip() for x in str(artist).split(" / ") if x.strip()}

def choose(rows,previous_names,previous_composers):
    excluded_components=set().union(*(components(a) for a in previous_composers))
    found=[]
    for r in rows:
        try:seconds=float(r["duration"])
        except (ValueError,KeyError):continue
        wav=r.get("audio_filename","")
        midi=r.get("midi_filename","")
        artist=r.get("canonical_composer","")
        if (r.get("split")=="train" and 40<=seconds<=240
            and wav.endswith(".wav") and midi.endswith(".midi")
            and wav.rsplit(".",1)[0]==midi.rsplit(".",1)[0]
            and ".." not in wav and ".." not in midi
            and not wav.startswith("/") and not midi.startswith("/")
            and wav not in previous_names and artist not in previous_composers
            and not (components(artist)&excluded_components)
            and artist):
            found.append(r)
    found.sort(key=lambda z:digest(
       ("S554-after-13-song-previous-source-registrations-new-mid-gate-true-prospective-v1/"+
        z["audio_filename"]).encode()))
    composers=set();included_components=set();selected=[]
    for r in found:
        artist=r["canonical_composer"]
        if artist in composers or components(artist)&included_components:continue
        composers.add(artist)
        included_components.update(components(artist))
        wav=r["audio_filename"]
        selected.append(dict(composer=artist,composition=r["canonical_title"],
           official_MAESTRO_split="train",
           audio_filename=wav,
           MIDI_filename_path_ONLY_not_opened=r["midi_filename"],
           filename_SHA256=digest(wav.encode()),
           source_duration_seconds=float(r["duration"]),
           audio_excerpt_start_seconds=OFFSET,
           audio_excerpt_duration_seconds=SECONDS))
        if len(selected)==N:break
    need(len(selected)==N and len(composers)==N
         and all(q["composer"] not in previous_composers
                 and not (components(q["composer"])&excluded_components)
                 and q["audio_filename"] not in previous_names for q in selected),
         "S554 all five next validation composers must be new to eight prior development recordings")
    return selected,len(found)

def run(a):
    need(not a.out.exists(),"S554 pre-MIDI new final held-out metadata already selected")
    old=json.loads(a.old.read_text());new=json.loads(a.new.read_text());prior=json.loads(a.prior.read_text())
    need(old["status"]=="S469_PREREGISTERED_MAESTRO_V3_THIRD_SOURCE_FILENAME_ONLY"
         and new["status"]=="S537_PREREGISTERED_DISTINCT_MAESTRO_V3_TEST_PIANO_COHORT_NO_WAV_NO_MIDI"
         and old["selected_piano_recordings"]==3
         and new["selected_recordings"]==5
         and prior["status"]=="S545_PREREGISTERED_FIVE_MORE_TRAIN_SPLIT_COMPOSERS_AFTER_S544_NO_WAV_MIDI"
         and prior["five_next_distinct_composers"]==5
         and prior["co_composer_COMPONENTS_fully_excluded_and_pairwise_disjoint"]
         and prior["source_MIDI_files_not_read"]
         and new["official_csv_sha256"]==
             "cb88a64a12971a995dd41852aa3a33879501cf70d01b9a6feb637f4ad8d8ddff",
         "S554 source original 3+5 full SHA cohort invalid")
    history_names={s["audio_filename"] for s in old["selected_sources"]+new["selected_sources"]+prior["chosen_metadata_only"]}
    history_artist={s["composer"] for s in old["selected_sources"]+new["selected_sources"]}
    need(len(history_names)==13 and len(history_artist)>=9,
         "S554 all 13 prior source filenames and their composer components must be excluded")
    req=urllib.request.Request(URL,headers={
        "Accept":"text/csv","User-Agent":"S554-next-composer-metadata-only/1.0"})
    with urllib.request.urlopen(req,timeout=75) as resp:
        b=resp.read(1_000_000)
        need(resp.status==200 and 150000<len(b)<900000
             and digest(b)==new["official_csv_sha256"],
             "S554 independent official MAESTRO exact CSV metadata proof changed")
    rows=list(csv.DictReader(io.StringIO(b.decode("utf-8-sig"))))
    selected,eligible=choose(rows,history_names,history_artist)
    a.out.mkdir(parents=True)
    report=dict(status="S554_PREREGISTERED_FIVE_MORE_TRAIN_SPLIT_COMPOSERS_AFTER_S544_NO_WAV_MIDI",
         official_metadata_URL=URL,official_metadata_sha256=digest(b),
         original_previous_S469_3_S537_5_S545_5_composer_components_excluded=True,
         co_composer_COMPONENTS_fully_excluded_and_pairwise_disjoint=True,
         original_previous_thirteen_composer_names=sorted(history_artist),
         five_next_distinct_composers=N,
         eligible_train_split_tracks_from_metadata_after_exclusions=eligible,
         MAESTRO_original_split_used="train",
         NOT_official_MAESTRO_test_split=True,
         NOT_guaranteed_never_used_elsewhere_in_repo=True,
         filenames_frozen_before_audio_model_inference=True,
         source_MIDI_files_not_read=True,source_WAV_files_not_read=True,
         original_reference_S80_ExactK_Live_untouched=True,
         no_promotion=True,source_audio_mirror=AUDIO_MIRROR,
         chosen_metadata_only=selected)
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S554 NEXT 5 INDEPENDENT COMPOSERS METADATA SEALED",
        json.dumps(dict(n=5,eligible=eligible,
          pieces=[dict(composer=x["composer"],audio=x["audio_filename"])
                  for x in selected],
          MIDI_AND_WAV_OPENED=False)),flush=True)

def selftest():
    rows=[dict(split="train",duration="80",audio_filename=f"2018/{k}.wav",
         midi_filename=f"2018/{k}.midi",canonical_composer=k,
         canonical_title="test") for k in "ABCDEFGHIJKLM"]
    old={"A","B","C","D","E","F","G","H"}
    results,_=choose(rows,{f"2018/{k}.wav" for k in old},old)
    need(len(results)==5 and all(x["composer"] not in old for x in results)
         and results==choose(list(reversed(rows)),
                 {f"2018/{k}.wav" for k in old},old)[0],
         "S554 test composer exclusion must be fixed before MIDI")
    extra=components("Franz Schubert / Franz Liszt")
    need("Franz Liszt" in extra and "Franz Schubert" in extra and
         not (components("Johann Sebastian Bach") & extra),
         "S554 compound composer components must be fully blocked")
    print("S554 SELFTEST PASS actual composers and co-composers disjoint, no WAV/MIDI")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    p.add_argument("--old",type=Path);p.add_argument("--new",type=Path)
    p.add_argument("--prior",type=Path)
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(a.old and a.new and a.prior and a.out,"S554 source 8 historic composers and output required")
        run(a)
