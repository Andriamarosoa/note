"""S276: IDMT S275 prospective heldout CANDIDATE inventory, filenames ONLY.

Read Zenodo pinned metadata and ZIP central-directory member NAMES ONLY.
No WAV byte content, XML/MIDI content, event truth, model fit, or Exact-K.
This is a cohort-selection and file-overlap audit, NOT song disjoint proof.
"""
from __future__ import annotations
import argparse, hashlib, json, urllib.request, zipfile, re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from scripts.s177_preregister_external_polyphony_source import API, audit as pinned
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP

def req(cond, why):
    if not cond: raise ValueError(why)
def sha(b):
    return hashlib.sha256(b).hexdigest()
def basic_set():
    files=[
       "IDMT-SMT-GUITAR_V2/dataset3/audio/songA.wav",
       "IDMT-SMT-GUITAR_V2/dataset3/audio/songB.wav",
       "IDMT-SMT-GUITAR_V2/dataset3/annotation/songA.xml",
       "IDMT-SMT-GUITAR_V2/dataset3/annotation/songB.xml",
       "IDMT-SMT-GUITAR_V2/dataset1/audio/AR_Lick1_FN.wav",
       "IDMT-SMT-GUITAR_V2/dataset1/annotation/AR_Lick1_FN.xml",
    ] + [f"IDMT-SMT-GUITAR_V2/dataset2/audio/clip{i}.wav" for i in range(261)]
    return files
def ismember(n, dataset, extension):
    p=PurePosixPath(n)
    return (dataset.lower() in (seg.lower() for seg in p.parts)
            and p.suffix.lower()==extension
            and "__MACOSX" not in n
            and not p.name.startswith("._"))
def orderhash(n):
    return sha(n.encode("utf-8"))
def match_annotation(wav, xmllist):
    p=PurePosixPath(wav)
    names=[x for x in xmllist if PurePosixPath(x).stem.lower()==p.stem.lower()]
    # Prior IDMT S254 established audio/ -> annotation/ identity in dataset2.
    expected=wav.replace("/audio/","/annotation/").rsplit(".",1)[0]+".xml"
    if expected in xmllist:return expected, "exact_expected_subdirectory"
    if len(names)==1:return names[0],"unique_matching_stem"
    return None,"missing_or_ambiguous"
def cohort(names):
    wav={d:sorted([n for n in names if ismember(n,d,".wav")],key=orderhash)
         for d in ("dataset1","dataset2","dataset3")}
    xml={d:set(n for n in names if ismember(n,d,".xml"))
         for d in wav}
    req(len(wav["dataset2"])==261,
        "S188 immutable dataset2 must contain 261 WAV; index changed")
    req(len(wav["dataset3"])==5,
        "publisher specified 5 dataset3 WAV; central directory differs")
    req(len(wav["dataset1"])>0,"dataset1 WAV not present")
    # Cohorts 1-48 of dataset2 are ALREADY EXPOSED; NEVER reuse as prospective.
    exposed=wav["dataset2"][:48]
    remaining=wav["dataset2"][48:]
    first=wav["dataset3"]  # all five pieces, candidate, no labels touched
    licks=[p for p in wav["dataset1"] if re.search("lick",PurePosixPath(p).stem,re.I)]
    # One candidate per canonical lick stem, avoiding same-named repeats.
    grouped=defaultdict(list)
    for p in licks:
        core=PurePosixPath(p).stem.lower()
        core=re.sub(r"^(ar|fs|lp|ib|gt)_","",core)
        grouped[core].append(p)
    first_lucks=[min(v,key=orderhash) for k,v in sorted(grouped.items())]
    first_lucks=sorted(first_lucks,key=orderhash)[:12]
    following=remaining[:16]
    chosen={"dataset3_first":first,
            "dataset1_licks_second":first_lucks,
            "dataset2_ranks49to64_exploratory":following}
    crosswalk={}
    all_wav=[w for p in chosen.values() for w in p]
    req(len(set(all_wav))==len(all_wav),"duplicate WAV path across cohorts")
    for key,entries in chosen.items():
        ds=key.split("_",1)[0]
        for w in entries:
            xmlpath,status=match_annotation(w,xml[ds])
            crosswalk[w]=dict(wav=w,wav_filename_sha256=orderhash(w),
                annotation_path_name_only=xmlpath,annotation_pair_status=status)
    stemtofiles=defaultdict(list)
    for f in sum((wav[d] for d in wav),[]):
        stemtofiles[PurePosixPath(f).stem.lower()].append(f)
    same_stem=[dict(stem=k,wavs=v) for k,v in stemtofiles.items()
               if len(v)>1 and any(q in v for q in all_wav)]
    return wav,xml,exposed,remaining,chosen,crosswalk,same_stem

def run(out):
    req(not out.exists(),"immutable metadata-only S276 report")
    with urllib.request.urlopen(urllib.request.Request(
        API,headers={"User-Agent":"S276-unseen-tracks-central-directory-only",
                     "Accept":"application/json"}),timeout=60) as r:
        source=pinned(json.load(r))
    reader=HttpRangeZIP(source["source_url"],source["size_bytes"])
    with zipfile.ZipFile(reader,"r") as z:
        names=z.namelist()  # no z.read, no WAV/XML/MIDI
        inventory_wav,inventory_xml,exposed,remaining,chosen,crosswalk,same_stem=cohort(names)
    pairs_missing=[w for w,a in crosswalk.items() if not a["annotation_path_name_only"]]
    report=dict(
        status="S276_IDMT_CANDIDATE_COHORT_CENTRAL_DIRECTORY_ONLY_NOT_SCORED",
        source_record_id=source["record_id"],
        archive_checksum_published_MD5=source["expected_md5"],
        archive_file_count=len(names),
        zip_central_names_sha256=sha("\n".join(sorted(names)).encode()),
        counts_WAV={k:len(v) for k,v in inventory_wav.items()},
        counts_XML={k:len(v) for k,v in inventory_xml.items()},
        dataset2_previous_48_wav_names_sha256=[orderhash(n) for n in exposed],
        dataset2_unexposed_rank49to261_count=len(remaining),
        candidate_groups={k:[crosswalk[w] for w in v] for k,v in chosen.items()},
        selected_total=len(crosswalk),
        selected_XML_name_pairs_missing=pairs_missing,
        same_stem_cross_dataset_candidates=same_stem,
        all_named_WAV_paths_unique=True,
        song_composition_disjoint_verified=False,
        WAV_file_contents_read=False, XML_MIDI_annotation_contents_read=False,
        train_model_changed=False,
        predicted_event_K_frozen=False,
        true_ExactK_scored=False,
        promotion_permitted=False,
        notes=[
          "All dataset3 five candidate files are included; musical-composition identity not certified.",
          "Dataset1 candidate licks selected label-blind from unique stems; avoid extrapolating K0.",
          "Dataset2 rank49to64 is reserve only, already opened rank1to48 banned for independent reuse.",
          "XML existence by names does not prove exhaustive ground-truth K0 coverage.",
          "Next: verify composition disjointness from training and previously exposed cohorts, then freeze native events and both models before labels."
        ],
        range_requests=reader.range_requests,
        range_bytes_read=reader.network_bytes,
    )
    out.mkdir(parents=True)
    path=out/"S276_IDMT_filename_only_cohort_manifest.json"
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({k:report[k] for k in
      ("status","counts_WAV","counts_XML",
       "dataset2_unexposed_rank49to261_count","selected_total",
       "selected_XML_name_pairs_missing",
       "same_stem_cross_dataset_candidates",
       "song_composition_disjoint_verified","WAV_file_contents_read",
       "XML_MIDI_annotation_contents_read","true_ExactK_scored")},
       indent=2,ensure_ascii=False),flush=True)
    print("DATASET3 WAV NAMES",json.dumps([x["wav"] for x in report["candidate_groups"]["dataset3_first"]]),flush=True)
if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--out",type=Path,required=True)
    p=a.parse_args()
    run(p.out)
