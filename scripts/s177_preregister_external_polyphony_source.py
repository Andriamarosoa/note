"""S177: preregister an unexposed guitar-polyphony AUDIO acquisition source.

Uses public Zenodo JSON metadata only. Never downloads or opens annotated XML,
MIDI, audio, or held-out labels. Source file MD5 and size pinned before data
acquisition. Nothing from future labels can be used to tune the model.
"""
from __future__ import annotations
import argparse,json,urllib.request
from pathlib import Path

RECORD=7544110
API=f"https://zenodo.org/api/records/{RECORD}"
FILENAME="IDMT-SMT-GUITAR_V2.zip"
ORIGINAL_MD5="06796e08731bccffaed6ae59361486e4"
def need(ok,msg):
    if not ok:raise ValueError(msg)
def audit(j):
    need(str(j.get("id"))==str(RECORD),"wrong Zenodo dataset record")
    filelist=j.get("files",[])
    found=[a for a in filelist if a.get("key")==FILENAME]
    need(len(found)==1,"original 1.3GB IDMT SMT guitar zip missing")
    data=found[0]
    actual=str(data.get("checksum",""))
    need(actual=="md5:"+ORIGINAL_MD5,
         "IDMT original recording bytes changed from published source checksum")
    size=int(data.get("size",0))
    need(1_000_000_000<size<2_000_000_000,"unexpected archive size")
    links=data.get("links",{})
    download=links.get("self") or links.get("content")
    need(isinstance(download,str) and download.startswith("https://zenodo.org/"),
         "untrusted archive download destination")
    return {
       "status":"S177_PREREGISTERED_EXTERNAL_IDMT_GUITAR_AUDIO_ONLY_SOURCE",
       "record_id":RECORD,
       "file_name":FILENAME,
       "size_bytes":size,
       "expected_md5":"md5:"+ORIGINAL_MD5,
       "source_url":download,
       "benchmark_identity":"IDMT-SMT-GUITAR V2 (unseen external, prior 19 source piece fold IDs distinct)",
       "proposed_evaluation":"Song-disjoint novel piece split; K0-K6 global, K2plus poly; paired corrections/regressions",
       "heldout_annotation_policy":"No MIDI/XML or labels read before model architecture, weights, thresholds and predictions are frozen with SHA256",
       "metadata_only":True,
       "archive_downloaded":False,
       "audio_access_tested":False,
       "labels_read":False,
       "original_model_frozen_for_this_dataset":False,
       "S80_new_audio_predicted":False,
       "S80_independent_ExactK_measured":False,
       "S80_promoted":False}
def main():
    a=argparse.ArgumentParser()
    a.add_argument("--self-test",action="store_true")
    a.add_argument("--out",type=Path)
    args=a.parse_args()
    if args.self_test:
        sample={"id":RECORD,"files":[{"key":FILENAME,
                "size":1_250_000_000,"checksum":"md5:"+ORIGINAL_MD5,
                "links":{"self":"https://zenodo.org/api/records/7544110/files/IDMT-SMT-GUITAR_V2.zip/content"}}]}
        check=audit(sample)
        need(check["metadata_only"] and not check["S80_promoted"],"truth-free source manifest")
        print("S177 PASS original published IDMT source MD5/size and label-blind validation gate")
        return
    need(args.out and not args.out.exists(),"immutable output path required")
    request=urllib.request.Request(API,headers={"User-Agent":"S80-research-data-metadata/1.0",
                                                  "Accept":"application/json"})
    with urllib.request.urlopen(request,timeout=60) as response:
        need(response.status==200,"Zenodo metadata not accessible")
        j=json.load(response)
    report=audit(j)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
