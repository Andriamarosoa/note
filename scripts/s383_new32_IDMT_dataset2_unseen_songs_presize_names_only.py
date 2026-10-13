"""S383 names-only 32 NEW IDMT Dataset2 XML-paired WAVs for prospective K3.

Original S276 exposed first 48 filename-SHA tracks, then first 16
reserve rank49-64 now XML-opened in S327. This stage excludes all
64 by immutable ORIGINAL S276 central SHA and selects next 32
SHA-ranked DISTINCT exact filename stems from remaining pool with
unambiguous XML name pair. No audio/XML bytes or model predictions;
cannot prove composition-disjointness merely from stems.
Do NOT tune trained S376 K3 head after seeing these future WAV labels.
"""
from __future__ import annotations
import argparse,json,urllib.request,zipfile,hashlib
from pathlib import Path,PurePosixPath
from collections import defaultdict
from scripts.s276_idmt_prelabel_song_cohort_inventory import ismember,orderhash,match_annotation,sha
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
def need(x,msg):
    if not x:raise ValueError(msg)
def run(prior,out):
    need(not out.exists(),'S383 no overwrite original future test inventory')
    evidence=json.loads(prior.read_text())
    need(evidence['status']=='S276_IDMT_CANDIDATE_COHORT_CENTRAL_DIRECTORY_ONLY_NOT_SCORED'
         and not evidence['WAV_file_contents_read'] and not evidence['XML_MIDI_annotation_contents_read']
         and evidence['counts_WAV']['dataset2']==261,
         'source original S276 published ZIP central directory SHA required')
    old_sha=set(evidence['dataset2_previous_48_wav_names_sha256'])
    old16=evidence['candidate_groups']['dataset2_unexposed_XML_paired_reserve']
    need(len(old_sha)==48 and len(old16)==16,'S276 known 48+16 prior XML exposures required')
    old16_sha={z['wav_filename_sha256'] for z in old16}
    need(len(old16_sha)==16 and not old_sha.intersection(old16_sha),
         'S276 candidate rank49-64 already used')
    with urllib.request.urlopen(urllib.request.Request(API,headers={
         'Accept':'application/json','User-Agent':'S383-CENTRAL-NAMES-ONLY-PRESCORE-32-UNUSED'}),
         timeout=60) as response:
        pinned=audit(json.load(response))
    reader=HttpRangeZIP(pinned['source_url'],pinned['size_bytes'])
    with zipfile.ZipFile(reader,'r') as z:
        names=z.namelist()  # never z.read()
    need(sha('\n'.join(sorted(names)).encode())==evidence['zip_central_names_sha256'],
         'official ZIP central directory SHA drift')
    wav=sorted([name for name in names if ismember(name,'dataset2','.wav')],key=orderhash)
    xml=set(name for name in names if ismember(name,'dataset2','.xml'))
    need(len(wav)==261 and {orderhash(w) for w in wav[:48]}==old_sha,
         'original first 48 already exposed WAV order SHA changed')
    # Other 16 are exactly the first paired WAVs after rank48, not
    # necessarily positions 48:64 if some lacked matching XML.
    candidates=[]
    seen=set(PurePosixPath(z['wav']).stem.lower() for z in old16)
    for name in wav[48:]:
        ident=orderhash(name)
        if ident in old_sha or ident in old16_sha:
            continue
        pair,how=match_annotation(name,xml)
        if pair is None:continue
        core=PurePosixPath(name).stem.lower()
        if core in seen:continue
        seen.add(core)
        candidates.append(dict(wav=name,wav_filename_sha256=ident,
          annotation_path_name_only=pair,
          pairing_source=how,filename_composition_group_only=core))
    need(len(candidates)>=32,'less than 32 unused unique XML-paired Dataset2 stem groups remain')
    chosen=candidates[:32]
    need(len({e['wav_filename_sha256'] for e in chosen})==32
         and not old_sha.intersection(e['wav_filename_sha256'] for e in chosen)
         and not old16_sha.intersection(e['wav_filename_sha256'] for e in chosen),
         'S383 leaks original exposed 64 Dataset2 WAV identities')
    report=dict(status='S383_NEW32_PROSPECTIVE_DATASET2_NAMES_ONLY_SOURCE_UNOPENED',
      prior_original_S276_manifest_sha256=sha(prior.read_bytes()),
      original_ZIP_central_names_sha256=evidence['zip_central_names_sha256'],
      published_dataset2_WAV_count=len(wav),previous_48_exposed_explicit_SHA_excluded=True,
      previous_16_exposed_original_S302_S327_SHA_excluded=True,
      previously_exposed_dataset2_WAV_count=64,
      new_sha_sorted_distinct_stem_candidates=32,
      untouched_new_32=chosen,
      exact_rule='original rank49..261 sorted SHA, remove first 48 and previously scored 16, require unique paired XML by central ZIP member name, choose next 32 distinct exact stems in SHA order',
      new_WAV_audio_read=False,new_XML_contents_read=False,
      already_trained_GuitarSet_S376_not_refit_on_this_source=True,
      true_composition_disjointness_unverified=True,
      negative_K0_exhaustiveness_unverified=True,
      official_promotion=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='untouched_new_32'},indent=2))
    print('S383 prospective 32 SHA WAVE names',json.dumps([e['wav_filename_sha256'] for e in chosen]))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('s276','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.s276,a.out)
