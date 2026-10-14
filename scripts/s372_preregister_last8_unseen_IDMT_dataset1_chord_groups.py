"""S372: prospective last 8 additional unexposed IDMT Dataset1 WAVs, NAMES ONLY.

Original S276 pre-exposed 12 chords and S357 another 24 excluded by full filename AND canonical
family stem. Explicit filename-based candidate selection before audio or XML.
Cannot claim song disjointness on filename identity alone.
"""
from __future__ import annotations
import argparse,hashlib,json,re,urllib.request,zipfile
from collections import defaultdict
from pathlib import Path,PurePosixPath
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s276_idmt_prelabel_song_cohort_inventory import ismember,match_annotation,orderhash,sha
N=32
def need(v,msg):
    if not v:raise ValueError(msg)
def group(name):
    stem=PurePosixPath(name).stem.lower()
    # Same canonicalizer used by original 12 prior preregistration.
    stem=re.sub(r"^(ar|fs|lp|ib|gt)_","",stem)
    return stem
def run(s276,s357,out):
    need(not out.exists(),"prospective cohort cannot be overwritten")
    prior=json.loads(s276.read_text())
    need(prior['status']=='S276_IDMT_CANDIDATE_COHORT_CENTRAL_DIRECTORY_ONLY_NOT_SCORED'
         and not prior['WAV_file_contents_read'] and not prior['XML_MIDI_annotation_contents_read'],
         "S276 original central-directory-only source unavailable")
    already=json.loads(s357.read_text())
    need(already['status']=='S357_IDMT_DATASET1_24_NEW_CHORD_FAMILY_FILENAME_ONLY_PRELABEL' and
         already['central_directory_sha256']==prior['zip_central_names_sha256'] and
         len(already['next_twentfour'])==24 and
         not already['all_XML_annotation_contents_read'],
         'original 24 unseen S357 SHA-committed names-only manifest required')
    prev=prior['candidate_groups']['dataset1_filename_prioritized_second']
    need(len(prev)==12,"historical already-opened Dataset1 twelve chord samples required")
    previous_families={group(p['wav']) for p in prev}
    with urllib.request.urlopen(urllib.request.Request(API,headers={
       'Accept':'application/json',
       'User-Agent':'S372-twenty-four-prospective-names-only-no-WAV-XML'}),timeout=60) as response:
        publisher=audit(json.load(response))
    reader=HttpRangeZIP(publisher['source_url'],publisher['size_bytes'])
    with zipfile.ZipFile(reader,'r') as z:
        names=z.namelist()  # ZIP central directory only. NO z.read.
    need(sha("\n".join(sorted(names)).encode())==prior['zip_central_names_sha256'],
         "source ZIP listings drifted after S276")
    wav=sorted((name for name in names if ismember(name,'dataset1','.wav')),key=orderhash)
    xml=set(name for name in names if ismember(name,'dataset1','.xml'))
    need(len(wav)==400,"S276 400 published WAVs changed")
    group_candidates=defaultdict(list)
    for name in wav:
        if group(name) in previous_families:
            continue
        matching,how=match_annotation(name,xml)
        if matching is None:
            continue
        # Keep EXACT original intended filename category selection.
        if not re.search(r'lick|chord|poly|major|minor|_maj|_min',
                         PurePosixPath(name).stem,re.I):
            continue
        group_candidates[group(name)].append(dict(
            wav=name,wav_filename_sha256=orderhash(name),
            annotation_path_name_only=matching,
            pairing_source=how,composition_family_from_filename_only=group(name)))
    heads=[min(items,key=lambda item:item['wav_filename_sha256'])
           for items in group_candidates.values()]
    heads.sort(key=lambda item:item['wav_filename_sha256'])
    need(len(heads)>=N,
         f"only {len(heads)} distinct filename-only chord families; preregister new fixed N")
    need([p['wav_filename_sha256'] for p in heads[:24]]==
         [p['wav_filename_sha256'] for p in already['next_twentfour']],
         '24 historically preselected SHA ordered source files must not shift')
    chosen=heads[24:32]
    need(len(chosen)==8 and all(group(x['wav']) not in previous_families for x in chosen)
         and not {group(x['wav']) for x in chosen}.intersection({group(x['wav']) for x in already['next_twentfour']})
         and len({x['composition_family_from_filename_only'] for x in chosen})==8
         and all(x['wav'] not in {p['wav'] for p in prev} for x in chosen),
         "prospective source collides with 12+24 exposed core groups or is not eight")
    result=dict(status='S372_IDMT_DATASET1_FINAL_EIGHT_CHORD_FAMILIES_FILENAME_ONLY_PRELABEL',
      source_original_S276_manifest_SHA256=sha(s276.read_bytes()),
      central_directory_sha256=prior['zip_central_names_sha256'],
      total_published_dataset1_WAVs=len(wav),
      eligible_priority_new_distinct_filename_families=len(heads),
      original_S276_known_exposed_dataset1_WAV_count=12,
      original_S357_already_exposed_dataset1_WAV_count=24,
      previous_12_filename_family_hard_excluded=True,
      previous_24_filename_family_hard_excluded=True,
      new_unseen_WAV_count=8,
      exact_selection_rule='SHA ordered heads indexes 24..31 after checking exact first 24 equals original SHA committed S357, while original 12 exposed cores excluded',
      final_eight=chosen,
      source_S357_prior24_manifest_SHA256=sha(s357.read_bytes()),
      all_WAV_contents_read=False,all_XML_annotation_contents_read=False,
      independent_composition_identity_proven=False,
      all_negative_K0_transcription_complete_proven=False,
      original_official_freeze_predicted=False,
      full_S80_predicted=False,
      official_promotion=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='final_eight'},indent=2),flush=True)
    print('ALL 8 PRECOMMITTED XML NAMES',json.dumps([x['annotation_path_name_only'] for x in chosen]),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--s276',type=Path,required=True)
    p.add_argument('--s357',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.s276,args.s357,args.out)
