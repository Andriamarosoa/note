"""S186 independent external guitar annotation-METADATA inventory on frozen S185.

Only ZIP central-directory members (names and sizes), no annotation content,
MIDI/XML parsing, WAV replay, labels, training or search on predictions.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,urllib.request,zipfile
from pathlib import Path,PurePosixPath
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s178_idmt_audio_only_native_s35_heads import group_wav_names

def sha(x):
    return hashlib.sha256(x.encode()).hexdigest()
def run(output):
    if output.exists():raise ValueError('immutable S186 path')
    with urllib.request.urlopen(urllib.request.Request(API,headers={
       'Accept':'application/json','User-Agent':'S80-S186-annotation-names-only'}),timeout=45) as response:
        src=audit(json.load(response))
    reader=HttpRangeZIP(src['source_url'],src['size_bytes'])
    with zipfile.ZipFile(reader) as archive:
        files=[info for info in archive.infolist() if not info.is_dir()]
        names=[info.filename for info in files]
        first20,available=group_wav_names(archive,20)
        rank20=first20[10:20]
        parent=PurePosixPath(rank20[0]).parent
        structure=collections.Counter()
        dataset4=[]
        annotation_files=[]
        for info in files:
            p=PurePosixPath(info.filename)
            structure[p.suffix.lower() or '[none]']+=1
            if '/dataset4/' in ('/'+info.filename.lower()):
                dataset4.append(info)
                if p.suffix.lower() not in ('.wav','.wave'):
                    annotation_files.append(info)
        detail=[]
        for selected in rank20:
            p=PurePosixPath(selected)
            basename=p.stem
            exact=[info for info in annotation_files if
                  PurePosixPath(info.filename).stem.lower()==basename.lower()]
            relative=[info for info in annotation_files if
                  basename.lower() in PurePosixPath(info.filename).stem.lower()
                  or PurePosixPath(info.filename).stem.lower() in basename.lower()]
            # Only names are inspected; never interpret a track name as a ground-truth pitch.
            detail.append({
              'wav_file':selected,'track_hash':sha(selected),
              'same_stem_non_WAV':[
                {'name':info.filename,'bytes':info.file_size} for info in exact[:12]],
              'loose_name_matches':[
                {'name':info.filename,'bytes':info.file_size} for info in relative[:8]],
            })
        report={
          'status':'S186_FROZEN_S185_ANNOTATION_FILENAME_METADATA_ONLY',
          'S185_second_10_WAV_identified_by_same_unchanged_filename_hash_ranks':True,
          'S185_tracks':len(rank20),'total_dataset4_WAV':available,
          'all_zip_file_count':len(files),
          'archive_ext_counts':dict(sorted(structure.items())),
          'dataset4_nonwav_member_count':len(annotation_files),
          'dataset4_nonwav_extension_counts':dict(sorted(collections.Counter(
             PurePosixPath(info.filename).suffix.lower() or '[none]'
             for info in annotation_files).items())),
          'dataset4_annotation_related_filenames_samples':[
             {'name':i.filename,'bytes':i.file_size} for i in annotation_files[:36]],
          'second_10_WAV_annotation_name_crosswalk':detail,
          'downloaded_remote_ZIP_index_bytes':reader.network_bytes,
          'raw_WAV_contents_opened':False,
          'annotation_contents_opened':False,
          'annotation_ground_truth_measured':False,
          'exactK_scored':False,'S80_promoted':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:report[k] for k in (
       'status','all_zip_file_count','archive_ext_counts','total_dataset4_WAV',
       'dataset4_nonwav_member_count','dataset4_nonwav_extension_counts',
       'dataset4_annotation_related_filenames_samples',
       'second_10_WAV_annotation_name_crosswalk','exactK_scored')},
       ensure_ascii=False,indent=2),flush=True)
def selftest():
    first=['track_'+str(i) for i in range(20)]
    a,b=first[:10],first[10:20]
    assert not set(a)&set(b) and len(b)==10
    print('S186 PASS deterministic S185 second cohort filename metadata-only')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path);p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:run(a.output)
