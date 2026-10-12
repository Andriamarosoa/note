"""S187 inspect original CSV annotation schema ONLY AFTER frozen S185 WAV predictions.

The frozen audio predictions already exist (S185 run 38209386300).
No training, threshold changes, guessed chord cardinality, or label-derived K.
Reads only same-instrument, same-tempo matched CSV annotation members.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,urllib.request,zipfile,collections
from pathlib import PurePosixPath,Path
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s178_idmt_audio_only_native_s35_heads import group_wav_names

def digest(v):
    return hashlib.sha256(v).hexdigest()
def verify(c,m):
    if not c:raise ValueError(m)
def csv_rows(data):
    try:text=data.decode('utf-8-sig')
    except UnicodeDecodeError:text=data.decode('latin1')
    if len(text)>30000:raise ValueError('oversized annotation text')
    # Original format may be comma, semicolon or tab; keep original lines in evidence.
    lines=[s.strip() for s in text.splitlines() if s.strip()]
    if not lines:return {'first_lines':[],'separator':'none','field_counts':[]}
    sep=max((',', ';', '\t'),key=lambda t:sum(line.count(t) for line in lines[:5]))
    head=list(csv.reader(lines[:5],delimiter=sep))
    return {'first_lines':[line[:180] for line in lines[:5]],
            'separator':repr(sep),'field_counts':[len(row) for row in head],
            'rows_read_for_schema':len(head),'total_nonempty_lines':len(lines)}
def run(out):
    verify(not out.exists(),'immutable annotation schema after frozen predictions')
    with urllib.request.urlopen(urllib.request.Request(
      API,headers={'Accept':'application/json','User-Agent':'S80-S187-postfreeze-schema'}),timeout=60) as r:
        meta=audit(json.load(r))
    reader=HttpRangeZIP(meta['source_url'],meta['size_bytes'])
    reports=[];unmatched=[]
    with zipfile.ZipFile(reader) as z:
        names,available=group_wav_names(z,20)
        held=names[10:20]
        allnames=set(z.namelist())
        verify(len(held)==10,'S185 ten same deterministic frozen tracks changed')
        for wav in held:
            w=PurePosixPath(wav)
            base=wav.replace('/audio/','/annotation/chords/').rsplit('.',1)[0]+'.csv'
            if base not in allnames:
                unmatched.append({'track_name_SHA256':digest(wav.encode()),
                                  'reason':'no same-instrument/tempo chord CSV','track':wav})
                continue
            info=z.getinfo(base)
            verify(0<info.file_size<100000,'unexpected annotation size')
            payload=z.read(base)
            res=csv_rows(payload)
            reports.append({'track':wav,'annotated_CSV':base,
              'csv_member_SHA256':digest(payload),'csv_bytes':len(payload),
              'schema_only':res})
            print(json.dumps({'wav_sha256':digest(wav.encode()),
                              'schema_only':res},ensure_ascii=False),flush=True)
    result={'status':'S187_IDMT_S185_POSTFREEZE_CHORD_CSV_FORMAT_ONLY',
      'frozen_before_reading_truth':True,
      'frozen_S185_GitHub_run_id':38209386300,
      'frozen_10_disjoint_WAV_count':len(held),
      'exact_matched_same_instrument_chord_CSV_files':len(reports),
      'unmatched_track_count':len(unmatched),
      'annotated_csv_schema':reports,
      'unmatched':unmatched,
      'network_ZIP_range_bytes':reader.network_bytes,
      'annotation_opened_after_original_source_predictions_frozen':True,
      'predictions_or_model_weights_read_during_schema_audit':False,
      'new_annotations_converted_into_exact_pitch_count':False,
      'ExactK_accuracy_measured':False,
      'S80_official_reference_compared':False,
      'S80_promoted':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in (
      'status','frozen_10_disjoint_WAV_count',
      'exact_matched_same_instrument_chord_CSV_files',
      'unmatched_track_count','unmatched',
      'new_annotations_converted_into_exact_pitch_count','ExactK_accuracy_measured',
      'S80_promoted')},indent=2,ensure_ascii=False),flush=True)
def selftest():
    a='abc/dataset4/acoustic_pickup/fast/reggae_ska/audio/ska_2_180BPM.wav'
    b=a.replace('/audio/','/annotation/chords/').rsplit('.',1)[0]+'.csv'
    verify(b.endswith('/annotation/chords/ska_2_180BPM.csv'),'exact path mapping')
    q=csv_rows(b'0.10,0.14,CM\n0.14,0.5,G7\n')
    verify(q['field_counts']==[3,3],'CSV delimiter mapping')
    print('S187 PASS exact same-instrument matching and no guessed chord cardinality')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        verify(a.output is not None,'output required')
        run(a.output)
