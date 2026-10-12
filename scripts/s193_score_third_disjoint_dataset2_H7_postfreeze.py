"""S193 independently validate prior H7 hypothesis on third filename-disjoint dataset2 WAVs after S192 prediction freeze.

Reads original XML only after S192 source WAV predictions+matching protocol frozen.
Evaluates *adapted S35/S179*, not original S80 or freeze_local_combo. No fitting.
"""
from __future__ import annotations
import argparse,hashlib,json,urllib.request,zipfile,xml.etree.ElementTree as ET,collections
from pathlib import Path
import numpy as np
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP

PROHIBITED={'target','y'}
def need(ok,m):
    if not ok:raise ValueError(m)
def sha(b):
    return hashlib.sha256(b).hexdigest()
def load_audio_predictions(base):
    r=json.loads((base/'report.json').read_text())
    policy=r['preregistered_label_matching_and_exactK_cardinality_policy']
    need(r['status'].startswith('S192_FROZEN_PRELABEL_') and
         not r['XML_ground_truth_content_read'] and
         policy['max_onset_difference_seconds']==.080,
         'S192 source-only SHA/label matching policy not frozen')
    a=base/'S192_frozen_original_S35_dataset2_audio_only.npz'
    b=base/'S192_frozen_event_index_no_labels.npz'
    need(sha(a.read_bytes())==r['audio_source_only_sha256'] and
         sha(b.read_bytes())==r['frozen_event_index_sha256'],
         'frozen original predictions or event index modified')
    with np.load(a,allow_pickle=False) as z:
        need(set(z.files)=={'row_id','summary58','flow245','morphology490',
          'physics294','sequence42x49','source_head_ids','source_probabilities',
          'source_predicted_K'},'source predictor has unexpected fields')
        original={k:z[k].copy() for k in ('row_id','source_head_ids','source_predicted_K')}
    with np.load(b,allow_pickle=False) as z:
        need(set(z.files)=={'row_id','source_wav_path','acoustic_onset_sample'},
             'annotation injected into audio index')
        index={k:z[k].copy() for k in z.files}
    with np.load(base/'new_255_experimental'/'S179_new_8vote_255group_audio_only.npz',
                 allow_pickle=False) as z:
        adapted=z['new_parent_K'].copy()
        ids=z['row_id'].copy()
    need(np.array_equal(original['row_id'],index['row_id']) and
         np.array_equal(ids,index['row_id']) and
         len(ids)==r['audio_only_selected_onsets'],
         'frozen original/adapted source events out of order')
    return r,policy,original,index,adapted
def tag_lower(element):
    return element.tag.split('}')[-1].lower().replace('_','')
def parse_original_pitch_events(xml):
    # Standard IDMT XML transcription lists precise MIDI pitch and onset/offset seconds.
    need(len(xml)<4_000_000,'unexpected original XML size')
    root=ET.fromstring(xml)
    events=[]
    for e in root.iter():
        if tag_lower(e)!='event':continue
        field={}
        for child in e.iter():
            if child is e:continue
            t=tag_lower(child)
            if t in ('onsetsec','offsetsec','pitch'):
                field[t]=(child.text or '').strip()
        if not all(k in field for k in ('onsetsec','offsetsec','pitch')):continue
        if field['pitch'].upper() in ('NA','N/A','NONE',''):
            continue
        try:
            onset=float(field['onsetsec']);end=float(field['offsetsec']);pitch=int(float(field['pitch']))
        except (ValueError,OverflowError):continue
        if not (np.isfinite(onset) and np.isfinite(end) and
                0<=onset<end and 0<=pitch<=127):continue
        events.append((onset,end,pitch))
    return sorted(events,key=lambda x:(x[0],x[2]))
def nearest_annotated_k(ann,acoustic_times,tolerance=.080):
    """Frozen rule: one available onset anchor per audio selected onset.

    XML notes grouped into one onset if within 20ms of anchor.
    Label cardinality is DISTINCT pitch sounding 30ms after that anchor.
    Don't count unlabeled acoustic events as K0.
    """
    need(abs(tolerance-.080)<1e-10,'predeclared onset max deviation changed')
    groups=[]
    for onset,end,pitch in ann:
        if not groups or onset-groups[-1][0]>.020:
            groups.append((onset,[]))
        groups[-1][1].append((onset,end,pitch))
    used=set();true=np.full(len(acoustic_times),-1,np.int16)
    differences=np.full(len(acoustic_times),np.nan,np.float32)
    for row,t in enumerate(acoustic_times):
        best=sorted(
          ((abs(g[0]-t),i) for i,g in enumerate(groups)
           if i not in used and abs(g[0]-t)<=tolerance),
          key=lambda x:(x[0],x[1]))
        if not best:continue
        _,j=best[0];used.add(j)
        onset=groups[j][0]
        playing={p for st,en,p in ann if st<=onset+.030 and en>onset+.030}
        if len(playing)==0 or len(playing)>6:
            continue
        true[row]=len(playing)
        differences[row]=abs(t-onset)
    return true,differences,len(groups)
def stats(y,p):
    conf=np.zeros((7,7),np.int32)
    for k,z in zip(y,p):conf[k,z]+=1
    poly=(y>=2)
    return {'total':len(y),'correct':int(np.sum(y==p)),
      'global_ExactK':float(np.mean(y==p)) if len(y) else None,
      'poly_true_rows':int(np.sum(poly)),
      'poly_ExactK':float(np.mean(y[poly]==p[poly])) if np.any(poly) else None,
      'true_K0toK6':np.bincount(y,minlength=7).tolist(),
      'pred_K0toK6':np.bincount(p,minlength=7).tolist(),
      'per_K_correct':np.bincount(y[y==p],minlength=7).tolist(),
      'confusion_true_rows_pred_columns':conf.tolist()}
def run(source_dir,out):
    need(not out.exists(),'immutable independent frozen-audio XML scoring result')
    frozen,policy,orig,index,parent=load_audio_predictions(source_dir)
    need(frozen['dataset2_independent_WAVs']==16,'dataset2 benchmark corpus count changed')
    with urllib.request.urlopen(urllib.request.Request(API,headers={
        'Accept':'application/json','User-Agent':'S80-S193-postfreeze-exact-note-K'}),
        timeout=60) as r:
        meta=audit(json.load(r))
    reader=HttpRangeZIP(meta['source_url'],meta['size_bytes'])
    source=index['source_wav_path'].astype(str)
    starts=index['acoustic_onset_sample'].astype(np.int64)
    labels=np.full(len(starts),-1,np.int16)
    delta=np.full(len(starts),np.nan,np.float32)
    proof=[]
    with zipfile.ZipFile(reader) as z:
        for wav in dict.fromkeys(source):
            location=wav.replace('/dataset2/audio/','/dataset2/annotation/').rsplit('.',1)[0]+'.xml'
            need(location in z.namelist(),'original note-level XML file missing: '+location)
            x=z.read(location)
            ann=parse_original_pitch_events(x)
            pos=np.flatnonzero(source==wav)
            matched,differences,n_groups=nearest_annotated_k(
                ann,starts[pos]/44100.,policy['max_onset_difference_seconds'])
            labels[pos]=matched;delta[pos]=differences
            proof.append({'wav_name_SHA256':sha(wav.encode()),'XML_sha256':sha(x),
               'XML_parsed_note_events':len(ann),'XML_onset_groups':n_groups,
               'audio_selected_onsets':len(pos),
               'XML_ExactK_scored_audio_events':int(np.sum(matched>=0))})
    valid=labels>=0
    supported=int(valid.sum())
    pred=parent[valid]
    y=labels[valid]
    need(np.isin(y,range(7)).all() and np.isin(pred,range(7)).all(),'K0..6 bounds changed')
    original_hist={}
    for j,name in enumerate(orig['source_head_ids'].astype(str)):
        source_pred=orig['source_predicted_K'][valid,j].astype(int)
        original_hist[str(name)]=stats(y,source_pred)
    result={
      'status':'S193_FROZEN_PRELABEL_DATASET2_TRUE_MIDI_PITCH_K_AUDIT_ADAPTED_ONLY',
      'S192_preXML_source_artifact_sha256':frozen['audio_source_only_sha256'],
      'S192_event_index_sha256':frozen['frozen_event_index_sha256'],
      'preregistered_fixed_label_matching_policy':policy,
      'source_wav_count':len(proof),
      'source_acoustic_only_selected_events':len(labels),
      'annotation_matched_events':supported,
      'annotation_matching_coverage':float(np.mean(valid)),
      'annotation_unmatched_events':int((~valid).sum()),
      'matched_onset_diff_ms_max':float(np.max(delta[valid])*1000) if supported else None,
      'annotation_total_note_events':sum(p['XML_parsed_note_events'] for p in proof),
      'dataset2_true_pitch_cardinality_from_XML_not_chordname':True,
      'original_7_expert_individual_comparators':original_hist,
      'S179_new_parent_adapted_selection':stats(y,pred),
      'H7_morphology_preselected_from_exposed_S190_prior_cohort':True,
      'H7_morphology_new_unseen_cohort_exactK':original_hist['H7_attack_morphology'],
      'H7_selection_tuned_on_S192_or_S193_labels':False,
      'source_XML_checksum_proofs':proof,
      'original_freeze_local_combo_run_executed':False,
      'original_16_family_S45_complete':False,
      'original_193_variant_S46_executed':False,
      'S80_vs_official_reference_proven':False,
      'S80_promoted':False}
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in (
        'status','source_wav_count','source_acoustic_only_selected_events',
        'annotation_matched_events','annotation_matching_coverage',
        'annotation_unmatched_events',
        'S179_new_parent_adapted_selection','S80_promoted')},indent=2),flush=True)
def selftest():
    root='<doc><transcription><event><onsetSec>1.000</onsetSec><offsetSec>1.5</offsetSec><pitch>60</pitch></event><event><onsetSec>1.008</onsetSec><offsetSec>1.5</offsetSec><pitch>64</pitch></event><event><onsetSec>2.0</onsetSec><offsetSec>2.2</offsetSec><pitch>67</pitch></event></transcription></doc>'
    events=parse_original_pitch_events(root.encode())
    truth,delta,n=nearest_annotated_k(events,np.array([1.03,2.03,3.0]))
    need(truth.tolist()==[2,1,-1] and n==2,'pitch count/80ms/no label→K0 false interpretation')
    print('S193 PASS XML note parse, simultaneous note grouping, 80ms matching, unmatched exclusion')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source-dir',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(a.source_dir and a.out,'actual frozen S192 source and output required')
        run(a.source_dir,a.out)
