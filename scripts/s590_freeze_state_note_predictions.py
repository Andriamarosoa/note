"""S590: three fixed 28-source policies, frozen on new WAVs before target MIDI."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.s585_attack_sustain_restrike import need,sha,save,temporal52,predictions,POLICIES
from scripts.s577_preregister_next_composer_cohort_after28_audio_unopened import components

def run(root,out):
    import joblib
    need(not out.exists(),'S590 immutable pre-MIDI output')
    manifest=json.loads((root/'S586/report.json').read_text())
    wav=json.loads((root/'S587/report.json').read_text())
    bp=json.loads((root/'S588/report.json').read_text())
    feat=json.loads((root/'S589/report.json').read_text())
    old=json.loads((root/'S578/report.json').read_text())
    new=json.loads((root/'S585/report.json').read_text())
    path=root/'S589/S589_ALL_V88_K0_K6_ORIGINAL_WAV_49_PITCH_FEATURES_PREMIDI.jsonl'
    need(manifest['previous_source_count']==33 and manifest['source_MIDI_files_not_read']
        and manifest['source_WAV_files_not_read'] and manifest['new_distinct_composers']==2
        and wav['preregistered_S586_SHA']==sha(root/'S586/report.json')
        and wav['prelabel_predictions_frozen'] and not wav['paired_MIDI_annotation_files_OPENED']
        and not bp['original_MAESTRO_MIDI_ANNOTATIONS_OPENED']
        and feat['frozen_feature_ledger_SHA256']==sha(path) and feat['no_original_true_MIDI_annotation_read']
        and new['status']=='S585_28_SOURCE_FOUR_STATE_NOTE_HEAD_TRAINED'
        and new['policies']==list(POLICIES) and new['original_S578_OOF_counts_reproduced'],
        'S590 provenance must hold before any target MIDI access')
    records=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    rec=np.array([r['source_recording_sha'] for r in records],str)
    chosen={x['filename_SHA256']:x for x in manifest['chosen_metadata_only']}
    need(set(rec)==set(chosen) and len(records)==feat['original_V88_onset_candidates'],'S590 full source coverage')
    oldspec=[x for x in old['weights_before_S577_exposure'] if x['architecture']=='general28']
    need(len(oldspec)==1 and oldspec[0]['source_count']==28,'S590 original full28 control')
    op=root/'S578'/oldspec[0]['filename'];npth=root/'S585'/new['model_filename']
    need(sha(op)==oldspec[0]['sha'] and sha(npth)==new['model_sha256'],'S590 original trained weights immutable')
    need(set(oldspec[0]['source_composers'])==set(new['training_sources']) and not (set(rec)&set(new['training_sources'])),
        'S590 same 28 train sources, no target source leakage')
    trainartists=set().union(*(components(c) for c in new['training_composers'].values()))
    need(all(not (components(v['composer'])&trainartists) for v in chosen.values()),'S590 composer leakage')
    X=temporal52(records)
    control=joblib.load(op);state=joblib.load(npth)
    gp=control.predict_proba(X.reshape(-1,52))[:,list(control.classes_).index(1)].reshape(len(records),49)
    need(list(state.classes_)==[0,1,2,3],'S590 four-state class ABI')
    sp=state.predict_proba(X.reshape(-1,52)).reshape(len(records),49,4)
    pred,counts=predictions(records,rec,gp,sp)
    out.mkdir(parents=True)
    save(out/'predictions.json',pred)
    np.savez_compressed(out/'probabilities.npz',global28=gp,state4=sp,source=rec)
    report=dict(status='S590_THREE_POLICIES_SEALED_BEFORE_SIXTH_MIDI',policies=list(POLICIES),
        MIDI_NOT_OPENED=True,source_count=2,notes_sha256=sha(out/'predictions.json'),
        probabilities_sha256=sha(out/'probabilities.npz'),old_model_sha256=sha(op),new_model_sha256=sha(npth),
        input_hashes={f'S{s}':sha(root/f'S{s}/report.json') for s in (585,586,587,588,589)},
        feature_sha256=sha(path),counts=counts,
        note_counts={p:{s:sum(len(x['notes']) for x in g) for s,g in sources.items()} for p,sources in pred.items()},
        full_S80_ExactK_NOT_REPLAYED=True,live_NOT_modified=True,no_promotion=True)
    save(out/'report.json',report)
    print('S590 PREMIDI NOTES SEALED',report['notes_sha256'],json.dumps(report['note_counts']),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.root,a.out)
