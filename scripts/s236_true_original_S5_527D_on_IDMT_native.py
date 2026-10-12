"""S236 restore original S5 527D input features on EXACT official IDMT V88 events.

Inputs S234 truly native reference-aligned acoustic waveform features,
S235 original V88 cluster start/end and certified unchanged S233 official
K predictions. Reconstruct true PR16 ownership geometry46 via recovered
original S82 mask coordinates and original S4-owned 527D constructor
(proven exact historically on all 59309 events by S156). Output is
original trained S5 INPUT ONLY; original S5 predictions and full S80
not yet run. No XML/MIDI labels, no tuning, no fold oracle.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s226_original_S71_S72_frozen_on_GOAT import geometry46
from scripts.v273_s4_owned527 import build_owned527
BAD={'true_K','truth','label','labels','midi','target','correct','y','accuracy','regression'}
def req(condition,message):
    if not condition:raise ValueError(message)
def load(p,fields):
    with np.load(p,allow_pickle=False) as z:
        req(set(z.files)==set(fields) and not BAD.intersection(z.files),
            'strict source contract, forbidden labels in '+str(p))
        return {x:z[x].copy() for x in z.files}
def sha(data):return hashlib.sha256(data).hexdigest()
def run(acoustic_file,endpoints_file,out):
    req(not out.exists(),'do not overwrite original S5 527D new audio features')
    acoustic=load(acoustic_file,{'row_id','member','start_samples',
        'incumbent_freeze_K','summary58','sequence42x49','flow245',
        'morphology490','physics294'})
    endpoint=load(endpoints_file,{'row_id','member','start_samples',
        'end_samples','incumbent_K','incumbent_probabilities',
        'nearest_S232_V84_onset_distance_samples'})
    n=len(acoustic['row_id'])
    req(n==1034 and
        np.array_equal(acoustic['row_id'],endpoint['row_id']) and
        np.array_equal(acoustic['member'],endpoint['member']) and
        np.array_equal(acoustic['start_samples'],endpoint['start_samples']) and
        np.array_equal(acoustic['incumbent_freeze_K'],endpoint['incumbent_K']),
        'S234/S235 original official native event ID or K changed')
    req(endpoint['end_samples'].shape==(n,) and
        np.all(endpoint['end_samples']>=endpoint['start_samples']) and
        np.all(endpoint['end_samples']-endpoint['start_samples']<=1764),
        'invalid real V88 group interval exceeds 40ms')
    original46=geometry46({
       'member':endpoint['member'],
       'start_samples':endpoint['start_samples'],
       'end_samples':endpoint['end_samples']})
    req(original46.shape==(n,46) and
        np.isfinite(original46).all() and
        np.isin(original46[:,:42],(0,1)).all(),
        'original S82 ownership 46D shape/mask mismatch')
    summary=acoustic['summary58']
    sequence=acoustic['sequence42x49']
    freeze=endpoint['incumbent_K']
    x=build_owned527(summary,original46,sequence,freeze)
    req(x.shape==(n,527) and np.isfinite(x).all(),
        'S5 original owned527 matrix invalid')
    out.mkdir(parents=True)
    path=out/'S236_original_owned527_native_reference_IDMT.npz'
    np.savez_compressed(path,
      row_id=acoustic['row_id'],member=acoustic['member'],
      start_samples=endpoint['start_samples'],
      end_samples=endpoint['end_samples'],
      incumbent_freeze_K=freeze,
      original_PR16_geometry46=original46,
      original_S5_input_owned527=x)
    rep=dict(status='S236_S5_ORIGINAL_527D_FROM_CANONICAL_FREEZE_V88_NATIVE_EVENTS',
        reference_baseline_run=38301633310,
        exact_native_V88_endpoints_run=38302495775,
        original_S97_audio_views_run=38302179397,
        original_native_event_count=n,
        original_input_feature_width=527,
        S4_owned416_derived_from_true_V88_geometry=True,
        original_46D_ownership_geometry=True,
        strict_reference_K_and_event_ID_identity_verified=True,
        model_training_527D_source_hist_replay_is_59309_rows=True,
        input_SHA256=sha(path.read_bytes()),
        original_trained_S5_checkpoint_applied_to_IDMT=False,
        exact_original_S5_teacher_probability_on_IDMT_unknown=True,
        recovered_full_original_S80_generalized_to_1034_new_events=False,
        XML_MIDI_content_read=False,
        training_labels_used_for_new_audio=False,
        models_fitted_on_new_audio=False,
        independent_accuracy_measured=False,
        S80_promoted=False)
    (out/'report.json').write_text(json.dumps(rep,indent=2)+'\n')
    print(json.dumps(rep,indent=2),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser()
    for key in ('s234','s235','out'):a.add_argument('--'+key,type=Path,required=True)
    q=a.parse_args();run(q.s234,q.s235,q.out)
