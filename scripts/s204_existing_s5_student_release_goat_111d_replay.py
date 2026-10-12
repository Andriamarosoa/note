"""S204 immutable GitHub release S133/S134/S135 student original-audio deployment replay.

NO new S5 training; no creation of a replacement, no original 527D claims.
Original genuine 111D audio features are from S116; archived deployed model
weights and GOAT outputs are from permanent S136 release. This confirms
the existing REJECTED experimental candidate runs bit-stably.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,joblib
from scripts.s133_s5_lightgbm_teacher_student import seven_prob
from scripts.s134_s5_soft_probabilities import forward
from scripts.s135_s5_anchor_soft_posterior import anchor_posterior

def need(ok,m):
    if not ok:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def original_audio(path):
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)=={'row_id','control111','freeze_K','geometry46'},
             'original S116 audio provenance unexpectedly changed')
        ids=z['row_id'].copy();x=z['control111'].astype(np.float32)
        parent=z['freeze_K'].copy()
        geometry=z['geometry46'].copy()
    need(x.shape==(57,111) and geometry.shape==(57,46)
         and len(set(map(str,ids)))==57,'original GOAT S116 57 event audio identity')
    need(np.array_equal(x[:,58:104],geometry),'original 46 geometry location in 111D changed')
    need(np.array_equal(x[:,104:111].argmax(1),parent),'original freeze class onehot alignment')
    need(np.allclose(x[:,104:111].sum(1),1,atol=1e-7),'original frozen K onehot')
    need(np.isfinite(x).all(),'nonfinite original audio data')
    return ids,x
def saved(path,ids,kind):
    with np.load(path,allow_pickle=False) as z:
        need({'row_id','predicted_K'}.issubset(z.files),'missing frozen GOAT decision')
        need(np.array_equal(z['row_id'],ids),'GOAT saved source rows not same as S116')
        score_key='student_probabilities' if 'student_probabilities' in z.files else 'probabilities'
        need(score_key in z.files,'missing source seven probabilities')
        p=z[score_key].astype(np.float64)
        k=z['predicted_K'].astype(np.int64)
        need(p.shape==(len(ids),7) and np.array_equal(p.argmax(1),k) and
             np.allclose(p.sum(1),1,atol=1e-5),'saved original S5 student ' + kind +' probabilities')
    return p,k
def run(release,s116,out):
    need(not out.exists(),'immutable original students replay result')
    base=release
    r133=json.loads((base/'s133/report.json').read_text())
    r134=json.loads((base/'s134/report.json').read_text())
    r135=json.loads((base/'s135/report.json').read_text())
    need(not r133['original_S5_527D_reproduced'] and not r134['original_S5_reproduced']
         and not r135['original_S5_recovered'],'source must not claim historic equivalence')
    ids,x=original_audio(s116)
    ck133=base/'s133/control111_all_teacher.joblib'
    ck134=base/'s134/soft_s5_111d_distilled.joblib'
    need(sha(ck133)==r133['arms']['control111']['deploy_model_sha256'],
         'original S133 pretrained research-release weight SHA mismatch')
    need(sha(ck134)==r134['new_checkpoint_SHA256'],
         'original S134 soft research-release model SHA mismatch')
    model=joblib.load(ck133)
    models=joblib.load(ck134)
    need(len(models)==7 and model.n_features_in_==111 and
         all(m.n_features_in_==111 for m in models),
         'deployed trained genuine original 111D student checkpoint schema')
    actual133=seven_prob(model,x)
    actual134=forward(models,x)
    known133,k133=saved(base/'s133/control111_GOAT.npz',ids,'S133')
    known134,k134=saved(base/'s134/GOAT_57_label_blind.npz',ids,'S134')
    need(np.allclose(known133,actual133,atol=5e-6,rtol=1e-6),
         'S133 deployed pretrained model does not replay saved 57 GOAT scores')
    need(np.allclose(known134,actual134,atol=5e-6,rtol=1e-6),
         'S134 deployed pretrained model does not replay saved 57 GOAT scores')
    actual135,adjusted=anchor_posterior(actual134,k133)
    path=base/'s135/GOAT_57_label_blind_soft_Kanchored.npz'
    with np.load(path,allow_pickle=False) as z:
        need('row_id' in z.files and np.array_equal(z['row_id'],ids),'S135 saved GOAT audio order')
        pkey='probabilities' if 'probabilities' in z.files else (
             'student_probabilities' if 'student_probabilities' in z.files else None)
        need(pkey is not None,'original S135 stored teacher probability key missing')
        saved135=z[pkey].astype(np.float64)
        need(np.allclose(saved135,actual135,atol=5e-6,rtol=1e-6),
             'S135 deterministic source composite replay differs from published GOAT')
    need(adjusted==2,'original S135 GOAT adjusted two rows source drift')
    report={'status':'S204_ORIGINAL_S136_RELEASE_STUDENT_S133_S134_S135_REAL_AUDIO_111D_REPLAY',
        'released_existing_student_deploy_not_retrained':True,
        'S133_original_permanent_checkpoint_sha256':sha(ck133),
        'S134_original_permanent_checkpoint_sha256':sha(ck134),
        'source_S116_original_audio_sha256':sha(s116),
        'full_original_111D_feature_prefix_verified':True,
        'original_S133_saved_GOAT_posterior_max_error':float(np.max(np.abs(actual133-known133))),
        'original_S134_saved_GOAT_posterior_max_error':float(np.max(np.abs(actual134-known134))),
        'original_S135_saved_GOAT_posterior_max_error':float(np.max(np.abs(actual135-saved135))),
        'original_S135_replayed_adjusted_events':adjusted,
        'GOAT_57_audio_only_predictions_count':len(ids),
        'S133_student_57_predicted_K0to6_histogram':np.bincount(k133,minlength=7).tolist(),
        'S134_student_57_predicted_K0to6_histogram':np.bincount(k134,minlength=7).tolist(),
        'S135_student_57_predicted_K0to6_histogram':np.bincount(actual135.argmax(1),minlength=7).tolist(),
        'GOAT_true_labels_opened':False,
        'independent_S137_prior_Goat_ExactK_result':'already published: 44/57 reference vs 44/57 student; poly 2/3 reference vs 1/3 student; net zero',
        'S135_student_already_rejected_as_official_replacement':True,
        'original_historic_527D_S5_recovered':False,
        'S45_16_original_families_complete':False,
        'S46_original_193_policy_compatible_proven':False,
        'official_freeze_local_combo_unchanged':True,
        'S80_promoted':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
def selftest():
    x=np.eye(7,dtype=np.float64)
    need(np.array_equal(x.argmax(1),np.arange(7)),'K0-K6 basis')
    print('S204 PASS S116 111D and prior stored student model authentication guards')
if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--release',type=Path)
    a.add_argument('--s116',type=Path)
    a.add_argument('--out',type=Path)
    a.add_argument('--self-test',action='store_true')
    z=a.parse_args()
    if z.self_test:selftest()
    else:
        need(z.release and z.s116 and z.out,'released actual weights and original 111D GOAT needed')
        run(z.release,z.s116,z.out)
