"""S238 genuine historical S8→S18 trained cascade on 1034 IDMT V88 native events.

Consumes S237 four ORIGINAL trained S5 fold probabilities, S234 original
audio summary58/flow245 and S235 exact original V88 event boundaries;
recovers original unchanged 133 frozen logistic checkpoint predictors
from three SHA-pinned release archives. Reuses original historical
vote32/acoustic335 input builders and exact stage S8/S9/S13–S18
original selection/decoding rules from v273_recovered_s5_s18.py.
Each historical held-piece head only votes within its fold; four folds
remain separate. NO labels/XML/MIDI, no new training or fold oracle,
and no accuracy/promotion claim.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np
from scripts.v273_recovered_s5_s18 import (
  FOLDS,SELECTED,prepare_models,acoustic335,score,decode,tilt)
from scripts.s97_external_s45_original_features import geometry
from scripts.loop_v273_learned_risk import design as vote_design
def need(ok,msg):
    if not ok:raise ValueError(msg)
def read(p,expected):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==set(expected),
             'S238 only label-free upstream source: '+str(p)+' observed '+str(z.files))
        return {k:z[k].copy() for k in z.files}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(s237file,s234file,s235file,cache,out):
    need(not out.exists(),'S238 never overwrites immutable independent test outputs')
    s5=read(s237file,{'row_id','incumbent_freeze_K','fold_ids',
       'original_S5_raw_probability_by_fold','original_S5_prior_probability_by_fold',
       'original_S5_K_by_fold','all_four_folds_agree'})
    audio=read(s234file,{'row_id','member','start_samples',
       'incumbent_freeze_K','summary58','sequence42x49','flow245',
       'morphology490','physics294'})
    timing=read(s235file,{'row_id','member','start_samples','end_samples',
       'incumbent_K','incumbent_probabilities',
       'nearest_S232_V84_onset_distance_samples'})
    ids=s5['row_id'];n=len(ids);b=s5['incumbent_freeze_K']
    need(n==1034 and np.array_equal(s5['fold_ids'],np.asarray(FOLDS)) and
       np.array_equal(ids,audio['row_id']) and
       np.array_equal(ids,timing['row_id']) and
       np.array_equal(b,audio['incumbent_freeze_K']) and
       np.array_equal(b,timing['incumbent_K']) and
       np.array_equal(timing['member'],audio['member']) and
       np.array_equal(timing['start_samples'],audio['start_samples']),
       'S238 1034 original native event identities or baseline K differ')
    raw=s5['original_S5_raw_probability_by_fold']
    corrected=s5['original_S5_prior_probability_by_fold']
    need(raw.shape==(4,n,7) and corrected.shape==(4,n,7),
       'all four original S5 saved K0–K6 posteriors required')
    g=tilt(corrected);p=tilt(raw)
    summary=audio['summary58'];flow=audio['flow245']
    geom=geometry(audio['member'],timing['start_samples'],timing['end_samples'])
    need(geom.shape==(n,3) and np.isfinite(geom).all(),
       'original source 3D timing features must be derived from real V88 clusters')
    acoustic=[acoustic335(summary,flow,geom,b,g[i],p[i]) for i in range(4)]
    votes=[vote_design({'features':summary},b,g[i],p[i])['votes']
           for i in range(4)]
    need(all(x.shape==(n,335) for x in acoustic) and
       all(x.shape==(n,32) for x in votes),
       'historic trained risk model input widths differ')
    model_roots=prepare_models(cache)
    required={'S8','S13','S14','S16','S17','S18'}
    need(required.issubset(model_roots),'original frozen trained stage archives incomplete')
    outputs={}
    parent=p.copy()
    patterns={'S8':('*__votes__C0.3.joblib',),
      'S13':('*__flow_logistic.joblib',),
      'S14':('*__flow_logistic.joblib','*__flow_hgb.joblib'),
      'S16':('k23__*__flow_logistic.joblib',),
      'S17':('k32__*__flow_logistic.joblib',),
      'S18':('k21__*__flow_logistic.joblib',)}
    proof=[];stage_report=[]
    for stage in ('S8','S13','S14','S16','S17','S18'):
        family_views=[]
        fold_counts=[]
        for pattern in patterns[stage]:
            selected=sorted((model_roots[stage]/'models').glob(pattern))
            need(len(selected)==19,'missing exact 19 historic piece checkpoints '+stage+' '+pattern)
            per_fold=[[] for _ in FOLDS]
            for file in selected:
                state=joblib.load(file)
                fold=int(state['fold'])
                need(fold in FOLDS,'original checkpoint missing historical fold')
                fi=FOLDS.index(fold)
                need('model' in state and 'scaler' in state,
                    'unmodified original trained classifier/scaler required')
                X=votes[fi] if stage=='S8' else acoustic[fi]
                q=score(state['model'],state['scaler'],X,stage=='S8')
                need(len(q)==n and np.isfinite(q).all(),
                     'real saved original trained model output invalid')
                per_fold[fi].append(q)
                proof.append(dict(stage=stage,original_checkpoint=file.name,
                    original_fold=fold,sha256=sha(file)))
            need([len(x) for x in per_fold]==[5,4,5,5],
                 'original held-piece checkpoint counts per fold drifted '+stage)
            family_views.append(np.stack([np.mean(q,axis=0) for q in per_fold]))
            fold_counts.append([len(q) for q in per_fold])
        mean=np.mean(family_views,axis=0)
        current=outputs['S9'] if stage in ('S13','S14') else parent
        result=np.stack([decode(stage,b,g[i],p[i],current[i],mean[i])
                         for i in range(4)],axis=0)
        need(result.shape==(4,n) and np.isin(result,range(7)).all(),
             'original frozen K0-K6 stage output invalid '+stage)
        outputs[stage]=result
        stage_report.append(dict(original_stage=stage,
            original_policy_key=SELECTED[stage],
            checkpoint_counts_by_family_and_fold=fold_counts,
            original_stage_changed_K_vs_own_parent_by_fold=list(map(int,
                np.count_nonzero(result!=current,axis=1))),
            original_stage_prediction_histogram_by_fold=[
                np.bincount(z,minlength=7).tolist() for z in result]))
        parent=result
        if stage=='S8':
            parent=np.where((p>=5)&(g<=3),p,result).astype(np.int8)
            outputs['S9']=parent.copy()
        elif stage=='S14':
            parent=np.where((outputs['S13']!=outputs['S9']) |
                  (result!=outputs['S9']),b[None,:],outputs['S9']).astype(np.int8)
            outputs['S15']=parent.copy()
        print(json.dumps(stage_report[-1]),flush=True)
    need(len(proof)==133 and len(outputs)==8 and
         all(outputs[x].shape==(4,n) for x in outputs),
         'expected exactly 133 original trained stage checkpoints and 8 frozen outputs')
    out.mkdir(parents=True)
    predpath=out/'S238_original_S5_to_S18_true_V88_native_IDMT_1034.npz'
    np.savez_compressed(predpath,row_id=ids,freeze_K=b,fold_ids=np.array(FOLDS),
      global_K=g,poly_K=p,**{stage+'_K':value for stage,value in outputs.items()})
    features=out/'S238_original_335D_acoustic_by_fold_1034.npz'
    np.savez_compressed(features,row_id=ids,fold_ids=np.array(FOLDS),
      acoustic335_by_fold=np.stack(acoustic),
      geometry3=geom,summary58=summary,flow245=flow)
    result=dict(status='S238_GENUINE_ORIGINAL_TRAINED_S8_TO_S18_ON_NATIVE_IDMT',
      count=n,original_baseline_V88_native=True,
      original_S5_trained_fold_count=4,
      original_risk_checkpoint_count=len(proof),original_decoded_stage_count=len(outputs),
      original_sealed_S5_parents_run=38303503793,
      original_audio_source_run=38302179397,
      original_boundary_source_run=38303449423,
      stage_proofs=stage_report,
      saved_original_checkpoint_SHA256=proof,
      S18_fourfold_K_histograms=[np.bincount(v,minlength=7).tolist()
                                  for v in outputs['S18']],
      S18_fourfold_disagreement_events=int(np.count_nonzero(
          np.any(outputs['S18']!=outputs['S18'][0],axis=0))),
      original_risk_models_not_retrained=True,
      original_training_selected_thresholds_preserved=True,
      historical_replay_previously_verified_on_59309_rows=True,
      unknown_song_fold_not_selected=True,
      XML_MIDI_annotation_content_read=False,
      S29_S35_S45_S46_S69_to_S80_not_yet_computed_on_IDMT=True,
      independent_ExactK_accuracy_measured=False,
      official_S80_promotion=False,
      prediction_artifact_SHA256=sha(predpath),
      acoustic_input_SHA256=sha(features))
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()
                 if k not in ('saved_original_checkpoint_SHA256','stage_proofs')},
                 indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('s237','s234','s235','cache','out'):
        p.add_argument('--'+key,type=Path,required=True)
    z=p.parse_args()
    run(z.s237,z.s234,z.s235,z.cache,z.out)
