"""S141: execute four ORIGINAL S40 saved temporal CNN checkpoints on new GOAT audio.

Real original 42x49 trajectories and S97 original audio summaries+flow.
The S40 'votes32' source requires historic S11 baseline/global/poly, which
do NOT yet deploy exactly. Therefore a NEW preregistered GOAT parent mapping
is used: b=official freeze, g=S135 new student, p=original PR16 policy22.
This is a research-only *new deployment composite*, never S40 historical
equivalent nor the real S45/S80 original. No MIDI/truth used.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from scripts.train_v273_s40_temporal_cnn import TemporalKNet
from scripts.loop_v273_native_risk import designs

FOLDS=(0,1,2,4)
FORBIDDEN={'truth','true_K','label','labels','y','target','midi','k'}
def check(x,why):
    if not x:raise ValueError(why)
def strict(path,keys):
    with np.load(path,allow_pickle=False) as z:
        check(set(z.files)==set(keys) and not set(z.files).intersection(FORBIDDEN),
              f'label-free array contract: {path}')
        return {k:z[k].copy() for k in z.files}
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_input(s97,s85,s135):
    a=strict(s97,('row_id','summary58','flow245','geometry3','sequence42x49',
                  'morphology490','physics294','member','start_samples','end_samples'))
    b=strict(s85,('member','start_samples','end_samples','incumbent_freeze_K',
                  'pr16_policy_K','pr16_vote_count','pr16_policy22_K',
                  'pr16_network_probabilities'))
    c=strict(s135,('row_id','predicted_K','probabilities'))
    ids=a['row_id'].astype(str)
    check(len(ids)==57 and np.array_equal(ids,c['row_id'].astype(str)), 'GOAT row native ID join')
    check(all(np.array_equal(a[key],b[key]) for key in ('member','start_samples','end_samples')),
          'GOAT timing join')
    seq=a['sequence42x49'].astype(np.float32)
    summary=a['summary58'].astype(np.float32)
    flow=a['flow245'].astype(np.float32)
    geom=a['geometry3'].astype(np.float32)
    b0=b['incumbent_freeze_K'].astype(np.int8)
    g=c['predicted_K'].astype(np.int8)
    p=b['pr16_policy22_K'].astype(np.int8)
    check(seq.shape==(57,42,49) and summary.shape==(57,58) and
          flow.shape==(57,245) and geom.shape==(57,3),'native acoustic contracts')
    check(all(np.isin(k,range(7)).all() for k in (b0,g,p)),'K0..K6 models')
    check(np.isfinite(seq).all() and np.isfinite(summary).all()
          and np.isfinite(flow).all() and np.isfinite(geom).all(),'nonfinite audio feature')
    vote=designs(dict(native_position=np.arange(57),global_index=np.arange(57),
                      features=summary),np.arange(57),b0,g,p,geom)['votes_time']
    x=np.column_stack((vote,summary,flow)).astype(np.float32)
    check(x.shape==(57,335),'historic S40 input geometry: votes32+summary58+flow245')
    return ids,x,seq,dict(official_freeze=b0,student_S135=g,original_PR16_policy22=p)

def safe_checkpoint(path,fold):
    check(path.is_file(),f'missing genuine S40 fold {fold} checkpoint')
    # Only the pinned trusted GitHub original run archive may be loaded.
    src=torch.load(path,map_location='cpu',weights_only=False)
    check(src['held_fold']==fold,'held fold did not match trusted run')
    mean=np.asarray(src['audio_mean'],np.float32)
    scale=np.asarray(src['audio_scale'],np.float32)
    tmean=np.asarray(src['time_mean'],np.float32)
    tstd=np.asarray(src['time_std'],np.float32)
    check(mean.shape==(335,) and scale.shape==(335,) and
          tmean.shape==(1,1,49) and tstd.shape==(1,1,49),
          'S40 checkpoint normalization dims')
    check(np.isfinite(mean).all() and np.all(scale>0)
          and np.isfinite(scale).all() and np.all(tstd>.0),
          'S40 checkpoint native scaler')
    net=TemporalKNet()
    net.load_state_dict(src['state_dict'],strict=True)
    net.eval()
    return net,mean,scale,tmean,tstd
def forward(src,x,seq,fold):
    net,mean,scale,tmean,tstd=safe_checkpoint(src,fold)
    acoustic=np.clip((x-mean)/scale,-5,5).astype(np.float32)
    temporal=np.clip((seq-tmean)/tstd,-5,5).transpose(0,2,1).copy()
    with torch.no_grad():
        raw=net(torch.from_numpy(temporal),torch.from_numpy(acoustic))
        p=torch.softmax(raw,dim=1).numpy()
    probs=np.column_stack((np.zeros(len(x),np.float32),p.astype(np.float32)))
    check(probs.shape==(57,7) and np.isfinite(probs).all()
          and np.allclose(probs.sum(1),1,atol=1e-5),'S40 authentic trained model posteriors')
    return probs
def run(s97,s85,s135,s40,out):
    check(not out.exists(),'immutable outputs')
    ids,x,seq,parents=load_input(s97,s85,s135)
    torch.set_num_threads(2)
    all_prob=[];folds=[]
    for fold in FOLDS:
        model=s40/'models'/f'fold{fold}__temporalKnet.pt'
        probs=forward(model,x,seq,fold)
        all_prob.append(probs)
        folds.append({'fold':fold,'checkpoint_sha256':sha(model),
                      'GOAT_K_counts':np.bincount(probs.argmax(1),minlength=7).tolist(),
                      'original_trained_weights_forward':True})
        print(json.dumps(folds[-1]),flush=True)
    stack=np.stack(all_prob,axis=0)
    avg=stack.mean(0)
    predictions=avg.argmax(1).astype(np.int8)
    check(np.isin(predictions,range(1,7)).all(),'S40 is K1..K6 only')
    out.mkdir(parents=True)
    np.savez_compressed(out/'native_s40_GOAT_research.npz',
        row_id=ids,fold_ids=np.asarray(FOLDS),fold_probabilities=stack,
        mean_probabilities=avg,predicted_K=predictions,
        parent_freeze_K=parents['official_freeze'],
        parent_s135_K=parents['student_S135'],
        parent_policy22_K=parents['original_PR16_policy22'])
    report={
      'status':'S141_FOUR_REAL_ORIGINAL_S40_CHECKPOINTS_EXECUTED_ON_GOAT',
      'genuine_S40_checkpoint_run':37880762289,
      'frozen_original_CNN_weights_unchanged':True,
      'GOAT_audio_event_count':57,'checkpoints':folds,
      'mean_ensemble_rule':'equally weight fold0/1/2/4 trained S40 outputs; fixed before GOAT label scoring',
      'K_histogram_0to6':np.bincount(predictions,minlength=7).tolist(),
      'S40_parent_inputs_original_equivalence':False,
      'S40_parent_mapping':'NEW DEPLOYMENT ADAPTATION b=freeze_local_combo, g=S135, p=PR16 policy22',
      'S45_original_16_families_reproduced':False,
      'full_S80_external_pipeline_executed':False,
      'GOAT_MIDI_opened':False,'original_S80_ExactK_scored':False,'promoted':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
def selftest():
    net=TemporalKNet()
    with torch.no_grad():
        y=net(torch.zeros(3,49,42),torch.zeros(3,335))
    check(y.shape==(3,6),'genuine torch S40 model shape')
    print('S141 PASS: historical S40 class and K1..K6 original state-layout')
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('s97','s85','s135','s40','out'):ap.add_argument('--'+name,type=Path)
    ap.add_argument('--self-test',action='store_true')
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        check(all(getattr(a,n) is not None for n in ('s97','s85','s135','s40','out')),
              'need five pinned audio/checkpoint paths')
        run(a.s97,a.s85,a.s135,a.s40,a.out)
