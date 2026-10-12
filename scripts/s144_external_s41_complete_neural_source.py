"""S144 genuine ORIGINAL S38+S40 expert pipelines into ORIGINAL S41 gates.

Real S38 4 expert views x4 folds, real S40 4 CNNs, and S41 19 saved
piece-excluded learned fusion checkpoint gates on 57 genuine GOAT audio
events. All original trained weights preserved. NEW deterministic mean
of original models by fold/heldpiece for unfamiliar recording; NEW
parent mapping b=freeze,g=S135,p=original PR16#22. Thus a reproducible
S41 deployment candidate but NOT exact historic S18/S41/S45/S80.
No MIDI label or ground-truth is loaded, no model fitted.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np,torch
from scripts.s141_external_original_s40_cnn import load_input,forward as s40_forward
from scripts.train_v273_s41_neural_fusion import FusionGate,make_proposals,gated_prediction,ACTION_NAMES
FOLDS=(0,1,2,4)
HEADS=("raw335__hgb","raw335__extra_trees","morph825__hgb","morph825__extra_trees")
def check(c,why):
    if not c:raise ValueError(why)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def normalize(p,model,shape):
    check(np.allclose(p.sum(1),1,atol=1e-4) and np.isfinite(p).all() and p.shape==shape,model+' posterior contract')
    return p
def run(s97,s85,s135,s38,s40,s41,out):
    check(not out.exists(),"immutable provenance")
    ids,x,seq,parents=load_input(s97,s85,s135)
    b=parents['official_freeze']
    with np.load(s97,allow_pickle=False) as z:
        morph=z['morphology490'].astype(np.float32)
    views=dict(raw335=x,morph825=np.column_stack((x,morph)).astype(np.float32))
    check(views['raw335'].shape==(57,335) and views['morph825'].shape==(57,825),'real source views')
    checks=[];experts=[]
    # Original 16 scikit-learn checkpoints from the four S38 heads x four heldfolds.
    for head in HEADS:
        a=[]
        view,algo=head.split('__',1)
        for fold in FOLDS:
            cp=s38/'models'/f'fold{fold}__{head}.joblib'
            check(cp.is_file(),'missing original S38 '+str(cp))
            state=joblib.load(cp)
            check(state['fold']==fold and state['view']==view and state['algo']==algo,
                  'original S38 provenance mismatch')
            estimator=state['estimator'];scaler=state['scaler']
            check(estimator.n_features_in_==views[view].shape[1],'S38 original features width')
            check(np.array_equal(estimator.classes_,np.arange(1,7)),
                  'S38 trained class set')
            z=np.clip(scaler.transform(views[view]),-5,5).astype(np.float32)
            raw=estimator.predict_proba(z)
            p=np.column_stack([np.zeros(57),raw]).astype(np.float32)
            normalize(p,'S38 '+head,(57,7))
            a.append(p)
            checks.append({'role':'S38_'+head,'fold':fold,'sha256':sha(cp)})
        experts.append(np.stack(a).mean(0))
    tab=np.stack(experts,axis=1).astype(np.float32)
    check(tab.shape==(57,4,7),'S41 4 real S38 expert posteriors')
    s40all=[]
    for fold in FOLDS:
        cp=s40/'models'/f'fold{fold}__temporalKnet.pt'
        check(cp.is_file(),'original S40 CNN missing '+str(cp))
        p=s40_forward(cp,x,seq,fold)
        normalize(p,'S40 CNN',(57,7))
        s40all.append(p)
        checks.append({'role':'S40_CNN','fold':fold,'sha256':sha(cp)})
    cnn=np.stack(s40all).mean(0)
    proposal=make_proposals(b,tab,cnn)
    check(proposal.shape==(57,6),'S41 exactly 6 action candidates')
    # One rule for NEW music fixed before applying saved models: average all
    # 19 original held-piece gate predictions without selecting GOAT labels.
    gates=[]
    s41paths=sorted((s41/'models').glob('*__sixhead_fusion.pt'))
    check(len(s41paths)==19,'S41 original 19 piece-held checkpoints required')
    torch.set_num_threads(2)
    for cp in s41paths:
        state=torch.load(cp,map_location='cpu',weights_only=False)
        check(state['held_piece'] in cp.name and state['fold'] in FOLDS,
              'saved original S41 checkpoint identity mismatch')
        mu=np.asarray(state['scaler_mean'],np.float32)
        sd=np.asarray(state['scaler_scale'],np.float32)
        check(mu.shape==(335,) and sd.shape==(335,) and np.all(sd>0),
              'original S41 trained acoustic scaler')
        scaled=np.clip((x-mu)/sd,-5,5).astype(np.float32)
        features=np.column_stack((scaled,np.eye(7,dtype=np.float32)[b],
                          tab.reshape(57,28),cnn)).astype(np.float32)
        check(features.shape==(57,377),'original S41 dense input contract')
        network=FusionGate()
        network.load_state_dict(state['model'],strict=True)
        network.eval()
        with torch.no_grad():
            p=torch.softmax(network(torch.from_numpy(features)),dim=1).numpy()
        normalize(p,'S41 6-route gate',(57,6))
        gates.append(p)
        checks.append({'role':'S41_sixhead_gate','fold':state['fold'],
                       'held_piece':state['held_piece'],'sha256':sha(cp)})
    scores=np.stack(gates).mean(0)
    require=np.asarray(scores.argmax(1),np.int8)
    k=gated_prediction(b,proposal,scores,'source_K1to6',0.)
    check(k.shape==(57,) and np.isin(k,range(7)).all(),'real S41 K0..6 output')
    out.mkdir(parents=True)
    np.savez_compressed(out/'original_s41_new_deploy_composite_GOAT.npz',
       row_id=ids,parent_K=b,proposal_K=proposal,gate_scores=scores,
       proposed_K=k,S38_four_expert_probabilities=tab,S40_cnn_probabilities=cnn,
       selected_route= require)
    report={
      'status':'S144_EXECUTED_ORIGINAL_S38_S40_S41_WEIGHTS_ON_NEW_AUDIO',
      'source_S38_run':37880386031,
      'source_S40_run':37880762289,
      'source_S41_run':37881191015,
      'all_39_original_weight_checkpoints_executed':len(checks)==39,
      'checkpoint_count':len(checks),'checkpoints':checks,
      'native_GOAT_audio_events':57,
      'six_routes':list(ACTION_NAMES),
      'route_selection_histogram':np.bincount(require,minlength=6).tolist(),
      'final_K_histogram':np.bincount(k,minlength=7).tolist(),
      'K_changed_vs_frozen_parent':int(np.sum(k!=b)),
      'unseen_music_rule':'S38 and S40 4fold mean; S41 19 heldpiece mean; no labels',
      'parent_adapter':'NEW b=freeze,g=S135,p=PR16#22 rather than original S18 S11',
      'full_historical_S41_equivalence_verified':False,
      'original_S45_16_families_complete':False,
      'new_S80_pipeline_validated':False,
      'MIDI_labels_read':False,'promotion':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in (
       'status','checkpoint_count','native_GOAT_audio_events',
       'route_selection_histogram','final_K_histogram','K_changed_vs_frozen_parent',
       'original_S45_16_families_complete','promotion')},indent=2))
def selftest():
    old=np.array([0,1,2],dtype=np.int8)
    tab=np.zeros((3,4,7),np.float32);cnn=np.zeros((3,7),np.float32)
    tab[:,:,1]=1;cnn[:,1]=1
    props=make_proposals(old,tab,cnn)
    check(props.shape==(3,6) and np.array_equal(props[:,0],old),'S41 six routes')
    print('S144 PASS original S41 six-route neural fusion contract')
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for field in ('s97','s85','s135','s38','s40','s41','out'):
        parser.add_argument('--'+field,type=Path)
    parser.add_argument('--self-test',action='store_true')
    a=parser.parse_args()
    if a.self_test:selftest()
    else:
        check(all(getattr(a,x) is not None for x in
                  ('s97','s85','s135','s38','s40','s41','out')),'seven inputs required')
        run(a.s97,a.s85,a.s135,a.s38,a.s40,a.s41,a.out)
