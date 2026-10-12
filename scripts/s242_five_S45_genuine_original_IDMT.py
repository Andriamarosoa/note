"""S242 restore FIVE original S45 producer formulas on reconstructed S5/S18.

Runs genuine saved S11, S12, S38, S40, S41 checkpoints for all 19
held-piece branches on NEW exposed IDMT 57 events. Uses actual four
reconstructed S5 global/poly votes, S9 and S18 fold parents rather than
obsolete S135+PR16 adapters. No fitting, truth, MIDI, fold/piece routing
rule for new music, or claim of historical out-of-piece identity.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np,torch
from scripts.loop_v273_native_risk import designs
from scripts.train_v273_s38_poly_harmonic_expert import score_predictions
from scripts.train_v273_s41_neural_fusion import FusionGate,make_proposals,gated_prediction
from scripts.s141_external_original_s40_cnn import forward as s40_forward
FOLDS=(0,1,2,4)
FAMILIES=('S11_votes_time_audio','S12_audio_logistic','S38_morphology',
          'S40_temporal_CNN','S41_neural_fusion')
HEADS=('raw335__hgb','raw335__extra_trees',
       'morph825__hgb','morph825__extra_trees')
BAD={'true_K','truth','label','labels','target','midi','correct'}
def need(v,m):
    if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p,cols):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==set(cols) and not BAD.intersection(z.files),
             'strict S242 input fields: '+str(p))
        return {k:z[k].copy() for k in z.files}
def paths_19(path,suffix):
    files=sorted((path/'models').glob('*__'+suffix))
    need(len(files)==19,'need 19 original held-piece checkpoints: '+suffix)
    return files
def predict7(st,X,seven=True):
    clf=st['model'];scaler=st['scaler']
    need(clf.n_features_in_==scaler.n_features_in_==X.shape[1],
         'historic trained estimator feature width mismatch')
    q=clf.predict_proba(np.clip(scaler.transform(X),-6,6))
    prob=np.zeros((len(X),7),np.float32)
    classes=np.asarray(clf.classes_,int)
    need(np.isin(classes,np.arange(7)).all(),'trained class ID mismatch')
    prob[:,classes]=q
    need(prob.shape==(len(X),7) and np.isfinite(prob).all() and
         np.allclose(prob.sum(1),1,atol=1e-4),'original estimator outputs malformed')
    return prob
def run(s18,s97,s11,s12,s38,s40,s41,out):
    need(not out.exists(),'refuse overwrite sealed S242 outputs')
    source=read(s18,{'row_id','freeze_K','fold_ids','global_K','poly_K',
       'S8_K','S9_K','S13_K','S14_K','S15_K','S16_K','S17_K','S18_K'})
    audio=read(s97,{'row_id','summary58','flow245','geometry3',
       'sequence42x49','morphology490','physics294','member','start_samples','end_samples'})
    n=len(source['row_id'])
    need(n==1034 and np.array_equal(source['row_id'],audio['row_id']) and
         tuple(map(int,source['fold_ids']))==FOLDS,'IDMT source + fold join')
    b=source['freeze_K'].astype(np.int8)
    need(np.isin(b,range(7)).all(),'invalid original freeze classes')
    s11_files=paths_19(s11,'votes_time_audio__hgb.joblib')
    s12_files=paths_19(s12,'logistic.joblib')
    s41_files=paths_19(s41,'sixhead_fusion.pt')
    def by_piece(files,load,key):
        result={}
        for file in files:
            state=load(file)
            name=str(state[key])
            need(name not in result,'duplicate original trained held-piece checkpoint: '+name)
            result[name]=(file,state)
        need(len(result)==19,'original saved 19 piece checkpoints')
        return result
    s11_state=by_piece(s11_files,joblib.load,'piece')
    s12_state=by_piece(s12_files,joblib.load,'piece')
    s41_state=by_piece(s41_files,lambda p:torch.load(p,map_location='cpu',weights_only=False),'held_piece')
    need(set(s11_state)==set(s12_state)==set(s41_state),'all 19 original heldpiece identities')
    torch.set_num_threads(2)
    out_piece=[];out_fold=[];proposals=[];proof=[]
    fold_report=[]
    for fi,fold in enumerate(FOLDS):
        g,p=source['global_K'][fi].astype(np.int8),source['poly_K'][fi].astype(np.int8)
        s9,s18parent=source['S9_K'][fi].astype(np.int8),source['S18_K'][fi].astype(np.int8)
        need(all(z.shape==(n,) and np.isin(z,range(7)).all() for z in (g,p,s9,s18parent)),
             'original fold parent K0..6 class coverage')
        design=designs({'native_position':np.arange(n),'global_index':np.arange(n),
                        'features':audio['summary58']},
                       np.arange(n),b,g,p,audio['geometry3'])
        x=np.column_stack((design['votes_time'],audio['summary58'],audio['flow245'])).astype(np.float32)
        x90=x[:,:90]
        need(x.shape==(n,335) and design['votes_time_audio'].shape==(n,91)
             and x90.shape==(n,90) and np.isfinite(x).all(),'exact original source features')
        views={'raw335':x,
               'morph825':np.column_stack((x,audio['morphology490'])).astype(np.float32)}
        tab=[]
        for head in HEADS:
            file=s38/'models'/f'fold{fold}__{head}.joblib'
            need(file.is_file(),'original S38 checkpoint absent '+str(file))
            st=joblib.load(file)
            view,alg=head.split('__',1)
            need(int(st['fold'])==fold and st['view']==view and st['algo']==alg,
                 'S38 original fold and estimator contract')
            clf=st['estimator'];sc=st['scaler']
            need(np.array_equal(clf.classes_,np.arange(1,7)) and
                 clf.n_features_in_==sc.n_features_in_==views[view].shape[1],
                 'S38 source classes and view drift')
            q=clf.predict_proba(np.clip(sc.transform(views[view]),-5,5))
            tab.append(np.column_stack((np.zeros(n),q)).astype(np.float32))
            proof.append(dict(fold=fold,type='S38',head=head,sha256=sha(file)))
        tab=np.stack(tab,axis=1)
        source38=score_predictions(s18parent,tab[:,3].argmax(1),tab[:,3],
                                    'source_K1to6','argmax')
        cnnfile=s40/'models'/f'fold{fold}__temporalKnet.pt'
        need(cnnfile.is_file(),'S40 original trained CNN missing')
        cnn=s40_forward(cnnfile,x,audio['sequence42x49'],fold)
        source40=np.where(s18parent>=1,cnn.argmax(1),s18parent).astype(np.int8)
        proof.append(dict(fold=fold,type='S40',sha256=sha(cnnfile)))
        s41proposal=make_proposals(s18parent,tab,cnn)
        fold_count=0
        for piece in sorted(s11_state):
            file11,st11=s11_state[piece]
            file12,st12=s12_state[piece]
            file41,st41=s41_state[piece]
            need(int(st11['fold'])==int(st12['fold'])==int(st41['fold']) or True,
                 'not used')
            if int(st11['fold'])!=fold:continue
            need(int(st12['fold'])==int(st41['fold'])==fold,
                 'original S11/12/41 piece fold mismatch')
            need(st11['arm']=='votes_time_audio' and st11['kind']=='hgb',
                 'incorrect S11 original variant')
            q11=predict7(st11,design['votes_time_audio'])
            advantage=q11[np.arange(n),b]-q11[np.arange(n),s9]
            k11=np.where((s9!=b)&(advantage>0),b,s9).astype(np.int8)
            q12=predict7(st12,x90)
            margin=q12[np.arange(n),b]-q12[np.arange(n),s9]
            k12=np.where((s9!=b)&(margin>0),b,s9).astype(np.int8)
            mu=np.asarray(st41['scaler_mean'],np.float32)
            sd=np.asarray(st41['scaler_scale'],np.float32)
            need(mu.shape==sd.shape==(335,) and np.all(sd>0),
                 'S41 original heldpiece normalization')
            scaled=np.clip((x-mu)/sd,-5,5)
            features=np.column_stack((scaled,np.eye(7,dtype=np.float32)[s18parent],
                 tab.reshape(n,28),cnn)).astype(np.float32)
            need(features.shape==(n,377),'original S41 377D fusion contract')
            net=FusionGate()
            net.load_state_dict(st41['model'],strict=True)
            net.eval()
            with torch.no_grad():
                q41=torch.softmax(net(torch.from_numpy(features)),dim=1).numpy()
            need(q41.shape==(n,6) and np.isfinite(q41).all(),'real S41 action posteriors')
            k41=gated_prediction(s18parent,s41proposal,q41,'source_K1to6',0.)
            result=np.column_stack((k11,k12,source38,source40,k41)).astype(np.int8)
            need(result.shape==(n,5) and np.isin(result,range(7)).all(),
                 'original 5-family class contract')
            proposals.append(result);out_piece.append(piece);out_fold.append(fold)
            proof.extend([
                dict(fold=fold,piece=piece,type='S11',sha256=sha(file11)),
                dict(fold=fold,piece=piece,type='S12',sha256=sha(file12)),
                dict(fold=fold,piece=piece,type='S41',sha256=sha(file41))])
            fold_count+=1
        fold_report.append(dict(fold=fold,branches=fold_count,
            S38_changed_events=int(np.count_nonzero(source38!=s18parent)),
            S40_changed_events=int(np.count_nonzero(source40!=s18parent))))
    need(len(proposals)==19 and [out_fold.count(f) for f in FOLDS]==[5,4,5,5],
         'original 19 branch five-family coverage')
    out.mkdir(parents=True)
    np.savez_compressed(out/'five_rebased_S11_S12_S38_S40_S41.npz',
        row_id=source['row_id'],branch_piece=np.asarray(out_piece),
        branch_fold=np.asarray(out_fold,dtype=np.int8),
        family_ids=np.asarray(FAMILIES),
        proposals_K=np.stack(proposals).astype(np.int8))
    report=dict(status='S242_FIVE_REAL_FAMILIES_ON_RECOVERED_S5_S18_PARENTS',
        events=n,branches=19,fold_branches=[5,4,5,5],family_ids=list(FAMILIES),
        checkpoint_count=len(proof),checkpoint_sha256=proof,folds=fold_report,
        former_S135_PR16_source_replaced=True,
        S11_S12_original_S9_parent_used=True,
        S38_S40_S41_original_S18_parent_used=True,
        original_held_piece_routing_available_for_unseen_music=False,
        other_original_S45_sources_fully_reconstructed=False,
        labels_read=False,new_models_trained=False,
        independent_accuracy_measured=False,S80_promoted=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checkpoint_sha256'},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('s18','s97','s11','s12','s38','s40','s41','out'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    run(a.s18,a.s97,a.s11,a.s12,a.s38,a.s40,a.s41,a.out)
