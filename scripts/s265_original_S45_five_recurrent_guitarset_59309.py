"""S265: original five recurrent S45 families on REAL S5/S18 GUITARSET parents.

Re-execute 19 historical held-piece pretrained RecurrentSelector S19,
ResidualAB S20, trained B-first S25, exact reverse-order S24, and
early-stable S27; use per-fold original S5 global/poly 335D acoustics
and original S18 parent instead of S135/PR16 adapters. Label-free
research branches; no held-piece assignment for future audio.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from scripts.loop_v273_native_risk import designs
from scripts.train_v273_aba_recurrent import RecurrentSelector
from scripts.train_v273_aba_residual import ResidualAB
from scripts.train_v273_Bfirst_recurrent import TrainedBfirst
from scripts.audit_v273_partial_revisits import run_policy
from scripts.audit_v273_inverse_ab_order import infer_reverse

FOLDS=(0,1,2,4)
FAMILIES=('S19_recurrent_ABA','S20_repeated_ABA',
          'S24_Bfirst','S25_Bfirst','S27_early_stop')
BAD={'true_K','truth','label','labels','target','midi','correct'}

def need(test,msg):
    if not test:raise ValueError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p,keys):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==set(keys) and not BAD.intersection(z.files),
             'strict S265 input schema '+str(p))
        return {k:z[k].copy() for k in z.files}
def loadbank(root,globpattern,code):
    files=sorted((root/'models').glob(globpattern))
    need(len(files)==19,'nineteen authentic source model files required '+code)
    bank={}
    for file in files:
        z=torch.load(file,map_location='cpu',weights_only=False)
        piece=str(z['piece']); fold=int(z['fold'])
        need(piece in file.name and fold in FOLDS and piece not in bank,
             'original held-piece checkpoint provenance '+code)
        mean=np.asarray(z['scaler_mean'],np.float32)
        sd=np.asarray(z['scaler_scale'],np.float32)
        need(mean.shape==sd.shape==(335,) and
             np.isfinite(mean).all() and np.all(sd>0),
             '335D original scaler')
        bank[piece]=(file,z)
    return bank
def make_input(audio,s18,fi):
    n=len(s18['row_id'])
    vote=designs({'native_position':np.arange(n),'global_index':np.arange(n),
                  'features':audio['summary58']},
                 np.arange(n),s18['freeze_K'],s18['global_K'][fi],
                 s18['poly_K'][fi],audio['geometry3'])['votes_time']
    X=np.column_stack((vote,audio['summary58'],audio['flow245'])).astype(np.float32)
    need(X.shape==(n,335) and np.isfinite(X).all(),'original trained recurrent acoustic source')
    return X
def scaled(X,state):
    z=np.clip((X-np.asarray(state['scaler_mean'],np.float32))/
        np.asarray(state['scaler_scale'],np.float32),-5,5)
    return torch.from_numpy(z.astype(np.float32))
def run(s18_path,s97,s19,s20,s25,out):
    need(not out.exists(),'immutable S265 output')
    chain=load(s18_path,{'row_id','freeze_K','fold_ids','global_K','poly_K',
        'S8_K','S9_K','S13_K','S14_K','S15_K','S16_K','S17_K','S18_K'})
    audio=load(s97,{'row_id','summary58','flow245','geometry3',
      'sequence42x49','morphology490','physics294','member','start_samples','end_samples'})
    ids=chain['row_id'];n=len(ids)
    need(n==59309 and np.array_equal(ids,audio['row_id']) and
         tuple(map(int,chain['fold_ids']))==FOLDS and
         chain['S18_K'].shape==(4,n),'original S18 parents and GUITARSET IDs')
    b19=loadbank(s19,'*__A_B_A.pt','S19')
    b20=loadbank(s20,'*__residual_A_B.pt','S20')
    b25=loadbank(s25,'*__trained_Bfirst.pt','S25')
    need(set(b19)==set(b20)==set(b25),'original 19 piece fold identities')
    torch.set_num_threads(2)
    folds=[];pieces=[];pred=[];proof=[]
    for fi,fold in enumerate(FOLDS):
        X=make_input(audio,chain,fi)
        parent=chain['S18_K'][fi].astype(np.int8)
        need(np.isin(parent,range(7)).all(),'invalid original S18 parent')
        original_parent=torch.from_numpy(parent.astype(np.int64))
        for piece in sorted(b20):
            f19,q19=b19[piece];f20,q20=b20[piece];f25,q25=b25[piece]
            if int(q20['fold'])!=fold:continue
            need(int(q19['fold'])==int(q25['fold'])==fold,
                 'source S19/S20/S25 original heldfold mismatch '+piece)
            neural19=RecurrentSelector()
            neural19.load_state_dict(q19['state_dict'],strict=True)
            neural19.eval()
            neural20=ResidualAB()
            neural20.load_state_dict(q20['model'],strict=True)
            neural20.eval()
            neural25=TrainedBfirst()
            neural25.load_state_dict(q25['model'],strict=True)
            neural25.eval()
            xx19=scaled(X,q19);xx20=scaled(X,q20);xx25=scaled(X,q25)
            with torch.no_grad():
                k19=neural19(xx19,passes=2)[2][1].argmax(1).numpy().astype(np.int8)
                k20=neural20(xx20,original_parent,passes=1)[2][0].argmax(1).numpy().astype(np.int8)
                stable,_,_=run_policy(neural20,xx20,original_parent,'early_stable_005')
                k27=stable.argmax(1).numpy().astype(np.int8)
                _,b24=infer_reverse(neural20,xx20,original_parent,passes=1)
                k24=torch.sigmoid(b24[0]).argmax(1).numpy().astype(np.int8)
                _,b25_logits,_=neural25(xx25,original_parent,passes=1)
                k25=torch.sigmoid(b25_logits[0]).argmax(1).numpy().astype(np.int8)
            ks=np.column_stack((k19,k20,k24,k25,k27)).astype(np.int8)
            need(ks.shape==(n,5) and np.isin(ks,range(7)).all(),
                 'five real recurrent source K classes')
            folds.append(fold);pieces.append(piece);pred.append(ks)
            proof.append(dict(piece=piece,fold=fold,sha_S19=sha(f19),
                sha_S20=sha(f20),sha_S25=sha(f25),
                source_changed_from_true_S18={
                   fam:int((ks[:,j]!=parent).sum()) for j,fam in enumerate(FAMILIES)}))
    need(len(pred)==19 and [folds.count(f) for f in FOLDS]==[5,4,5,5],
         'nineteen original heldpiece recurrent sources not replayed')
    out.mkdir(parents=True)
    np.savez_compressed(out/'five_rebased_recurrent_S19_S20_S24_S25_S27.npz',
       row_id=ids,branch_piece=np.asarray(pieces),branch_fold=np.asarray(folds,dtype=np.int8),
       family_ids=np.asarray(FAMILIES),proposals_K=np.stack(pred))
    result=dict(status='S265_FIVE_ORIGINAL_RECURRENT_S45_FAMILIES_REBASED',
       events=n,branches=19,fold_branch_counts=[5,4,5,5],
       original_S19_models=19,original_S20_models=19,original_S25_models=19,
       source_variants=list(FAMILIES),checkpoints=proof,
       source_S135_PR16_removed=True,source_parent_recovered_S18=True,
       original_models_unchanged=True,truth_labels_read=False,new_training=False,
       new_piece_heldpiece_aggregation_unresolved=True,
       historical_16_family_S45_equivalence=False,
       independent_ExactK_accuracy_measured=False,S80_promoted=False)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checkpoints'},indent=2),flush=True)
    print(json.dumps(proof),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('s18','s97','s19','s20','s25','out'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    run(a.s18,a.s97,a.s19,a.s20,a.s25,a.out)
