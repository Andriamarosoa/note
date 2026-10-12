"""S40 authentic temporal convolution over 42x49 native note trajectories.

Train K1..K6 only on three outer folds, test the held fourth fold.
H9/YourMT3+ never enters the neural input. No label-based test gates.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from sklearn.preprocessing import StandardScaler
from scripts.train_v273_aba_recurrent import prepare
from scripts.train_v273_s38_poly_harmonic_expert import score_predictions
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require

SEED=27340
EPOCHS=10
BATCH=256

class TemporalKNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal=nn.Sequential(
            nn.Conv1d(49,64,5,padding=2),nn.BatchNorm1d(64),nn.GELU(),
            nn.Conv1d(64,128,3,padding=1),nn.BatchNorm1d(128),nn.GELU())
        self.audio=nn.Sequential(nn.Linear(335,96),nn.GELU())
        self.output=nn.Sequential(nn.Linear(352,128),nn.GELU(),
                                  nn.Dropout(.20),nn.Linear(128,6))
    def forward(self,seq,acoustic):
        z=self.temporal(seq)
        z=torch.cat((z.mean(dim=2),z.amax(dim=2),
                      self.audio(acoustic)),dim=1)
        return self.output(z)

def digest(ids):
    return hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()

def load_native_trajectory(features,d):
    n=len(d['ids'])
    arr=np.full((n,42,49),np.nan,np.float32)
    byid={int(x):i for i,x in enumerate(d['ids'])}
    records=[]
    for f in FOLDS:
        paths=list(features.rglob(f'features-fold-{f}.npz'))
        require(len(paths)==1,'original native 42x49 sequence unavailable')
        with np.load(paths[0],allow_pickle=False) as z:
            source=z['global_index']
            indices=np.asarray([byid[int(v)] for v in source],np.int64)
            require(np.all(d['fold'][indices]==f),'fold sequence alignment changed')
            arr[indices]=np.asarray(z['sequence'],np.float32)
            records.append(dict(fold=int(f),rows=int(len(indices)),
                                native_id_sha256=digest(source)))
    require(np.isfinite(arr).all() and arr.min()>=0,
            'original sound trajectories contain invalid amplitude')
    return arr,records

def selftest():
    net=TemporalKNet()
    with torch.no_grad():
        y=net(torch.randn(4,49,42),torch.randn(4,335))
    require(y.shape==(4,6),'CNN K1-6 output mismatch')
    require(sum(p.numel() for p in net.parameters())>35000,
            'not a real learned temporal model')
    print('PASS: actual convolutional time-series network with 6 K1..6 heads and no H9')

def run(a):
    require(not a.output.exists(),'no overwrite prior S40')
    start=time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    d=prepare(a)
    n=len(d['ids'])
    old=d['parent']
    require(n==59309 and metrics(d['y'],old)['correct']==49178
            and metrics(d['y'],old)['poly']['correct']==2998,
            'S18 reference was modified')
    raw,record=load_native_trajectory(a.features,d)
    probs=np.full((n,7),np.nan,np.float32)
    fit_report=[]
    a.output.mkdir(parents=True)
    (a.output/'models').mkdir()
    for fold in FOLDS:
        torch.manual_seed(SEED+int(fold))
        train=np.flatnonzero((d['fold']!=fold)&(d['y']>=1))
        held=np.flatnonzero(d['fold']==fold)
        assert not np.intersect1d(train,held).size
        y=(d['y'][train]-1).astype(np.int64)
        freq=np.bincount(y,minlength=6)
        require(np.all(freq>0),'K1..6 must all be present')
        weights=np.minimum(np.sqrt(freq.max()/freq),5).astype(np.float32)
        scale=StandardScaler().fit(d['X'][train])
        x_train=np.clip(scale.transform(d['X'][train]),-5,5).astype(np.float32)
        x_test=np.clip(scale.transform(d['X'][held]),-5,5).astype(np.float32)
        # Train-only per-channel normalization across time.
        center=raw[train].mean(axis=(0,1),keepdims=True)
        std=np.maximum(raw[train].std(axis=(0,1),keepdims=True),.05)
        seq_train=np.clip((raw[train]-center)/std,-5,5).transpose(0,2,1).copy()
        seq_test=np.clip((raw[held]-center)/std,-5,5).transpose(0,2,1).copy()
        dataset=TensorDataset(torch.from_numpy(seq_train),
                              torch.from_numpy(x_train),torch.from_numpy(y))
        loader=DataLoader(dataset,batch_size=BATCH,shuffle=True,num_workers=0,
                          generator=torch.Generator().manual_seed(SEED+fold))
        net=TemporalKNet()
        optim=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
        loss=nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
        per_epoch=[]
        for epoch in range(EPOCHS):
            net.train()
            total=0.;count=0
            for seq,x,yy in loader:
                optim.zero_grad(set_to_none=True)
                p=net(seq,x)
                value=loss(p,yy)
                value.backward()
                optim.step()
                total+=float(value.detach())*len(yy)
                count+=len(yy)
            per_epoch.append(round(total/max(count,1),6))
        net.eval()
        pred_parts=[]
        with torch.no_grad():
            for start_idx in range(0,len(held),512):
                p=net(torch.from_numpy(seq_test[start_idx:start_idx+512]),
                      torch.from_numpy(x_test[start_idx:start_idx+512]))
                pred_parts.append(torch.softmax(p,dim=1).numpy())
        p=np.vstack(pred_parts)
        probs[held]=np.column_stack([np.zeros(len(held),np.float32),p])
        modelpath=a.output/'models'/f'fold{fold}__temporalKnet.pt'
        torch.save(dict(state_dict=net.state_dict(),
            audio_mean=scale.mean_.astype(np.float32),
            audio_scale=scale.scale_.astype(np.float32),
            time_mean=center.astype(np.float32),
            time_std=std.astype(np.float32),
            held_fold=int(fold),train_native_ids=d['ids'][train],
            held_native_ids=d['ids'][held],
            fit_sha256=digest(d['ids'][train]),
            held_sha256=digest(d['ids'][held]),
            loss_by_epoch=per_epoch),modelpath)
        fit_report.append(dict(fold=int(fold),models=1,train=int(len(train)),
            held=int(len(held)),file=str(modelpath.relative_to(a.output)),
            epoch_count=EPOCHS,loss_by_epoch=per_epoch,
            fit_sha256=digest(d['ids'][train]),
            held_sha256=digest(d['ids'][held])))
        print(json.dumps(dict(fold=int(fold),held=int(len(held)),
              last_loss=per_epoch[-1],elapsed=round(time.monotonic()-start,1))),
              flush=True)
    require(np.isfinite(probs).all() and
            np.allclose(probs.sum(1),1,atol=1e-4),
            'incomplete held-fold K-class scores')
    standalone=probs.argmax(1).astype(np.int8)
    outcome={'series18_parent':old.copy()}
    for domain in ('source_K1to6','source_poly_K2to6'):
        for gate in ('argmax','delta_gt0.10','delta_gt0.25'):
            key=f'series40__temporalCNN__{domain}__{gate}'
            outcome[key]=score_predictions(old,standalone,probs,domain,gate)
    analyses={}
    for key,v in outcome.items():
        change=v!=old
        fix=change&(old!=d['y'])&(v==d['y'])
        reg=change&(old==d['y'])&(v!=d['y'])
        analyses[key]=dict(metrics=metrics(d['y'],v),
            versus_S18=paired(d['y'],old,v),
            corrected=int(fix.sum()),regressed=int(reg.sum()),
            per_true_K={str(k):dict(
                fixed=int(((d['y']==k)&fix).sum()),
                broken=int(((d['y']==k)&reg).sum())) for k in range(7)})
    ranking=sorted(analyses,key=lambda k:(analyses[k]['metrics']['correct'],
                           analyses[k]['metrics']['poly']['correct']),reverse=True)
    poly=d['y']>=2
    diagnostic=dict(standalone_poly_correct=int(((standalone==d['y'])&poly).sum()),
        standalone_poly_exact=float((standalone[poly]==d['y'][poly]).mean()),
        per_true_K={str(k):int(((d['y']==k)&(standalone==d['y'])).sum())
                    for k in range(2,7)},
        newly_recognized_poly_vs_S18=int(((standalone==d['y'])&
                                        (old!=d['y'])&poly).sum()),
        old_poly_correct_missed=int(((standalone!=d['y'])&
                                    (old==d['y'])&poly).sum()))
    report=dict(status='completed',network='temporal_Conv1D_and_acoustic_MLP',
        H9_excluded=True,H8_not_simulated=True,
        old_S18_unchanged=True,fold_held_excluded_from_fit=True,
        new_compositions_unseen_from_research_start=False,
        no_production_promotion=True,
        original_42x49_sources=record,
        trained_models=fit_report,standalone_diagnostic=diagnostic,
        interventions=analyses,best_global=ranking[0],
        elapsed_seconds=round(time.monotonic()-start,1))
    (a.output/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    np.savez_compressed(a.output/'predictions.npz',
        global_index=d['ids'],true_K=d['y'],fold=d['fold'],
        variant_ids=np.asarray(list(outcome)),
        predictions=np.column_stack(list(outcome.values())))
    np.savez_compressed(a.output/'temporal_probs.npz',
        global_index=d['ids'],probabilities=probs)
    lines=['# S40 real temporal CNN: 42-frame note trajectory + acoustic input',
        '', f"Standalone true-K2–K6 accuracy: {diagnostic['standalone_poly_exact']*100:.4f}%"
          f" ({diagnostic['standalone_poly_correct']}/7385), versus S18 40.5958%.",
        f"Correct extra poly candidates beyond S18: {diagnostic['newly_recognized_poly_vs_S18']}.",
        f"S18 correct poly missed by CNN: {diagnostic['old_poly_correct_missed']}.",
        '**Research only: folds held out but historically exposed cohort, never production-promoted.**',
        '', '| Policy | Exact global | Exact poly | Fixes | Regressions |',
        '|---|---:|---:|---:|---:|']
    for k in ranking:
        v=analyses[k];m=v['metrics']
        lines.append(f"| {k} | {m['exact']*100:.4f}% | "
            f"{m['poly']['exact']*100:.4f}% | "
            f"{v['corrected']} | {v['regressed']} |")
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path)
    p.add_argument('--s18',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        require(a.features and a.s18 and a.output,'source native audio and S18 required')
        run(a)
