"""S41 six-action neural fusion: S18 KEEP or 5 truly trained polyphonic experts.

No head H9, no true-K test labels at decision time. Train one gating neural
network on OTHER pieces within each held piece's fold; producer heads S38/S40
were trained on completely different folds.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler
from scripts.train_v273_aba_recurrent import prepare
from scripts.train_v273_extended_dynamic_router import read
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require

SEED=27341
EPOCHS=12
BATCH=512
ACTION_NAMES=('KEEP_S18','raw335_HGB','raw335_ExtraTrees',
    'morph825_HGB','morph825_ExtraTrees','temporal_CNN')
THRESHOLDS=(0.,.05,.15)
DOMAINS=('source_K1to6','source_K2to6')

class FusionGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(377,192),nn.GELU(),
            nn.Dropout(.15),nn.Linear(192,96),nn.GELU(),
            nn.Linear(96,6))
    def forward(self,x):
        return self.net(x)

def sha(a):
    return hashlib.sha256(np.asarray(a,dtype='<i8').tobytes()).hexdigest()

def make_proposals(old,tabular,cnn):
    require(tabular.shape==(len(old),4,7) and
            cnn.shape==(len(old),7),'five source posteriors missing')
    proposals=np.column_stack([old,tabular.argmax(2),cnn.argmax(1)])
    require(proposals.shape==(len(old),6) and
            np.isin(proposals,np.arange(7)).all(),'invalid expert proposals')
    return proposals.astype(np.int8)

def features(x,parent,tab,cnn,scale):
    scaled=np.clip(scale.transform(x),-5,5).astype(np.float32)
    val=np.column_stack((scaled,np.eye(7,dtype=np.float32)[parent],
        tab.reshape(len(parent),28),cnn)).astype(np.float32)
    require(val.shape==(len(parent),377) and np.isfinite(val).all(),
            'unaligned six-action acoustic/posterior state')
    return val

def gated_prediction(base,proposals,scores,domain,margin):
    best=scores.argmax(1)
    gain=scores[np.arange(len(base)),best]-scores[:,0]
    eligible=(base>=1) if domain=='source_K1to6' else (base>=2)
    do=eligible&(best!=0)&(gain>margin)
    return np.where(do,proposals[np.arange(len(base)),best],base).astype(np.int8)

def selftest():
    net=FusionGate()
    with torch.no_grad():
        out=net(torch.randn(3,377))
    require(out.shape==(3,6),'not a real learned six-head neural network')
    base=np.array([0,2,3]);p=np.tile(base[:,None],(1,6))
    p[1,3]=4
    q=np.zeros((3,6),np.float32);q[1,3]=.9
    assert gated_prediction(base,p,q,'source_K2to6',.15).tolist()==[0,4,3]
    print('PASS: six-action neural expert+KEEP architecture, K-conditional masks, no H9/test truth')

def run(a):
    require(not a.output.exists(),'do not overwrite neural fusion outputs')
    torch.set_num_threads(2)
    began=time.monotonic()
    d=prepare(a)
    old=d['parent'].astype(np.int8)
    require(len(old)==59309 and
            metrics(d['y'],old)['correct']==49178 and
            metrics(d['y'],old)['poly']['correct']==2998,'S18 benchmark drift')
    s38=read(a.s38)
    s40=read(a.s40)
    require(np.array_equal(s38['global_index'],d['ids']) and
            np.array_equal(s40['global_index'],d['ids']),
            'expert native event identity drift')
    keys=list(map(str,s38['variant_ids']))
    require(keys==['raw335__hgb','raw335__extra_trees',
            'morph825__hgb','morph825__extra_trees'],
            'unexpected trained expert ordering')
    t=np.asarray(s38['poly_probs'],np.float32)
    u=np.asarray(s40['probabilities'],np.float32)
    require(np.isfinite(t).all() and np.isfinite(u).all(),
            'source acoustic class probabilities missing')
    require(np.allclose(t.sum(2),1,atol=1e-4) and
            np.allclose(u.sum(1),1,atol=1e-4),
            'producer class probabilities not normalized')
    proposal=make_proposals(old,t,u)
    a.output.mkdir(parents=True)
    (a.output/'models').mkdir()
    gates=np.full((len(old),6),np.nan,np.float32)
    trained=[]
    for piece in sorted(set(d['pieces'].tolist())):
        held=np.flatnonzero(d['pieces']==piece)
        fold=int(np.unique(d['fold'][held])[0])
        train=np.flatnonzero((d['fold']==fold)&(d['pieces']!=piece))
        require(len(train)>1000 and
                not np.intersect1d(train,held).size,
                'not enough same-fold other-piece examples')
        scale=StandardScaler().fit(d['X'][train])
        x=features(d['X'][train],old[train],t[train],u[train],scale)
        heldx=features(d['X'][held],old[held],t[held],u[held],scale)
        good=(proposal[train]==d['y'][train,None])
        # If all six routes are wrong, choose KEEP rather than an arbitrary
        # equally wrong alternative. This target uses FIT labels only.
        absent=~good.any(1)
        good[absent,0]=True
        poly=(d['y'][train]>=2)
        novel=(old[train]!=d['y'][train]) & (
            (proposal[train,1:]==d['y'][train,None]).any(1))
        weight=np.where(poly,2.,1.)*np.where(novel,4.,1.)
        weight=weight.astype(np.float32)
        data=TensorDataset(torch.from_numpy(x),
            torch.from_numpy(good.astype(np.float32)),
            torch.from_numpy(weight))
        torch.manual_seed(SEED+int(fold))
        model=FusionGate()
        optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        dl=DataLoader(data,batch_size=BATCH,shuffle=True,num_workers=0,
           generator=torch.Generator().manual_seed(SEED+fold))
        losses=[]
        for epoch in range(EPOCHS):
            model.train();loss_total=0.;count=0
            for xbatch,mask,w in dl:
                optim.zero_grad(set_to_none=True)
                pp=torch.softmax(model(xbatch),dim=1)
                chosen=(pp*mask).sum(dim=1)
                loss=-(w*torch.log(chosen.clamp_min(1e-8))).sum()/w.sum()
                loss.backward()
                optim.step()
                loss_total+=float(loss.detach())*len(w)
                count+=len(w)
            losses.append(round(loss_total/max(count,1),5))
        model.eval()
        pp=[]
        with torch.no_grad():
            for start in range(0,len(held),512):
                pp.append(torch.softmax(model(torch.from_numpy(
                        heldx[start:start+512])),dim=1).numpy())
        gates[held]=np.vstack(pp)
        filename=a.output/'models'/f'{piece}__fold{fold}__sixhead_fusion.pt'
        torch.save(dict(model=model.state_dict(),
            scaler_mean=scale.mean_.astype(np.float32),
            scaler_scale=scale.scale_.astype(np.float32),
            held_piece=str(piece),fold=int(fold),
            fit_native_ids=d['ids'][train],
            hold_native_ids=d['ids'][held],
            training_novel_count=int(novel.sum()),
            training_held_poly_count=int(poly.sum()),
            fit_sha256=sha(d['ids'][train]),
            held_sha256=sha(d['ids'][held]),
            losses=losses),filename)
        trained.append(dict(file=str(filename.relative_to(a.output)),
            fold=int(fold),piece=str(piece),
            fit_rows=int(len(train)),held_rows=int(len(held)),
            fit_ids_sha256=sha(d['ids'][train]),
            held_ids_sha256=sha(d['ids'][held]),
            train_cases_with_correct_new_head=int(novel.sum()),
            final_training_loss=losses[-1]))
        print(json.dumps(dict(piece=str(piece),fold=int(fold),
            trained=len(trained),hold=int(len(held)),
            loss=losses[-1],elapsed=round(time.monotonic()-began,1))),
            flush=True)
    require(np.isfinite(gates).all() and
            np.allclose(gates.sum(1),1,atol=1e-4),
            'some native actions not evaluated')
    out={'series18_parent':old.copy()}
    for domain in DOMAINS:
        for margin in THRESHOLDS:
            key=f'series41__neural_fusion__{domain}__margin{margin:g}'
            out[key]=gated_prediction(old,proposal,gates,domain,margin)
    audits={}
    for key,vec in out.items():
        ch=vec!=old
        fix=(old!=d['y'])&(vec==d['y'])
        reg=(old==d['y'])&(vec!=d['y'])
        neutral=(old!=d['y'])&(vec!=d['y'])&ch
        audits[key]=dict(metrics=metrics(d['y'],vec),
            versus_S18=paired(d['y'],old,vec),
            corrections=int(fix.sum()),regressions=int(reg.sum()),
            neutral=int(neutral.sum()),
            true_K={str(k):dict(fixes=int(((d['y']==k)&fix).sum()),
                 breaks=int(((d['y']==k)&reg).sum())) for k in range(7)},
            folds={str(f):dict(fixes=int(((d['fold']==f)&fix).sum()),
                 breaks=int(((d['fold']==f)&reg).sum())) for f in FOLDS})
    rank=sorted(audits,key=lambda k:(audits[k]['metrics']['correct'],
          audits[k]['metrics']['poly']['correct']),reverse=True)
    safe=[k for k in out if k.startswith('series41__')
         and audits[k]['corrections']>0 and audits[k]['regressions']==0
         and audits[k]['metrics']['poly']['correct']>=2998]
    counts={key:int((gates.argmax(1)==i).sum())
           for i,key in enumerate(ACTION_NAMES)}
    meta=dict(status='completed',learned_six_action_head=True,
        all_train_same_fold_other_pieces=True,
        acoustic_producers_excluded_test_fold=True,
        no_H9_input_no_H9_action=True,H8_unavailable_not_simulated=True,
        historic_parent_and_dataset_already_exposed=True,
        no_truly_new_music_validation=True,no_production_promotion=True,
        models=trained,models_count=len(trained),
        action_catalog=list(ACTION_NAMES),selected_argmax_counts=counts,
        all_policies=audits,best_global=rank[0],
        strict_zero_loss_candidates=safe,
        elapsed_seconds=round(time.monotonic()-began,1))
    (a.output/'report.json').write_text(json.dumps(meta,sort_keys=True,indent=2)+'\n')
    np.savez_compressed(a.output/'predictions.npz',
        global_index=d['ids'],true_K=d['y'],fold=d['fold'],
        variant_ids=np.asarray(list(out)),
        predictions=np.column_stack(list(out.values())))
    np.savez_compressed(a.output/'action_probs.npz',
        global_index=d['ids'],action_probs=gates,
        candidate_K=proposal,
        action_names=np.asarray(ACTION_NAMES))
    lines=['# S41 trained six-head dynamic fusion (one S18 keep + five real specialists)',
        '', '19 trained piece-held neural arbiters, acoustic producer fit excludes the entire evaluated fold.',
        '**Already explored development cohort: do not claim independent new-music validation.**',
        '| Variant | Global exact | Poly exact | New fixes | New regressions |',
        '|---|---:|---:|---:|---:|']
    for k in rank:
        v=audits[k];m=v['metrics']
        lines.append(f"| {k} | {m['exact']*100:.4f}% | "
           f"{m['poly']['exact']*100:.4f}% | "
           f"{v['corrections']} | {v['regressions']} |")
    lines+=['',f'Positive no-loss candidates: {len(safe)}',
        'H9 excluded; true K never in test-time action input.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path)
    p.add_argument('--s18',type=Path)
    p.add_argument('--s38',type=Path)
    p.add_argument('--s40',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        require(a.features and a.s18 and a.s38 and
            a.s40 and a.output,'all genuine trained expert inputs mandatory')
        run(a)
