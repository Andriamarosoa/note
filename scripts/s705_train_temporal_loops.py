"""S705 causal temporal/harmonic learning, independently calibrated loops.

FIT four composition families, CAL one other, HELD four others, then exposed
DEV. Test is unopened. All metrics refer to a single actual event stream.
"""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time
import zipfile
import joblib
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from threadpoolctl import threadpool_limits
from causal_note.strict_live import SR,Selection,load_heads
from causal_note.s672_live_offset import S672TrialNoteOff
from causal_note.s704_joint_heads import JointPolicy,decode_joint
from causal_note.s705_temporal_heads import TemporalHeads,CONTEXT,predict_track,TemporalHeadStream
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s672_causal_offset_timing_grid import ROOT,MODEL
from scripts.s704_train_joint_loops import record,targets,save,POLICIES,short,report_split
from scripts.s704_joint_metrics import metric,aggregate,target_rank,SCORES

EPOCHS=(10,20,32)
TARGET_STEPS=32


def setup(work,offset_checkpoint):
    protocol=json.loads((ROOT/'protocol.json').read_text())
    frozen=json.loads((ROOT/'selection_frozen.json').read_text())
    assert digest(MODEL.read_bytes())==frozen['model_sha256']
    assert digest((ROOT/'protocol.json').read_bytes())==frozen['protocol_sha256']
    heads=load_heads(MODEL,expected_sha256=frozen['model_sha256'])
    offset=S672TrialNoteOff.load(offset_checkpoint,heads).model
    return protocol,heads,Selection(**heads.metadata['selection']),offset


def prepare(args):
    from scripts.s699_player_held_full_train_K import expanded_train_specs
    if args.work.exists() or args.out.exists():
        raise FileExistsError('Immutable experiment/cache paths already exist')
    args.work.mkdir(parents=True);args.out.mkdir(parents=True)
    for folder in ('data','fit','cal'):
        (args.work/folder).mkdir()
    protocol,heads,selection,offset=setup(args.work,args.offset_checkpoint)
    for key,name in (('GuitarSet_audio','GuitarSet_audio.zip'),('GuitarSet_annotations','GuitarSet_annotations.zip')):
        source=protocol['guitar_archives'][key]
        download(source['url'],args.work/'data'/name,source['size'],source['sha256'])
    originals=[s for s in protocol['sources'] if s['corpus']=='GuitarSet' and s['split']=='train']
    comps=sorted({s['composition'] for s in originals})
    dev=[s for s in protocol['sources'] if s['corpus']=='GuitarSet' and s['split']=='dev']
    with zipfile.ZipFile(args.work/'data/GuitarSet_audio.zip') as archive,zipfile.ZipFile(args.work/'data/GuitarSet_annotations.zip') as ann:
        expanded=expanded_train_specs(originals,archive,ann)
        fit=[s for s in expanded if s['composition'] in comps[:4]]
        cal=[s for s in expanded if s['composition']==comps[4]]
        held=[s for s in expanded if s['composition'] in comps[5:]]
        assert (len(fit),len(cal),len(held),len(dev))==(64,16,64,12)
        stat_sum=np.zeros(192,np.float64);stat_square=stat_sum.copy();stat_count=0
        proof=[]
        for phase,specs in (('fit',fit),('cal',cal)):
            for i,spec in enumerate(specs):
                item=record(spec,archive,heads,selection,offset if phase=='cal' else None)
                truth=real_notes(labels(spec,args.work)['notes'],item['samples']/SR)
                item['x']=item['x'].astype(np.float16);item.pop('cx')
                if phase=='fit':
                    y,k=targets(item['decision_ends'],truth)
                    z=item['x'][::8].astype(np.float64).reshape(-1,192)
                    stat_sum+=z.sum(0);stat_square+=(z*z).sum(0);stat_count+=len(z)
                    data=dict(x=item['x'],y=y,k=k,name=spec['name'])
                    proof.append(dict(name=spec['name'],composition=spec['composition'],sha256=item['sha256'],frames=len(y)))
                else:
                    data=dict(item,truth=truth)
                joblib.dump(data,args.work/phase/f'{i:03d}.joblib',compress=0)
                print('S705 PREPARE',phase,i+1,len(specs),spec['name'],flush=True)
        mean=stat_sum/stat_count
        scale=np.maximum(np.sqrt(np.maximum(0,stat_square/stat_count-mean*mean)),.05)
        save(args.work/'manifest.json',dict(held=held,dev=dev,mean=mean.tolist(),scale=scale.tolist()))
        save(args.out/'report.json',dict(schema='S705_CAUSAL_TEMPORAL_HARMONIC_LOOPS',
            promoted=False,official_LIVE_changed=False,TEST_opened=False,
            DEV_previously_exposed=True,HELD_previously_exposed=True,
            fit_compositions=comps[:4],cal_compositions=[comps[4]],held_compositions=comps[5:],
            native_sha256=heads.sha256,training_sources=proof,learning_budgets=EPOCHS,
            thresholds={s:.80 for s in SCORES},policies={n:asdict(p) for n,p in POLICIES.items()},
            feature_dimensions=192,temporal_context_blocks=CONTEXT,batch_hops=4,
            cache_input_quantization='float16 -> float32, same during fit and inference',
            note='New model fits separate compositions; historical native feature-head exposure remains. '
                 'One emitted stream, strict causal replay. TEST unopened; microphone performance unverified.',
            rounds=[]))


class Windows(Dataset):
    def __init__(self, paths):
        self.rows=[joblib.load(p) for p in paths]
        self.windows=[(i,s) for i,r in enumerate(self.rows) for s in range(0,len(r['x']),TARGET_STEPS)]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self,index):
        i,start=self.windows[index];r=self.rows[i]
        lo=max(0,start-CONTEXT);hi=min(len(r['x']),start+TARGET_STEPS)
        left=CONTEXT-(start-lo);size=CONTEXT+TARGET_STEPS
        x=np.zeros((size,49,192),np.float32)
        y=np.zeros((size,49,3),np.float32);k=np.zeros(size,np.int64)
        valid=np.zeros(size,np.float32);loss_mask=valid.copy()
        x[left:left+hi-lo]=r['x'][lo:hi]
        y[left:left+hi-lo]=r['y'][lo:hi];k[left:left+hi-lo]=r['k'][lo:hi]
        valid[left:left+hi-lo]=1
        loss_mask[CONTEXT:CONTEXT+hi-start]=1
        return tuple(torch.from_numpy(a) for a in (x,y,k,valid,loss_mask))


def load_model(path):
    bundle=torch.load(path,map_location='cpu',weights_only=True)
    model=TemporalHeads(bundle['model']['mean'],bundle['model']['scale'],width=bundle['width'])
    model.load_state_dict(bundle['model'])
    return model,bundle


def train(args):
    report=json.loads((args.out/'report.json').read_text())
    if report.get('CAL_all_targets_above_80'):
        print('S705 target already passed CAL; remaining training skipped',flush=True);return
    manifest=json.loads((args.work/'manifest.json').read_text())
    torch.manual_seed(705)
    state=args.work/'latest.pt'
    if state.exists():
        model,bundle=load_model(state);start=bundle['epoch']+1
    else:
        model=TemporalHeads(manifest['mean'],manifest['scale']);bundle={};start=1
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    if 'optimizer' in bundle:
        optimizer.load_state_dict(bundle['optimizer'])
    dataset=Windows(sorted((args.work/'fit').glob('*.joblib')))
    loader=DataLoader(dataset,batch_size=8,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(705+args.epoch))
    positive=torch.tensor([10.,2.,8.])
    history=[]
    for epoch in range(start,args.epoch+1):
        begin=time.perf_counter();model.train();losses=[]
        lr=.001 if epoch<=12 else .0005 if epoch<=24 else .00025
        for group in optimizer.param_groups:group['lr']=lr
        for x,y,k,valid,mask in loader:
            optimizer.zero_grad(set_to_none=True)
            logits,k_logits=model(x,valid)
            pitch_loss=F.binary_cross_entropy_with_logits(logits,y,pos_weight=positive,reduction='none').mean((-1,-2))
            count_loss=F.cross_entropy(k_logits.flatten(0,1),k.flatten(),reduction='none').reshape_as(k)
            # The count target supervises training only; decoder sees predicted K.
            consistency=(logits[:,:,:,1].sigmoid().sum(-1)-k.float()).square()
            loss=((pitch_loss+.25*count_loss+.008*consistency)*mask).sum()/mask.sum()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
            optimizer.step();losses.append(float(loss.detach()))
        row=dict(epoch=epoch,loss=float(np.mean(losses)),seconds=round(time.perf_counter()-begin,2),lr=lr)
        history.append(row);print('S705 EPOCH',json.dumps(row),flush=True)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,width=model.width),state)
    save(args.out/f'training_to_{args.epoch}.json',history)
    torch.save(dict(model=model.state_dict(),epoch=args.epoch,width=model.width),args.out/f's705_epoch{args.epoch}.pt')


def calibrate(args):
    report=json.loads((args.out/'report.json').read_text())
    if report.get('CAL_all_targets_above_80'):
        return
    model,bundle=load_model(args.out/f's705_epoch{args.epoch}.pt')
    scores={n:[] for n in POLICIES};base=[]
    paths=sorted((args.work/'cal').glob('*.joblib'))
    for i,path in enumerate(paths):
        item=joblib.load(path);p,k=predict_track(model,item['x'])
        base.append(dict(name=item['spec']['name'],metrics={n:metric(e,item['truth'],item['ends']) for n,e in item['baseline'].items()}))
        for n,policy in POLICIES.items():
            post=policy.blend*p+(1-policy.blend)*item['native_post']
            events=decode_joint(item['decision_ends'],post,k,policy,item['samples'])
            scores[n].append(metric(events,item['truth'],item['ends']))
        print('S705 CAL',args.epoch,i+1,len(paths),flush=True)
    scores={n:aggregate(rows) for n,rows in scores.items()}
    best=max(scores,key=lambda n:target_rank(scores[n]))
    row=dict(epoch=args.epoch,best=best,scores=scores,baseline=report_split(base,('native','native_s675_causal')))
    report['rounds'].append(row)
    if 'frozen_candidate' not in report or target_rank(scores[best])>target_rank(report['frozen_candidate']['CAL']):
        report['frozen_candidate']=dict(epoch=args.epoch,policy=best,CAL=scores[best])
        torch.save(dict(bundle,policy=asdict(POLICIES[best])),args.out/'s705_selected.pt')
        save(args.out/'selection_frozen.json',report['frozen_candidate'])
    report['CAL_all_targets_above_80']=scores[best]['all_targets_above_80']
    save(args.out/'report.json',report);save(args.out/f'round_{args.epoch}.json',row)
    print('S705 ROUND',args.epoch,best,json.dumps(short(scores[best])),flush=True)


def evaluate(args):
    report=json.loads((args.out/'report.json').read_text())
    protocol,heads,selection,offset=setup(args.work,args.offset_checkpoint)
    manifest=json.loads((args.work/'manifest.json').read_text())
    model,bundle=load_model(args.out/'s705_selected.pt');policy=JointPolicy(**bundle['policy'])
    report['checkpoint_sha256']=digest((args.out/'s705_selected.pt').read_bytes())
    # Report head-only incremental timing separately from PCM/native frontend.
    item=joblib.load(sorted((args.work/'cal').glob('*.joblib'))[0]);stream=TemporalHeadStream(model)
    elapsed=[]
    for x in item['x'][:160]:
        at=time.perf_counter();stream.step(x);elapsed.append(1000*(time.perf_counter()-at))
    report['head_only_incremental_ms_p50_p95']=np.percentile(elapsed[10:],[50,95]).tolist()
    del item
    with zipfile.ZipFile(args.work/'data/GuitarSet_audio.zip') as archive:
        for phase in ('HELD','DEV'):
            specs=manifest[phase.lower()];rows=[]
            with (args.out/(phase.lower()+'_events.jsonl')).open('w') as f:
                for i,spec in enumerate(specs):
                    item=record(spec,archive,heads,selection,offset)
                    p,k=predict_track(model,item['x'])
                    post=policy.blend*p+(1-policy.blend)*item['native_post']
                    events=decode_joint(item['decision_ends'],post,k,policy,item['samples'])
                    event_dict=dict(item['baseline'],candidate=events)
                    (args.out/'latest_prediction_before_labels.json').write_text(json.dumps(dict(name=spec['name'],events=event_dict)))
                    truth=real_notes(labels(spec,args.work)['notes'],item['samples']/SR)
                    metrics={n:metric(e,truth,item['ends']) for n,e in event_dict.items()}
                    base,trial=metrics['native_s675_causal'],metrics['candidate']
                    paired=dict(recovered=len(set(trial['matched_truth_ids'])-set(base['matched_truth_ids'])),
                        lost=len(set(base['matched_truth_ids'])-set(trial['matched_truth_ids'])),
                        lost_long=len(set(base['long_onset_truth_ids'])-set(trial['long_onset_truth_ids'])),
                        lost_correct_offsets=len(set(base['correct_offset_truth_ids'])-set(trial['correct_offset_truth_ids'])))
                    rows.append(dict(name=spec['name'],composition=spec['composition'],metrics=metrics,paired=paired))
                    f.write(json.dumps(dict(name=spec['name'],sha256=item['sha256'],samples=item['samples'],truth=truth,events=event_dict))+'\n');f.flush()
                    print('S705',phase,i+1,len(specs),spec['name'],json.dumps(short(metrics['candidate'])),flush=True)
            report[phase]=report_split(rows,('native','native_s675_causal','candidate'))
            report[phase]['paired']={k:sum(r['paired'][k] for r in rows) for k in rows[0]['paired']}
            base,trial=report[phase]['native_s675_causal'],report[phase]['candidate']
            report[phase]['non_regression']=bool(all(trial[s]>=base[s] for s in SCORES)
                and trial['tp']>=base['tp'] and report[phase]['paired']['lost_long']==0
                and report[phase]['paired']['lost_correct_offsets']==0)
            save(args.out/(phase.lower()+'_audit.json'),rows);save(args.out/'report.json',report)
    report['all_targets_above_80']=all(report[p]['candidate']['all_targets_above_80'] for p in ('HELD','DEV'))
    report['non_regression_passed']=all(report[p]['non_regression'] for p in ('HELD','DEV'))
    save(args.out/'report.json',report)
    print('S705 FINAL',json.dumps({k:report[k] for k in ('frozen_candidate','HELD','DEV','all_targets_above_80','promoted')}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('prepare','train','calibrate','evaluate'))
    ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--offset-checkpoint',type=Path,required=True)
    ap.add_argument('--epoch',type=int,default=10)
    args=ap.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=2):
        globals()[args.mode](args)
