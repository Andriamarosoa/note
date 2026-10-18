"""S707 CAL-only adaptive refinement of frozen S705 or S706 checkpoints.

The first CAL results selected the largest tested attack threshold (.60).
This loop expands it to .90, refines release thresholds, then tests causal
S675 integration on shortlisted complete chains. HELD/DEV never choose.
"""
import argparse
from dataclasses import asdict,replace
import gc
import itertools
import json
from pathlib import Path
import tempfile
import zipfile
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from causal_note.strict_live import SR
from causal_note.s704_joint_heads import JointPolicy,decode_joint
from causal_note.s705_temporal_heads import predict_track
from causal_note.s706_sharp_spectra import sharp_track
from causal_note.s707_causal_release import prepare_hazards,render_release
from scripts.s613_prepare_strict_live import labels,download,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s672_causal_offset_timing_grid import raw_guitar_audio
from scripts.s704_train_joint_loops import record,save,short,report_split
from scripts.s704_joint_metrics import metric,aggregate,target_rank,SCORES
from scripts.s705_train_temporal_loops import setup,load_model


def key(policy):
    p=asdict(policy)
    return '_'.join(f'{k}{v}' for k,v in p.items())


INITIAL={key(p):p for p in [JointPolicy(onset=on,sustain_on=active,sustain_off=.30,
    release=.85,count_confidence=count,blend=blend) for on,active,count,blend in itertools.product(
        (.60,.70,.80,.90),(.30,.50,.70),(1.1,.70),(.70,1.))]}
RELEASES=[dict(gate=g,persistence=p,grace=d) for g,p,d in itertools.product((.40,.60,.80),(1,2),(0,12,24))]


def evaluate_policy(item,model_name,policy):
    p,k=item['predictions'][model_name]
    post=policy.blend*p+(1-policy.blend)*item['native_post']
    return decode_joint(item['decision_ends'],post,k,policy,item['samples'])


def source(spec,archive,heads,selection,offset,models,family):
    item=record(spec,archive,heads,selection,offset)
    x=item.pop('x');item.pop('cx')
    item['native_features']=x[:,:,:126].copy()
    if family==706:
        audio,sha=raw_guitar_audio(spec,archive)
        assert sha==item['sha256']
        x=np.concatenate((x,sharp_track(audio,item['decision_ends'])),-1)
    item['predictions']={name:predict_track(model,x) for name,model in models.items()}
    return item


def score_grid(paths,model_name,policies):
    rows={n:[] for n in policies}
    for i,path in enumerate(paths):
        item=joblib.load(path)
        for name,policy in policies.items():
            events=evaluate_policy(item,model_name,policy)
            rows[name].append(metric(events,item['truth'],item['ends']))
        print('S707 CAL GRID',model_name,i+1,len(paths),len(policies),flush=True)
    return {n:aggregate(v) for n,v in rows.items()}


def run(args):
    from scripts.s699_player_held_full_train_K import expanded_train_specs
    if args.out.exists():raise FileExistsError('Immutable results directory')
    args.out.mkdir(parents=True)
    previous=json.loads((args.checkpoints/'report.json').read_text())
    model_paths=sorted(args.checkpoints.glob(f's{args.family}_epoch*.pt'))
    assert len(model_paths)>=1
    models={p.name:load_model(p)[0] for p in model_paths}
    model_hashes={p.name:digest(p.read_bytes()) for p in model_paths}
    report=dict(schema=f'S707_FROZEN_S{args.family}_CAL_REFINEMENT',promoted=False,
        official_LIVE_changed=False,TEST_opened=False,HELD_previously_exposed=True,
        DEV_previously_exposed=True,source_family=args.family,checkpoint_hashes=model_hashes,
        cal_compositions=previous['cal_compositions'],held_compositions=previous['held_compositions'],
        initial_policies={n:asdict(p) for n,p in INITIAL.items()},release_policies=RELEASES,
        thresholds={s:.80 for s in SCORES},
        rationale='Earlier CAL optimum lay at onset .60 grid boundary. Expand to .90 and refine '
            'causal offsets on CAL only. Known native release plus grace is a forward timer, '
            'not a retrospectively chosen timestamp. Emitted polyphony always <=6.',rounds=[])
    best_value=None;best_choice=None
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'data').mkdir();(root/'cal').mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for srcname,filename in (('GuitarSet_audio','GuitarSet_audio.zip'),('GuitarSet_annotations','GuitarSet_annotations.zip')):
            src=protocol['guitar_archives'][srcname]
            download(src['url'],root/'data'/filename,src['size'],src['sha256'])
        originals=[s for s in protocol['sources'] if s['corpus']=='GuitarSet' and s['split']=='train']
        dev=[s for s in protocol['sources'] if s['corpus']=='GuitarSet' and s['split']=='dev']
        with zipfile.ZipFile(root/'data/GuitarSet_audio.zip') as archive,zipfile.ZipFile(root/'data/GuitarSet_annotations.zip') as ann:
            expanded=expanded_train_specs(originals,archive,ann)
            cal=[s for s in expanded if s['composition'] in previous['cal_compositions']]
            held=[s for s in expanded if s['composition'] in previous['held_compositions']]
            paths=[];baseline=[]
            for i,spec in enumerate(cal):
                item=source(spec,archive,heads,selection,offset,models,args.family)
                item['truth']=real_notes(labels(spec,root)['notes'],item['samples']/SR)
                baseline.append(dict(name=spec['name'],metrics={n:metric(e,item['truth'],item['ends']) for n,e in item['baseline'].items()}))
                path=root/'cal'/f'{i}.joblib';joblib.dump(item,path,compress=0);paths.append(path)
                print('S707 PREPARE CAL',i+1,len(cal),spec['name'],flush=True)
            report['CAL_baseline']=report_split(baseline,('native','native_s675_causal'))
            for model_name in models:
                policies=dict(INITIAL)
                initial=score_grid(paths,model_name,INITIAL)
                top_on=sorted(initial,key=lambda n:initial[n]['onset_f1'],reverse=True)[:4]
                top_joint=sorted(initial,key=lambda n:target_rank(initial[n]),reverse=True)[:2]
                selected=list(dict.fromkeys(top_on+top_joint))
                refined={}
                for n in selected:
                    for off,release,persist in itertools.product((.10,.20,.30,.50,.70),(.60,.80,.95),(1,2)):
                        p=replace(INITIAL[n],sustain_off=off,release=release,release_steps=persist)
                        refined[key(p)]=p
                policies.update(refined)
                scores=dict(initial,**score_grid(paths,model_name,refined))
                choices={n:dict(model_file=model_name,policy=asdict(p),release=None) for n,p in policies.items()}
                best_native=sorted(scores,key=lambda n:target_rank(scores[n]),reverse=True)[:4]
                hazard_rows={}
                for n in best_native:
                    for h in RELEASES:
                        name=n+'_S675_'+str(h)
                        hazard_rows[name]=[]
                        choices[name]=dict(model_file=model_name,policy=asdict(policies[n]),release=h)
                for i,path in enumerate(paths):
                    item=joblib.load(path)
                    for n in best_native:
                        events=evaluate_policy(item,model_name,policies[n])
                        plan=prepare_hazards(item,events,offset)
                        for h in RELEASES:
                            name=n+'_S675_'+str(h)
                            emitted,skipped=render_release(plan,item['ends'],item['samples'],**h)
                            hazard_rows[name].append(metric(emitted,item['truth'],item['ends']))
                    print('S707 CAL HAZARD',model_name,i+1,len(paths),flush=True)
                scores.update({n:aggregate(v) for n,v in hazard_rows.items()})
                best=max(scores,key=lambda n:target_rank(scores[n]))
                report['rounds'].append(dict(model_file=model_name,best=best,scores=scores,
                    initial_choices=len(initial),refined_choices=len(refined),release_choices=len(hazard_rows)))
                value=target_rank(scores[best])
                if best_value is None or value>best_value:
                    best_value=value;best_choice=choices[best]
                    report['frozen_candidate']=dict(choice=best_choice,CAL=scores[best])
                save(args.out/'report.json',report)
                print('S707 ROUND',model_name,best,json.dumps(short(scores[best])),flush=True)
            assert best_choice is not None
            selected_model=models[best_choice['model_file']]
            selected_path=args.checkpoints/best_choice['model_file']
            (args.out/'s707_selected.pt').write_bytes(selected_path.read_bytes())
            report['checkpoint_sha256']=model_hashes[best_choice['model_file']]
            save(args.out/'selection_frozen.json',report['frozen_candidate'])
            save(args.out/'report.json',report)
            print('S707 FROZEN',json.dumps(report['frozen_candidate']),flush=True)
            models={best_choice['model_file']:selected_model};gc.collect()
            policy=JointPolicy(**best_choice['policy'])
            for phase,specs in (('HELD',held),('DEV',dev)):
                rows=[]
                with (args.out/(phase.lower()+'_events.jsonl')).open('w') as f:
                    for i,spec in enumerate(specs):
                        item=source(spec,archive,heads,selection,offset,models,args.family)
                        events=evaluate_policy(item,best_choice['model_file'],policy)
                        if best_choice['release'] is not None:
                            plan=prepare_hazards(item,events,offset)
                            events,skipped=render_release(plan,item['ends'],item['samples'],**best_choice['release'])
                        event_dict=dict(item['baseline'],candidate=events)
                        (args.out/'latest_prediction_before_labels.json').write_text(json.dumps(dict(name=spec['name'],events=event_dict)))
                        truth=real_notes(labels(spec,root)['notes'],item['samples']/SR)
                        metrics={n:metric(e,truth,item['ends']) for n,e in event_dict.items()}
                        base,trial=metrics['native_s675_causal'],metrics['candidate']
                        paired=dict(recovered=len(set(trial['matched_truth_ids'])-set(base['matched_truth_ids'])),
                            lost=len(set(base['matched_truth_ids'])-set(trial['matched_truth_ids'])),
                            lost_long=len(set(base['long_onset_truth_ids'])-set(trial['long_onset_truth_ids'])),
                            lost_correct_offsets=len(set(base['correct_offset_truth_ids'])-set(trial['correct_offset_truth_ids'])))
                        rows.append(dict(name=spec['name'],composition=spec['composition'],metrics=metrics,paired=paired))
                        f.write(json.dumps(dict(name=spec['name'],sha256=item['sha256'],samples=item['samples'],truth=truth,events=event_dict))+'\n');f.flush()
                        print('S707',phase,i+1,len(specs),spec['name'],json.dumps(short(trial)),flush=True)
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
            print('S707 FINAL',json.dumps({k:report[k] for k in ('source_family','frozen_candidate','HELD','DEV','all_targets_above_80','promoted')}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family',type=int,choices=(705,706),required=True)
    ap.add_argument('--checkpoints',type=Path,required=True)
    ap.add_argument('--offset-checkpoint',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=2):run(ap.parse_args())
