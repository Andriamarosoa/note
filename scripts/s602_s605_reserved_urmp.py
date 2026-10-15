"""S602--S605: frozen decoder/head on the five composition-held-out sources.

The audio job cannot access test annotations through this entry point. A
separate score job requires byte-identical predictions and freeze metadata.
No tuning, fitting, source replacement, Live changes or S80 promotion.
"""
from __future__ import annotations
import argparse
import collections
import importlib.metadata
from pathlib import Path
import numpy as np
from scripts.s585_attack_sustain_restrike import need, sha, save, metrics
from scripts.s592_s597_urmp_prospective import read, fetch, decode_prepared, digest, parse_notes, truth_groups
from scripts.s600_bp_native_development import MANIFEST, MANIFEST_SHA, decode
from scripts.s601_bp_candidate_gate import candidates, feature_matrix, emit, counts
from scripts.s591_score_state_note_predictions import match_ledger
from scripts.s277_original_freeze_dataset3_native_prelabel import pcm

CONTRACT=Path('analysis/v273_s602_selected_candidate_freeze.json')
POLICIES=('BasicPitch_default','S600_global_decoder','S601_learned_candidate_gate')


def contract():
    c=read(CONTRACT);m=read(MANIFEST)
    need(sha(MANIFEST)==MANIFEST_SHA==c['manifest_sha256'],'Immutable reserved source list')
    need(c['primary_policy'] in POLICIES and c['test_audio_not_read'] and c['test_annotations_not_read'], 'Selection must precede test audio')
    need(c['test_source_ids']==[s['source_id'] for s in m['test']], 'All five preselected sources required')
    need(not(set(c['train_pieces'])&{s['piece'] for s in m['test']}),'Composition leakage')
    for name,h in c['script_hashes'].items():need(sha(name)==h,'Frozen inference implementation changed: '+name)
    need(sha(c['model_path'])==c['model_sha256'],'Frozen learned head changed')
    return c,m


def freeze(out):
    import joblib
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, run_inference
    c,m=contract();need(not out.exists(),'Immutable prospective freeze output')
    for package,version in c['runtime_versions'].items():
        need(importlib.metadata.version(package)==version,'Frozen runtime version: '+package)
    weights={str(p.relative_to(ICASSP_2022_MODEL_PATH)):sha(p)
        for p in sorted(Path(ICASSP_2022_MODEL_PATH).rglob('*')) if p.is_file()}
    need(weights==c['basic_pitch_weight_hashes'],'Pretrained neural weights changed')
    out.mkdir(parents=True);private=out/'private_audio';private.mkdir()
    bp=Model(ICASSP_2022_MODEL_PATH);head=joblib.load(c['model_path'])
    pred={name:{} for name in POLICIES};proof=[]
    for spec in m['test']:
        sid=spec['source_id'];data=fetch(spec['audio'])
        signal=decode_prepared(data);clip=signal[5*44100:29*44100]
        path=private/(sid+'.wav');path.write_bytes(pcm(clip))
        raw=run_inference(str(path),bp)
        need(all(np.isfinite(v).all() for v in raw.values()),'Finite prospective network outputs')
        np.savez_compressed(out/(sid+'.raw.npz'),**raw)
        pred[POLICIES[0]][sid]=decode(raw)
        pred[POLICIES[1]][sid]=decode(raw,**c['global_decoder'])
        ev=candidates(raw);X,names=feature_matrix(raw,ev)
        need(names==c['feature_names'],'Frozen candidate feature schema changed')
        probability=head.predict_proba(X)[:,1]
        pred[POLICIES[2]][sid]=emit(ev,probability,**c['learned_decoder'])
        save(out/(sid+'.candidates.json'),ev)
        np.savez_compressed(out/(sid+'.head.npz'),probability=probability)
        proof.append(dict(source_id=sid,folder=spec['folder'],audio_sha256=digest(data),
            prepared_bytes=len(data),source_duration=len(signal)/44100,clip_sha256=sha(path),
            raw_sha256=sha(out/(sid+'.raw.npz')),candidate_sha256=sha(out/(sid+'.candidates.json')),
            probability_sha256=sha(out/(sid+'.head.npz')),
            counts={name:sum(len(e['notes']) for e in pred[name][sid]) for name in POLICIES}))
        print('S603 TEST AUDIO ONLY',spec['folder'],proof[-1]['counts'],flush=True)
        path.unlink()
    private.rmdir()
    save(out/'predictions.json',pred)
    save(out/'report.json',dict(status='S604_ALL_FIVE_TEST_PREDICTIONS_FROZEN_BEFORE_ANNOTATIONS',
        annotations_read=False,fit_or_threshold_selection=False,primary_policy=c['primary_policy'],
        policies=list(POLICIES),manifest_sha256=sha(MANIFEST),contract_sha256=sha(CONTRACT),
        model_sha256=c['model_sha256'],prediction_sha256=sha(out/'predictions.json'),
        audio_proof=proof,offset=5,seconds=24,total_seconds=120,
        runtime_versions=c['runtime_versions'],no_promotion=True))
    print('S604 PRELABEL FREEZE',sha(out/'predictions.json'),flush=True)


def score(frozen,out):
    c,m=contract();need(not out.exists(),'First held-out result is immutable')
    r=read(frozen/'report.json');pred=read(frozen/'predictions.json')
    need(r['status']=='S604_ALL_FIVE_TEST_PREDICTIONS_FROZEN_BEFORE_ANNOTATIONS'
         and not r['annotations_read'] and r['prediction_sha256']==sha(frozen/'predictions.json')
         and r['contract_sha256']==sha(CONTRACT) and r['manifest_sha256']==sha(MANIFEST)
         and list(r['policies'])==list(POLICIES),'Annotation embargo requires the exact earlier freeze')
    ids={s['source_id'] for s in m['test']}
    need(set(pred)==set(POLICIES) and all(set(p)==ids for p in pred.values()),'Every policy scores every source')
    out.mkdir(parents=True);truth={};annotations={};provenance=[];scope={};rapid=[]
    for spec in m['test']:
        sid=spec['source_id'];events=[]
        for index,item in enumerate(spec['annotations']):
            data=fetch(item,annotation=True);ins=Path(item['path']).name.split('_')[2]
            events.extend(parse_notes(data,ins,index+1))
            provenance.append(dict(source_id=sid,path=item['path'],sha256=digest(data),bytes=len(data)))
        annotations[sid]=events;truth[sid]=truth_groups(events)
        scope[sid]=dict(folder=spec['folder'],family=spec['family'],instruments=spec['instruments'],
            raw_note_onsets_in_excerpt=sum(0<=x['time']<24 for x in events),
            out_of_range_notes=[x for x in events if 0<=x['time']<24 and not 40<=x['pitch']<=88],
            exact_unison_duplicates_removed=sum(0<=x['time']<24 and 40<=x['pitch']<=88 for x in events)-len(truth[sid]),
            scored_true_notes=len(truth[sid]))
        for track in {x['track'] for x in events}:
            for pitch in range(40,89):
                q=sorted(x['time'] for x in events if x['track']==track and x['pitch']==pitch and 0<=x['time']<24)
                for a,b in zip(q,q[1:]):
                    if 0<b-a<.1:rapid.append(dict(source_id=sid,track=track,pitch=pitch,first=a,second=b,delta_ms=(b-a)*1000))
    scores=metrics(pred,truth,{sid:dict(composer=s['folder']) for sid,s in scope.items()})
    independent_checks=0
    for name in POLICIES:
        for sid in ids:
            for tol in (.03,.05,.08):
                q=counts(pred[name][sid],truth[sid],tol)
                need(all(q[k]==scores[name]['per_song'][sid]['scores'][str(int(tol*1000))][k] for k in ('TP','FP','FN')),
                     'Independent maximum-cardinality count differs')
                independent_checks+=1
    audit=[];comparison={};misses=[];coverage={}
    for name in POLICIES[1:]:
        summary=collections.Counter()
        for sid in ids:
            old,oldfp=match_ledger(pred[POLICIES[0]][sid],truth[sid]);new,newfp=match_ledger(pred[name][sid],truth[sid])
            for change,values in (('gained',new-old),('lost',old-new)):
                summary[change]+=len(values)
                audit.extend(dict(policy=name,source_id=sid,folder=scope[sid]['folder'],change=change,
                    pitch=p,truth_index=j,truth_time=t) for p,j,t in sorted(values))
            oc=collections.Counter((x['pitch'],round(x['time'],9)) for x in oldfp)
            nc=collections.Counter((x['pitch'],round(x['time'],9)) for x in newfp)
            summary['FP_removed']+=sum((oc-nc).values());summary['FP_added']+=sum((nc-oc).values())
            summary['FP_common']+=sum((oc&nc).values())
        summary['TP_delta']=scores[name]['fullstream']['80']['TP']-scores[POLICIES[0]]['fullstream']['80']['TP']
        summary['FP_delta']=scores[name]['fullstream']['80']['FP']-scores[POLICIES[0]]['fullstream']['80']['FP']
        need(summary['gained']-summary['lost']==summary['TP_delta'] and summary['FP_added']-summary['FP_removed']==summary['FP_delta'], 'Paired audit reconciliation')
        comparison[name]=dict(summary)
    for sid in ids:
        proof=next(x for x in r['audio_proof'] if x['source_id']==sid)
        cp=frozen/(sid+'.candidates.json');pp=frozen/(sid+'.head.npz')
        need(sha(cp)==proof['candidate_sha256'] and sha(pp)==proof['probability_sha256'],'Frozen candidate evidence changed')
        ev=read(cp);prob=np.load(pp,allow_pickle=False)['probability']
        coverage[sid]=counts([dict(time=x['time'],notes=[x['pitch']]) for x in ev],truth[sid])
        matched,_=match_ledger(pred[c['primary_policy']][sid],truth[sid])
        for pitch in range(40,89):
            for j,t in enumerate(sorted(x['time'] for x in truth[sid] if pitch in x['notes'])):
                if (pitch,j,t) in matched:continue
                near=[i for i,x in enumerate(ev) if x['pitch']==pitch and abs(x['time']+c['learned_decoder']['shift']-t)<=.08]
                misses.append(dict(source_id=sid,pitch=pitch,truth_time=t,
                    maximum_candidate_probability=max((float(prob[i]) for i in near),default=None),
                    eligible_candidates=len(near)))
    for event in rapid:
        event['both_notes_matched_by_policy']={}
        for name in POLICIES:
            matched,_=match_ledger(pred[name][event['source_id']],truth[event['source_id']])
            event['both_notes_matched_by_policy'][name]=[any(p==event['pitch'] and abs(t-v)<1e-9 for p,j,t in matched)
                for v in (event['first'],event['second'])]
    outputs={'truth_after_freeze.json':truth,'annotations_after_freeze.json':annotations,
        'individual_note_audit.json':audit,'primary_FN_diagnosis.json':misses}
    for name,value in outputs.items():save(out/name,value)
    primary=scores[c['primary_policy']]['fullstream']['80']
    save(out/'report.json',dict(status='S605_FIRST_RESERVED_COMPOSITION_TEST_NO_POSTTEST_TUNING',
        primary_policy=c['primary_policy'],objective_F1_080_reached=primary['F1']>=.8,
        primary_F1=primary['F1'],scores=scores,comparison_to_default_BP=comparison,
        source_scope=scope,true_notes=sum(len(x) for x in truth.values()),
        source_count=5,distinct_reserved_pieces=5,total_seconds=120,
        no_claim_of_unseen_pretraining_or_composers=True,prepared_audio_rate=16000,
        independent_matching_checks=independent_checks,candidate_coverage=coverage,
        genuine_same_instrument_rapid_repeats=rapid,annotation_provenance=provenance,
        freeze_report_sha256=sha(frozen/'report.json'),prediction_sha256=sha(frozen/'predictions.json'),
        contract_sha256=sha(CONTRACT),audit_hashes={name:sha(out/name) for name in outputs},
        full_S80_ExactK_NOT_REPLAYED=True,Live_NOT_modified=True,NoteOff_NOT_validated=True,no_promotion=True))
    for name in POLICIES:print('S605 FIRST RESERVED TEST',name,scores[name]['fullstream']['80'],flush=True)
    print('S605 PRIMARY OBJECTIVE REACHED',primary['F1']>=.8,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','score'))
    p.add_argument('--frozen',type=Path,default=Path('model/S604'));p.add_argument('--out',type=Path,default=Path('model/S605'))
    a=p.parse_args()
    if a.stage=='freeze':freeze(a.frozen)
    else:score(a.frozen,a.out)
