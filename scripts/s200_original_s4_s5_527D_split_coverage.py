"""S200 authentic original S4 owned / S5 527D tree split coverage census.

Find exact feature indices used by pinned trained trees, shared active
suffix coordinates, and importance by S4/5 feature blocks; no matrices
inferred, labels read, or model predictions changed. Original training
features cannot be inferred just from active split indices.
"""
from __future__ import annotations
import argparse,hashlib,json,collections
from pathlib import Path
import joblib,numpy as np
FOLDS=(0,1,2,4)
BLOCKS=(('harmonic58',0,58),('geometry46',58,104),('freeze_onehot7',104,111),
        ('owned_zone_native104',111,215),('owned_zone_dilated104',215,319),
        ('owned_zone_before104',319,423),('owned_zone_after104',423,527))
def need(x,msg):
    if not x:raise ValueError(msg)
def checkpoint(file,sha=None):
    if sha:
        need(hashlib.sha256(file.read_bytes()).hexdigest()==sha,
             'S5 original pinned model sha drift '+str(file))
    item=joblib.load(file)
    model=item['model']
    need(model.n_features_in_ in (111,527) and np.array_equal(model.classes_,np.arange(7)),
         'original checkpoint model/class mismatch')
    return item,model
def active(model):
    counts=collections.Counter()
    for epoch in model._predictors:
        for tree in epoch:
            arr=tree.nodes
            counts.update(map(int,arr[arr['is_leaf']==0]['feature_idx']))
    need(counts and all(0<=x<model.n_features_in_ for x in counts),
         'model trained split coverage invalid')
    return counts
def scoped(counter):
    total=sum(counter.values())
    blocks=[]
    for name,lo,hi in BLOCKS:
        c={k:v for k,v in counter.items() if lo<=k<hi}
        blocks.append({'block':name,'range_half_open':[lo,hi],
                       'unique_active_columns':len(c),
                       'active_split_count':sum(c.values()),
                       'split_share':sum(c.values())/total})
    return blocks
def run(root,out):
    need(not out.exists(),'immutable pinned checkpoint audit')
    details=[];s5_counts=collections.Counter()
    s4_counts=collections.Counter()
    s5_folds=[];s4_folds=[];hashes=[];missing_original_s4_owned=[]
    for fold in FOLDS:
        p=root/'series5'/'models'/f'fold{fold}'
        manifest=json.loads((p/'report.json').read_text())
        saved,model=checkpoint(p/'model.joblib',manifest['model_sha256'])
        need(model.n_features_in_==527,'wrong S5 input')
        c=active(model)
        s5_counts.update(c)
        s5_folds.append(set(c))
        hashes.append({'fold':fold,'S5_model_sha256':manifest['model_sha256'],
                       'S5_training_design_sha256':str(saved.get('designs_sha256',''))})
        details.append({'stage':'S5','fold':fold,'unique_active':len(c),
                        'total_splits':sum(c.values()),'blocks':scoped(c)})
        for poly in (1,2):
            path=root/'series4'/'models'/f'fold{fold}__owned__poly{poly}.joblib'
            if not path.is_file():
                missing_original_s4_owned.append(str(path))
                continue
            item,owned=checkpoint(path)
            need(owned.n_features_in_==527,'wrong S4 owned training schema')
            current=active(owned)
            s4_counts.update(current);s4_folds.append(set(current))
            details.append({'stage':'S4_owned','fold':fold,'poly':poly,
                            'unique_active':len(current),'total_splits':sum(current.values()),
                            'blocks':scoped(current)})
    need(len(s4_folds)>=4,'too few verified S4 owned checkpoints')
    shared=set.intersection(*s5_folds)
    s5=set(s5_counts);s4=set(s4_counts)
    total=len(s5_counts)
    suffix=set(range(111,527))
    missing_all=sorted(suffix-(s5|s4))
    report={'status':'S200_ORIGINAL_527D_S4_OWNED_S5_TREE_ACTIVE_FEATURE_CENSUS',
       'original_S5_model_count':4,'original_S4_owned_model_count':len(s4_folds),
       'missing_original_S4_owned_models':missing_original_s4_owned,
       'S5_total_splits':sum(s5_counts.values()),
       'S4_owned_total_splits':sum(s4_counts.values()),
       'S5_active_527D_unique_columns':len(s5),
       'S4_active_527D_unique_columns':len(s4),
       'S5_owned_suffix_416_active_columns':len(s5&suffix),
       'S4_owned_suffix_416_active_columns':len(s4&suffix),
       'S5_S4_suffix_union_active':len((s5|s4)&suffix),
       'S5_S4_suffix_never_used_416_count':len(missing_all),
       'S5_S4_suffix_never_used_indices':missing_all,
       'S5_common_active_across_four_models':len(shared),
       'S5_active_527_feature_indices':sorted(s5),
       'S4_owned_active_527_feature_indices':sorted(s4),
       'S5_ranked_top40_split_indices':[{'index':k,'splits':v} for k,v in s5_counts.most_common(40)],
       'S5_fold_model_training_proofs':hashes,
       'original_source_block_statistics_S5':scoped(s5_counts),
       'original_source_block_statistics_S4':scoped(s4_counts),
       'individual_models':details,
       'original_527D_numeric_feature_constructor_recovered':False,
       'original_416D_column_semantics_identified':False,
       'source_models_untouched':True,
       'historical_truth_or_MIDI_read':False,
       'independent_audio_S80_exactk_measured':False,
       'S80_promoted':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in (
       'status','original_S5_model_count','original_S4_owned_model_count',
       'missing_original_S4_owned_models',
       'S5_total_splits','S4_owned_total_splits',
       'S5_owned_suffix_416_active_columns','S4_owned_suffix_416_active_columns',
       'S5_S4_suffix_union_active','S5_S4_suffix_never_used_416_count',
       'original_source_block_statistics_S5','original_source_block_statistics_S4',
       'S80_promoted')},indent=2),flush=True)
def selftest():
    need(len(BLOCKS)==7 and BLOCKS[3][1]==111 and BLOCKS[-1][-1]==527,'S5 feature block audit')
    sample=collections.Counter({0:2,111:1,526:4})
    need(scoped(sample)[-1]['active_split_count']==4,'S5 block range exact')
    print('S200 PASS original S4/S5 527D fixed block census')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--models',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(a.models and a.out,'pinned original models and output required')
        run(a.models,a.out)
