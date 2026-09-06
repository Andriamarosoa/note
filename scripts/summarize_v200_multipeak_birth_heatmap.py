"""Aggregate V20 and compare directly to V19, canonical V17.3 and V10.4."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scripts import summarize_v171_controlled_assignment_ab as s171
from scripts import summarize_v176_shared_set_decoder as s176
from scripts import summarize_v190_dense_birth_centers as s190
from scripts import train_v200_multipeak_birth_heatmap as v200

MODEL_KEY=v200.MODEL_KEY; PRED_KEY=v200.PRED_KEY; THRESHOLD=v200.PRESENCE_THRESHOLD

def safe_global(r,s,m):
    row=r.get('strata',{}).get(s); return row.get(m,{}).get('metrics',{}).get('global') if row else None

def load_v200(root):
    reports=[]; parts=[]; root=Path(root)
    for fold in range(5):
        hits=[]
        for rp in root.glob(f'**/report-fold-{fold}.json'):
            r=json.loads(rp.read_text())
            if r.get('protocol',{}).get('v200_multipeak_birth_heatmap') is True:hits.append((rp,r))
        if len(hits)!=1:raise RuntimeError(f'v200 fold={fold}: expected one report, got {len(hits)}')
        rp,r=hits[0]; reports.append(r); npzs=list(rp.parent.glob(f'predictions-fold-{fold}.npz'))
        if len(npzs)!=1:raise RuntimeError(f'v200 fold={fold}: predictions missing')
        with np.load(npzs[0],allow_pickle=False) as z:
            n=len(z['global_index']); parts.append({key:np.asarray(z[key]) for key in z.files if np.asarray(z[key]).ndim and len(np.asarray(z[key]))==n})
    common=set(parts[0])
    for p in parts[1:]:common&=set(p)
    merged={key:np.concatenate([p[key] for p in parts],axis=0) for key in common}; order=np.argsort(merged['global_index'],kind='stable'); merged={k:v[order] for k,v in merged.items()}
    if len(merged['global_index'])!=76768 or len(np.unique(merged['global_index']))!=76768:raise RuntimeError('v200 invalid outer-clean coverage')
    return reports,merged

def weighted_center(reports,block):
    pn=qn=pe=qe=qh=0.0; pk={str(k):{'n':0,'e':0.0,'h':0.0} for k in range(1,7)}
    for r in reports:
        d=r[block]['architecture']['dense_center_diagnostics']; a=int(d['eligible_positive_rows']); b=int(d['eligible_poly_rows']); pn+=a; qn+=b
        if d['top6_exact_center_coverage_positive'] is not None:pe+=a*float(d['top6_exact_center_coverage_positive'])
        if d['top6_exact_center_coverage_poly'] is not None:qe+=b*float(d['top6_exact_center_coverage_poly'])
        if d['top6_mean_center_hit_fraction_poly'] is not None:qh+=b*float(d['top6_mean_center_hit_fraction_poly'])
        for k in range(1,7):
            x=d['per_true_k'][str(k)]; n=int(x['rows']); pk[str(k)]['n']+=n
            if x['top6_exact_center_coverage'] is not None:pk[str(k)]['e']+=n*float(x['top6_exact_center_coverage'])
            if x['top6_mean_center_hit_fraction'] is not None:pk[str(k)]['h']+=n*float(x['top6_mean_center_hit_fraction'])
    return {'eligible_positive_rows':int(pn),'eligible_poly_rows':int(qn),'top6_exact_center_coverage_positive':pe/pn if pn else None,'top6_exact_center_coverage_poly':qe/qn if qn else None,'top6_mean_center_hit_fraction_poly':qh/qn if qn else None,'per_true_k':{k:{'rows':v['n'],'top6_exact_center_coverage':v['e']/v['n'] if v['n'] else None,'top6_mean_center_hit_fraction':v['h']/v['n'] if v['n'] else None} for k,v in pk.items()}}

def summarize(a):
    r200,m200=load_v200(a.input_dir); r190,m190=s190._load_v190(a.v190_fold_dir); r173,m173=s176._load_version(a.v173_fold_dir,'v173'); sum173=json.loads((a.v173_summary_dir/'report.json').read_text())
    for key in ('global_index','k','member','pred104'):
        x=np.asarray(m200[key]).astype(str)
        if not np.array_equal(x,np.asarray(m190[key]).astype(str)) or not np.array_equal(x,np.asarray(m173[key]).astype(str)):raise RuntimeError(f'row mismatch {key}')
    k=np.asarray(m200['k'],np.int32); p104=np.asarray(m200['pred104'],np.int32); p173=np.asarray(m173['pred173_poibin'],np.int32); p190=np.asarray(m190[s190.PRED_KEY],np.int32); p200=np.asarray(m200[PRED_KEY],np.int32)
    preds={'v104':p104,'v173_poibin':p173,s190.MODEL_KEY:p190,MODEL_KEY:p200}; names=['aggregate','comp','solo','player00','player00_comp','player00_rock_comp']; strata={}
    for name in names:
        b=s171._metric_sum([safe_global(r,name,'v104') for r in r200]); n=s171._metric_sum([safe_global(r,name,MODEL_KEY) for r in r200]); o190=s171._metric_sum([safe_global(r,name,s190.MODEL_KEY) for r in r190]); o173=sum173['strata'][name]['v173_poibin']
        strata[name]={'v104':b,'v173_poibin':o173,s190.MODEL_KEY:o190,MODEL_KEY:n,'delta_v200_minus_v190_f1':n['f1']-o190['f1'],'delta_v200_minus_v173_f1':n['f1']-o173['f1'],'delta_v200_minus_v104_f1':n['f1']-b['f1'],'delta_v200_minus_v190_precision':n['precision']-o190['precision'],'delta_v200_minus_v190_recall':n['recall']-o190['recall']}
    cards={name:s171._card(k,p) for name,p in preds.items()}; per_k=s171._per_k(k,preds); folds={}; w190=w173=w104=0
    for fold in range(5):
        a200=next(r for r in r200 if int(r['outer_fold'])==fold); a190=next(r for r in r190 if int(r['outer_fold'])==fold); a173=next(r for r in r173 if int(r['outer_fold'])==fold)
        f200=float(a200['strata']['aggregate'][MODEL_KEY]['metrics']['global']['f1']); f190=float(a190['strata']['aggregate'][s190.MODEL_KEY]['metrics']['global']['f1']); f173=float(a173['strata']['aggregate']['v173_poibin']['metrics']['global']['f1']); f104=float(a200['strata']['aggregate']['v104']['metrics']['global']['f1']); w190+=int(f200>f190); w173+=int(f200>f173); w104+=int(f200>f104); ar=a200['v200']['architecture']; c=ar['dense_center_diagnostics']
        folds[str(fold)]={'selected_epochs':int(a200['data']['selected_epochs']),'v104_f1':f104,'v173_f1':f173,'v190_f1':f190,'v200_f1':f200,'delta_v200_minus_v190_f1':f200-f190,'delta_v200_minus_v173_f1':f200-f173,'center_top6_exact_poly':c['top6_exact_center_coverage_poly'],'center_top6_hit_fraction_poly':c['top6_mean_center_hit_fraction_poly'],'activity_gini':float(ar['outer_activity_gini']),'effective_active_slots':float(ar['outer_effective_active_slots'])}
    c190=weighted_center(r190,'v190'); c200=weighted_center(r200,'v200'); sp190=s176._specialization(r190,'v190'); sp200=s176._specialization(r200,'v200'); dup190=s176._duplicate_slices(k,np.asarray(m190['presence'],np.float64),np.asarray(m190['event_candidate'],np.float64)); dup200=s176._duplicate_slices(k,np.asarray(m200['presence'],np.float64),np.asarray(m200['event_candidate'],np.float64))
    agg=strata['aggregate']; comp={'global_f1_v104':float(agg['v104']['f1']),'global_f1_v173':float(agg['v173_poibin']['f1']),'global_f1_v190':float(agg[s190.MODEL_KEY]['f1']),'global_f1_v200':float(agg[MODEL_KEY]['f1']),'delta_v200_minus_v190_f1':float(agg['delta_v200_minus_v190_f1']),'delta_v200_minus_v173_f1':float(agg['delta_v200_minus_v173_f1']),'delta_v200_minus_v104_f1':float(agg['delta_v200_minus_v104_f1']),'delta_v200_minus_v190_precision':float(agg['delta_v200_minus_v190_precision']),'delta_v200_minus_v190_recall':float(agg['delta_v200_minus_v190_recall']),'folds_v200_beats_v190':w190,'folds_v200_beats_v173':w173,'folds_v200_beats_v104':w104,'poly_exact_v190':float(cards[s190.MODEL_KEY]['poly_accuracy']),'poly_exact_v200':float(cards[MODEL_KEY]['poly_accuracy']),'delta_poly_v200_minus_v190':float(cards[MODEL_KEY]['poly_accuracy']-cards[s190.MODEL_KEY]['poly_accuracy']),'center_exact_poly_v190':c190['top6_exact_center_coverage_poly'],'center_exact_poly_v200':c200['top6_exact_center_coverage_poly'],'center_hit_poly_v190':c190['top6_mean_center_hit_fraction_poly'],'center_hit_poly_v200':c200['top6_mean_center_hit_fraction_poly'],'activity_gini_v190':sp190['mean_active_rate_gini'],'activity_gini_v200':sp200['mean_active_rate_gini'],'effective_slots_v190':sp190['mean_effective_active_slots'],'effective_slots_v200':sp200['mean_effective_active_slots'],'raw_candidate_duplicate_poly_exact_v190':dup190['raw_duplicate_poly_exact_count'],'raw_candidate_duplicate_poly_exact_v200':dup200['raw_duplicate_poly_exact_count'],'player00_rock_comp_f1_v104':float(strata['player00_rock_comp']['v104']['f1']),'player00_rock_comp_f1_v200':float(strata['player00_rock_comp'][MODEL_KEY]['f1'])}
    for value in range(2,7):
        comp[f'delta_k{value}_exact_v200_minus_v190']=float(per_k[str(value)][MODEL_KEY]['exact']-per_k[str(value)][s190.MODEL_KEY]['exact']); comp[f'delta_k{value}_exact_v200_minus_v173']=float(per_k[str(value)][MODEL_KEY]['exact']-per_k[str(value)]['v173_poibin']['exact'])
    gates={'center_exact_poly_improved_vs_v190':c200['top6_exact_center_coverage_poly']>c190['top6_exact_center_coverage_poly'],'center_hit_poly_improved_vs_v190':c200['top6_mean_center_hit_fraction_poly']>c190['top6_mean_center_hit_fraction_poly'],'global_f1_improved_vs_v190':comp['delta_v200_minus_v190_f1']>0,'majority_folds_improved_vs_v190':w190>=3,'beats_v104_global':comp['delta_v200_minus_v104_f1']>0,'protected_player00_rock_comp_above_v104':comp['player00_rock_comp_f1_v200']>comp['player00_rock_comp_f1_v104']}
    result={'schema_version':1,'protocol':{'v200_multipeak_birth_heatmap':True,'controlled_against_v190':True,'outer_clean_rows':76768,'same_rows_as_v190_v173':True,'same_seed':16061,'presence_threshold':THRESHOLD,'threshold_tuned':False,'center_loss':'independent_binary_focal','center_loss_weight':v200.CENTER_MAP_WEIGHT,'center_loss_weight_tuned':False,'v190_dense_conv_and_top6_nms_unchanged':True,'v173_set_and_poibin_objectives_unchanged':True,'locked12_indexed_or_evaluated':False},'strata':strata,'cardinality':cards,'per_true_k':per_k,'folds':folds,'dense_center':{'v190':c190,'v200':c200},'specialization':{'v190':sp190,'v200':sp200},'duplicates':{'v190':dup190,'v200':dup200},'comparison':comp,'gates':gates}
    a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); np.savez_compressed(a.output_dir/'predictions.npz',global_index=np.asarray(m200['global_index']),k=k,pred104=p104,pred173=p173,pred190=p190,pred200=p200,presence190=np.asarray(m190['presence']),presence200=np.asarray(m200['presence'])); print(json.dumps(comp,indent=2,sort_keys=True)); print(json.dumps(gates,indent=2,sort_keys=True)); return result

def parser():
    p=argparse.ArgumentParser(); p.add_argument('--input-dir',type=Path,required=True); p.add_argument('--v190-fold-dir',type=Path,required=True); p.add_argument('--v173-fold-dir',type=Path,required=True); p.add_argument('--v173-summary-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); return p
if __name__=='__main__':summarize(parser().parse_args())
