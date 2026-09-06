"""Aggregate V22.1 and compare to V19, V17.3 and protected V10.4."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scripts import summarize_v171_controlled_assignment_ab as s171
from scripts import summarize_v176_shared_set_decoder as s176
from scripts import summarize_v190_dense_birth_centers as s190
from scripts import train_v221_joint_slot_attention as v221

MODEL_KEY=v221.MODEL_KEY; PRED_KEY=v221.PRED_KEY

def safe_global(r,s,m):
    row=r.get('strata',{}).get(s); return row.get(m,{}).get('metrics',{}).get('global') if row else None

def load_v221(root):
    reports=[]; parts=[]; root=Path(root)
    for fold in range(5):
        hits=[]
        for rp in root.glob(f'**/report-fold-{fold}.json'):
            r=json.loads(rp.read_text())
            if r.get('protocol',{}).get('v221_joint_slot_attention') is True:hits.append((rp,r))
        if len(hits)!=1:raise RuntimeError(f'v221 fold={fold}: expected one report, got {len(hits)}')
        rp,r=hits[0]; reports.append(r); npzs=list(rp.parent.glob(f'predictions-fold-{fold}.npz'))
        if len(npzs)!=1:raise RuntimeError(f'v221 fold={fold}: predictions missing')
        with np.load(npzs[0],allow_pickle=False) as z:
            n=len(z['global_index']); parts.append({k:np.asarray(z[k]) for k in z.files if np.asarray(z[k]).ndim and len(np.asarray(z[k]))==n})
    common=set(parts[0])
    for p in parts[1:]:common&=set(p)
    merged={k:np.concatenate([p[k] for p in parts],axis=0) for k in common}; order=np.argsort(merged['global_index'],kind='stable'); merged={k:v[order] for k,v in merged.items()}
    if len(merged['global_index'])!=76768 or len(np.unique(merged['global_index']))!=76768:raise RuntimeError('v221 invalid outer-clean coverage')
    return reports,merged

def weighted_slot_diag(reports):
    keys=('mean_pairwise_slot_cosine','p90_pairwise_slot_cosine','mean_cell_assignment_entropy','hard_cell_share_gini')
    total=sum(int(r['v221']['architecture']['slot_attention_diagnostics']['rows_sampled']) for r in reports)
    out={}
    for key in keys:
        out[key]=sum(int(r['v221']['architecture']['slot_attention_diagnostics']['rows_sampled'])*float(r['v221']['architecture']['slot_attention_diagnostics'][key]) for r in reports)/total
    shares=np.zeros(6,np.float64)
    for r in reports:
        d=r['v221']['architecture']['slot_attention_diagnostics']; shares+=int(d['rows_sampled'])*np.asarray(d['mean_hard_cell_share_by_slot'],np.float64)
    out['mean_hard_cell_share_by_slot']=(shares/total).tolist(); out['rows_sampled']=total
    return out

def summarize(a):
    r221,m221=load_v221(a.input_dir); r190,m190=s190._load_v190(a.v190_fold_dir); r173,m173=s176._load_version(a.v173_fold_dir,'v173')
    for key in ('global_index','k','member','pred104'):
        x=np.asarray(m221[key]).astype(str)
        if not np.array_equal(x,np.asarray(m190[key]).astype(str)) or not np.array_equal(x,np.asarray(m173[key]).astype(str)):raise RuntimeError(f'row mismatch {key}')
    k=np.asarray(m221['k'],np.int32); p104=np.asarray(m221['pred104'],np.int32); p173=np.asarray(m173['pred173_poibin'],np.int32); p190=np.asarray(m190[s190.PRED_KEY],np.int32); p221=np.asarray(m221[PRED_KEY],np.int32)
    preds={'v104':p104,'v173_poibin':p173,s190.MODEL_KEY:p190,MODEL_KEY:p221}; names=['aggregate','comp','solo','player00','player00_comp','player00_rock_comp']; strata={}
    for name in names:
        b=s171._metric_sum([safe_global(r,name,'v104') for r in r221]); n=s171._metric_sum([safe_global(r,name,MODEL_KEY) for r in r221]); o190=s171._metric_sum([safe_global(r,name,s190.MODEL_KEY) for r in r190]); o173=s171._metric_sum([safe_global(r,name,'v173_poibin') for r in r173])
        strata[name]={'v104':b,'v173_poibin':o173,s190.MODEL_KEY:o190,MODEL_KEY:n,'delta_v221_minus_v190_f1':n['f1']-o190['f1'],'delta_v221_minus_v173_f1':n['f1']-o173['f1'],'delta_v221_minus_v104_f1':n['f1']-b['f1']}
    cards={name:s171._card(k,p) for name,p in preds.items()}; per_k=s171._per_k(k,preds); folds={}; w190=w173=w104=0
    for fold in range(5):
        a221=next(r for r in r221 if int(r['outer_fold'])==fold); a190=next(r for r in r190 if int(r['outer_fold'])==fold); a173=next(r for r in r173 if int(r['outer_fold'])==fold)
        f221=float(a221['strata']['aggregate'][MODEL_KEY]['metrics']['global']['f1']); f190=float(a190['strata']['aggregate'][s190.MODEL_KEY]['metrics']['global']['f1']); f173=float(a173['strata']['aggregate']['v173_poibin']['metrics']['global']['f1']); f104=float(a221['strata']['aggregate']['v104']['metrics']['global']['f1']); w190+=int(f221>f190); w173+=int(f221>f173); w104+=int(f221>f104); ar=a221['v221']['architecture']; sd=ar['slot_attention_diagnostics']
        folds[str(fold)]={'selected_epochs':int(a221['data']['selected_epochs']),'v104_f1':f104,'v173_f1':f173,'v190_f1':f190,'v221_f1':f221,'delta_v221_minus_v190_f1':f221-f190,'activity_gini':float(ar['outer_activity_gini']),'effective_active_slots':float(ar['outer_effective_active_slots']),'slot_pair_cosine':float(sd['mean_pairwise_slot_cosine']),'slot_assignment_entropy':float(sd['mean_cell_assignment_entropy'])}
    sp190=s176._specialization(r190,'v190'); sp221=s176._specialization(r221,'v221'); dup190=s176._duplicate_slices(k,np.asarray(m190['presence'],np.float64),np.asarray(m190['event_candidate'],np.float64)); dup221=s176._duplicate_slices(k,np.asarray(m221['presence'],np.float64),np.asarray(m221['event_candidate'],np.float64)); sd=weighted_slot_diag(r221)
    agg=strata['aggregate']; comp={'global_f1_v104':float(agg['v104']['f1']),'global_f1_v173':float(agg['v173_poibin']['f1']),'global_f1_v190':float(agg[s190.MODEL_KEY]['f1']),'global_f1_v221':float(agg[MODEL_KEY]['f1']),'delta_v221_minus_v190_f1':float(agg['delta_v221_minus_v190_f1']),'delta_v221_minus_v173_f1':float(agg['delta_v221_minus_v173_f1']),'delta_v221_minus_v104_f1':float(agg['delta_v221_minus_v104_f1']),'folds_v221_beats_v190':w190,'folds_v221_beats_v173':w173,'folds_v221_beats_v104':w104,'poly_exact_v190':float(cards[s190.MODEL_KEY]['poly_accuracy']),'poly_exact_v221':float(cards[MODEL_KEY]['poly_accuracy']),'delta_poly_v221_minus_v190':float(cards[MODEL_KEY]['poly_accuracy']-cards[s190.MODEL_KEY]['poly_accuracy']),'activity_gini_v190':sp190['mean_active_rate_gini'],'activity_gini_v221':sp221['mean_active_rate_gini'],'effective_slots_v190':sp190['mean_effective_active_slots'],'effective_slots_v221':sp221['mean_effective_active_slots'],'raw_candidate_duplicate_poly_exact_v190':dup190['raw_duplicate_poly_exact_count'],'raw_candidate_duplicate_poly_exact_v221':dup221['raw_duplicate_poly_exact_count'],'player00_rock_comp_f1_v104':float(strata['player00_rock_comp']['v104']['f1']),'player00_rock_comp_f1_v221':float(strata['player00_rock_comp'][MODEL_KEY]['f1']),'slot_pair_cosine':sd['mean_pairwise_slot_cosine'],'slot_assignment_entropy':sd['mean_cell_assignment_entropy'],'slot_hard_share_gini':sd['hard_cell_share_gini']}
    for value in range(2,7):
        comp[f'delta_k{value}_exact_v221_minus_v190']=float(per_k[str(value)][MODEL_KEY]['exact']-per_k[str(value)][s190.MODEL_KEY]['exact'])
    gates={'global_f1_improved_vs_v190':comp['delta_v221_minus_v190_f1']>0,'majority_folds_improved_vs_v190':w190>=3,'beats_v104_global':comp['delta_v221_minus_v104_f1']>0,'poly_exact_improved_vs_v190':comp['delta_poly_v221_minus_v190']>0,'protected_player00_rock_comp_above_v104':comp['player00_rock_comp_f1_v221']>comp['player00_rock_comp_f1_v104']}
    result={'schema_version':1,'protocol':{'v221_joint_slot_attention':True,'outer_clean_rows':76768,'same_rows_as_v190_v173':True,'same_seed':16061,'presence_threshold':.5,'threshold_tuned':False,'discrete_topk_or_nms_in_proposal_path':False,'center_auxiliary_loss':False,'exact_720_matching':True,'poibin_weight':.35,'locked12_indexed_or_evaluated':False},'strata':strata,'cardinality':cards,'per_true_k':per_k,'folds':folds,'specialization':{'v190':sp190,'v221':sp221},'slot_attention':sd,'duplicates':{'v190':dup190,'v221':dup221},'comparison':comp,'gates':gates}
    a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); np.savez_compressed(a.output_dir/'predictions.npz',global_index=np.asarray(m221['global_index']),k=k,pred104=p104,pred173=p173,pred190=p190,pred221=p221,presence190=np.asarray(m190['presence']),presence221=np.asarray(m221['presence'])); print(json.dumps(comp,indent=2,sort_keys=True)); print(json.dumps(gates,indent=2,sort_keys=True)); return result

def parser():
    p=argparse.ArgumentParser(); p.add_argument('--input-dir',type=Path,required=True); p.add_argument('--v190-fold-dir',type=Path,required=True); p.add_argument('--v173-fold-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); return p
if __name__=='__main__':summarize(parser().parse_args())
