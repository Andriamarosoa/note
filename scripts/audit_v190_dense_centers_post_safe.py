"""Memory-safe V19 dense-center post-audit; no retraining, Locked12 untouched."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'src'
for p in (ROOT,SRC):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from scripts import train_v100_spectral_string_slots as v100
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v172_mass_preserving_exchangeable as v172
from scripts import train_v190_dense_birth_centers as v190

TOLS=((0,0),(1,1),(2,2),(1,2),(2,4)); BATCH=384

def one(root,pat):
    xs=sorted(root.glob(pat))
    if len(xs)!=1: raise RuntimeError(f'expected one {pat}, got {len(xs)}')
    return xs[0]

def nms_batched(score):
    n=len(score); out=np.empty((n,6),np.int32)
    for s in range(0,n,BATCH):
        e=min(n,s+BATCH)
        out[s:e]=v190._nms_topk_np(np.asarray(score[s:e],dtype=np.float32))
    return out

def injective(truth,sel,dt,df):
    if not len(truth): return True
    truth=np.asarray(truth,np.int32); sel=np.asarray(sel,np.int32)
    tt,tf=truth//64,truth%64; st,sf=sel//64,sel%64
    ok=(np.abs(tt[:,None]-st[None,:])<=dt)&(np.abs(tf[:,None]-sf[None,:])<=df)
    states={0}
    for i in range(len(truth)):
        nxt=set()
        for state in states:
            for j in np.flatnonzero(ok[i]):
                bit=1<<int(j)
                if not state&bit: nxt.add(state|bit)
        if not nxt:return False
        states=nxt
    return True

def selection(score,target,eligible,k):
    chosen=nms_batched(score); eligible=np.asarray(eligible,bool); k=np.asarray(k,np.int32)
    exact={f'dt{a}_df{b}':np.zeros(len(k),bool) for a,b in TOLS}
    hit={f'dt{a}_df{b}':np.zeros(len(k),np.float32) for a,b in TOLS}
    ndt=[]; ndf=[]
    for r in np.flatnonzero(eligible&(k>0)):
        truth=np.flatnonzero(target[r]>0); sel=chosen[r]
        tt,tf=truth//64,truth%64; st,sf=sel//64,sel%64
        dtime=np.abs(tt[:,None]-st[None,:]); dfreq=np.abs(tf[:,None]-sf[None,:])
        for i in range(len(truth)):
            j=int(np.argmin(dtime[i]+dfreq[i])); ndt.append(int(dtime[i,j])); ndf.append(int(dfreq[i,j]))
        for a,b in TOLS:
            key=f'dt{a}_df{b}'; ok=(dtime<=a)&(dfreq<=b)
            hit[key][r]=np.mean(np.any(ok,axis=1)); exact[key][r]=injective(truth,sel,a,b)
    def pack(mask):
        return {key:{'exact_injective_coverage':float(np.mean(exact[key][mask])),'mean_truth_hit_fraction':float(np.mean(hit[key][mask]))} for key in exact}
    pos=eligible&(k>0); poly=eligible&(k>=2)
    return {'positive':pack(pos),'poly':pack(poly),'median_nearest_time_bins':float(np.median(ndt)),'median_nearest_frequency_bins':float(np.median(ndf)),'mean_nearest_time_bins':float(np.mean(ndt)),'mean_nearest_frequency_bins':float(np.mean(ndf))},chosen

def rank_ce(prob,target,eligible,k):
    ranks=[]; ps=[]; ces=[]; floors=[]; masses=[]; ks=[]
    for r in np.flatnonzero(np.asarray(eligible,bool)&(np.asarray(k)>0)):
        pr=np.asarray(prob[r],np.float64); y=np.asarray(target[r],np.float64); truth=np.flatnonzero(y>0)
        # rank without a full argsort: 1 + number of strictly larger probabilities.
        for t in truth:
            ranks.append(1+int(np.sum(pr>pr[t]))); ps.append(float(pr[t]))
        ce=float(-np.sum(y*np.log(np.clip(pr,1e-12,1.0)))); yy=y[y>0]; floor=float(-np.sum(yy*np.log(yy)))
        ces.append(ce); floors.append(floor); masses.append(float(np.sum(pr[truth]))); ks.append(int(k[r]))
    ranks=np.asarray(ranks); ps=np.asarray(ps); ces=np.asarray(ces); floors=np.asarray(floors); masses=np.asarray(masses); ks=np.asarray(ks)
    out={'truth_cell_median_rank':float(np.median(ranks)),'truth_cell_mean_rank':float(np.mean(ranks)),'truth_cell_rank_le_6':float(np.mean(ranks<=6)),'truth_cell_rank_le_20':float(np.mean(ranks<=20)),'truth_cell_rank_le_50':float(np.mean(ranks<=50)),'truth_cell_rank_le_100':float(np.mean(ranks<=100)),'truth_cell_mean_probability':float(np.mean(ps)),'row_mean_true_center_probability_mass':float(np.mean(masses)),'row_mean_categorical_ce':float(np.mean(ces)),'row_mean_information_floor':float(np.mean(floors)),'row_mean_excess_ce_over_floor':float(np.mean(ces-floors)),'per_true_k':{}}
    for v in range(1,7):
        m=ks==v
        if np.any(m): out['per_true_k'][str(v)]={'rows':int(np.sum(m)),'mean_ce':float(np.mean(ces[m])),'mean_information_floor':float(np.mean(floors[m])),'mean_excess_ce':float(np.mean((ces-floors)[m])),'mean_true_center_mass':float(np.mean(masses[m]))}
    return out

def js(chosen,target,eligible,k):
    p=np.zeros(v190.CENTER_CELLS,np.float64); q=np.zeros_like(p)
    for r in np.flatnonzero(np.asarray(eligible,bool)&(np.asarray(k)>0)):
        np.add.at(p,chosen[r],1); np.add.at(q,np.flatnonzero(target[r]>0),1)
    p/=p.sum(); q/=q.sum(); m=.5*(p+q)
    def kl(a,b):
        z=a>0; return float(np.sum(a[z]*np.log(a[z]/np.clip(b[z],1e-12,None))))
    return .5*kl(p,m)+.5*kl(q,m)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('dataset_dir',type=Path); ap.add_argument('--cache-dir',type=Path,required=True); ap.add_argument('--v190-fold-dir',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    cache=v100._load_spectral_caches(a.cache_dir); members=np.asarray([str(x) for x in cache['members']],dtype='U96')
    cand,recon=v102._reconstruct_candidates(cache); pitch,tm,td,ts,sup=v102._derive_supervision(members,cand,a.dataset_dir,expected_slot_targets=cache['slot_targets'])
    k=np.minimum(np.asarray(cache['exact'],np.int32),6); target,eligf,distinct=v190._birth_center_targets(cache,pitch,td,k); elig=eligf>.5
    active=np.asarray(tm)>.5; disag=checked=0
    grid=np.asarray(v102.FRAME_CENTER_SAMPLES,np.float64)
    for r in np.flatnonzero(elig):
        slots=np.flatnonzero(active[r]);
        if len(slots)!=int(k[r]): continue
        aa=np.argmin(np.abs(np.asarray(ts[r,slots],np.float64)[:,None]-grid[None,:]),axis=1); bb=np.argmax(np.asarray(td[r,slots]),axis=1)
        disag+=int(np.sum(aa!=bb)); checked+=len(slots)
    learned=np.zeros((len(k),v190.CENTER_CELLS),np.float32); owner=np.full(len(k),-1,np.int16); folds=[]
    import tensorflow as tf
    from tensorflow import keras
    for fold in range(5):
        ctx=v172._fold_context(SimpleNamespace(dataset_dir=a.dataset_dir,cache_dir=a.cache_dir,outer_fold=fold)); model,_,_=v190._build_model(ctx['final_spec']); model.load_weights(one(a.v190_fold_dir,f'**/v190-dense-birth-centers-fold-{fold}.weights.h5'))
        sin=next(t for t in model.inputs if t.name.split(':',1)[0]=='spectral_map'); cm=keras.Model(sin,model.get_layer('birth_center_map').output); idx=np.asarray(ctx['outer_idx'],np.int64)
        # Feed only the spectral input to the center-only submodel, in small batches.
        learned[idx]=cm.predict(np.asarray(cache['spectral'])[idx],batch_size=256,verbose=0); owner[idx]=fold
        rr=json.loads(one(a.v190_fold_dir,f'**/report-fold-{fold}.json').read_text()); d=rr['v190']['architecture']['dense_center_diagnostics']
        folds.append({'fold':fold,'v190_f1':float(rr['strata']['aggregate'][v190.MODEL_KEY]['metrics']['global']['f1']),'v104_f1':float(rr['strata']['aggregate']['v104']['metrics']['global']['f1']),'center_exact_poly_reported':float(d['top6_exact_center_coverage_poly']),'center_hit_poly_reported':float(d['top6_mean_center_hit_fraction_poly'])})
        tf.keras.backend.clear_session()
    if np.any(owner<0): raise RuntimeError('outer fold coverage incomplete')
    learned_sel,learned_chosen=selection(learned,target,elig,k); learned_rank=rank_ce(learned,target,elig,k)
    spectral=np.asarray(cache['spectral']) # keep source float16
    raw={}
    for ch,name in ((0,'log_power'),(1,'positive_pre'),(2,'flux')):
        score=spectral[:,:,:,ch].reshape((len(k),-1)); sm,chosen=selection(score,target,elig,k); raw[name]={'selection':sm,'selected_truth_js_divergence':js(chosen,target,elig,k)}
    poly=elig&(k>=2); center_hits=np.asarray([x['center_hit_poly_reported'] for x in folds]); f1=np.asarray([x['v190_f1'] for x in folds]); corr=float(np.corrcoef(center_hits,f1)[0,1]) if np.std(center_hits)*np.std(f1)>0 else 0.0
    result={'schema_version':1,'protocol':{'v190_post_audit':True,'outer_clean_rows':int(len(k)),'same_five_saved_v190_models':True,'no_retraining':True,'training_annotations_used_for_target_reconstruction_only':True,'runtime_annotations_required':False,'locked12_indexed_or_evaluated':False},'target_consistency':{'eligible_rows':int(np.sum(elig)),'eligible_poly_rows':int(np.sum(poly)),'time_targets_checked':int(checked),'argmax_time_vs_nearest_sample_disagreements':int(disag),'argmax_time_vs_nearest_sample_disagreement_rate':float(disag/max(1,checked)),'representable_distinct_center_poly_rate':float(np.mean(distinct[poly]==k[poly]))},'learned_center':{'selection':learned_sel,'rank_and_ce':learned_rank,'selected_truth_js_divergence':js(learned_chosen,target,elig,k)},'raw_spectral_baselines':raw,'folds':folds,'fold_correlation_center_hit_vs_v190_f1':corr,'structural_finding':{'topk_indices_differentiable_from_event_set_loss':False,'reason':'tf.math.top_k indices -> gather: downstream event-set gradients update gathered features but cannot move the discrete selected index; coordinate placement is directly trained only by center-map CE','categorical_softmax_multi_center_floor':'one 1472-way softmax shares total probability mass 1 across K true centers; for K distinct uniform targets the minimum CE is log(K), so this is a one-draw distribution rather than K independent occupancy decisions'},'supervision':{'source':sup,'candidate_reconstruction':recon}}
    (a.output_dir/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'target_consistency':result['target_consistency'],'learned_poly_tolerances':learned_sel['poly'],'learned_rank_and_ce':learned_rank,'raw_flux_poly':raw['flux']['selection']['poly'],'raw_positive_pre_poly':raw['positive_pre']['selection']['poly'],'fold_correlation_center_hit_vs_v190_f1':corr},indent=2,sort_keys=True))
if __name__=='__main__': main()
