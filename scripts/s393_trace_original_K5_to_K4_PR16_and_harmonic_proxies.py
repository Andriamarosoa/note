"""S393: Diagnose where S80 loses K5 using genuine ORIGINAL 36 PR16 proposals.

Read-only: true S80 K5->K4 misses versus correct K4 and correctly
classified K5, on the same GuitarSet native events. Test separately:
(1) whether any original 36 PR16 networks propose a fifth note,
(2) whether S72/S79/S80 kept K5,
(3) physical-ish 42x49 harmonic trajectory temporal onset contrasts.
This is descriptive on exposed GuitarSet. Original S272 contains training
overlap. No causal claim about WHICH fifth fundamental from count labels alone.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
def need(x,m):
    if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def describe(a,mask):
    values=np.asarray(a[mask],np.float64)
    need(len(values)>0 and np.isfinite(values).all(),'empty nonfinite stratum')
    return dict(n=len(values),median=float(np.median(values)),
        q25=float(np.quantile(values,.25)),q75=float(np.quantile(values,.75)),
        mean=float(np.mean(values)))
def run(audio,truth,pr,s80,your,out):
    need(not out.exists(),'read-only immutable S393 audit output')
    a=read(audio); t=read(truth); p=read(pr); s=read(s80); mt=read(your)
    order=np.argsort(mt['global_index'].astype(np.int64))
    ids=a['row_id'].astype(str);n=len(ids)
    y=t['true_K'].astype(int);d=s['s80_K'].astype(int)
    need(n==59309 and len(set(ids))==n
        and np.array_equal(ids,t['row_id'].astype(str))
        and np.array_equal(ids,p['row_id'].astype(str))
        and np.array_equal(ids,s['row_id'].astype(str))
        and np.array_equal(ids,mt['global_index'][order].astype(str))
        and np.array_equal(y,mt['k'][order])
        and np.array_equal(y[t['original_fold']==0],mt['k'][order][t['original_fold']==0])
        and int((y==5).sum())==355
        and int(((y==5)&(d==4)).sum())==224
        and int(((y==4)&(d==4)).sum())==806
        and int(((y==5)&(d==5)).sum())==99
        and p['pr16_policy_K'].shape==(n,36)
        and p['pr16_network_probabilities'].shape==(n,12,7)
        and np.array_equal(p['pr16_policy22_K'],s['original_PR16_policy22_K']),
        'genuine S258 ORIGINAL 36 PR16 and S272 S80 event alignment failed')
    votes=p['pr16_policy_K']
    probs=p['pr16_network_probabilities']
    need(np.isfinite(probs).all() and np.isin(votes,range(7)).all(),'network probs invalid')
    v5=(votes==5).sum(1);v4=(votes==4).sum(1)
    max5=np.max(probs[:,:,5],axis=1)
    mean5=np.mean(probs[:,:,5],axis=1)
    # 42 real spectral frames, 49 harmonic templates, event anchored at -80ms.
    eligible = d==4
    selected_idx=np.flatnonzero(eligible)
    # Avoid upcasting the full 59k×42×49 archive (unnecessary >1GB RAM).
    z=a['sequence42x49'][selected_idx].astype(np.float64)
    need(z.shape==(len(selected_idx),42,49) and np.isfinite(z).all(),'original audio harmonic trajectory 42x49 missing')
    x=np.expm1(z); del z
    need(np.isfinite(x).all() and np.all(x>=0.),'spectral activation negative')
    pre=x[:,:13,:].mean(axis=1)
    attack=x[:,13:24,:].mean(axis=1)
    late=x[:,24:,:].mean(axis=1)
    maxa=attack.max(axis=1)
    onset_gain=np.maximum(attack-pre,0)
    posflux=np.maximum(np.diff(x,axis=1),0).sum(axis=(1,2))
    negflux=np.maximum(-np.diff(x,axis=1),0).sum(axis=(1,2))
    local=dict(
        novel_track_count_at_20pct_peak=np.sum((attack>pre*2) & (attack>=maxa[:,None]*.2),axis=1),
        novel_track_count_at_10pct_peak=np.sum((attack>pre*2) & (attack>=maxa[:,None]*.1),axis=1),
        attack_gain_fraction=onset_gain.sum(1)/(attack.sum(1)+1e-9),
        attack_pre_mass_ratio=attack.sum(1)/(pre.sum(1)+1e-9),
        late_attack_mass_ratio=late.sum(1)/(attack.sum(1)+1e-9),
        positive_negative_flux_ratio=posflux/(negflux+1e-9),
        max_attack_template_share=maxa/(attack.sum(1)+1e-9),
        effective_attack_template_count=(attack.sum(1)**2)/(np.sum(attack**2,axis=1)+1e-9),
        positive_flow_mass=posflux,
        event_duration_ms=(a['end_samples'][selected_idx]-a['start_samples'][selected_idx])*1000/44100,
    )
    groups={
        'true_K5_missed_as_K4': (y==5)&(d==4),
        'true_K4_correct_as_K4': (y==4)&(d==4),
        'true_other_pred_K4': (~np.isin(y,[4,5]))&(d==4),
        'true_K5_correct_as_K5':(y==5)&(d==5),
        'true_K5_other_misses':(y==5)&(~np.isin(d,[4,5])),
    }
    diag={}
    for key,mask in groups.items():
        r=np.flatnonzero(mask)
        if not len(r): continue
        v=dict(n=int(len(r)),
           by_fold={str(f):int(np.count_nonzero(mask & (t['original_fold']==f)))
                    for f in (0,1,2,4)},
           unique_members=int(len(set(a['member'][r].astype(str)))),
           had_any_PR16_vote_K5=int(np.count_nonzero(v5[r]>=1)),
           had_2plus_PR16_vote_K5=int(np.count_nonzero(v5[r]>=2)),
           had_5plus_PR16_vote_K5=int(np.count_nonzero(v5[r]>=5)),
           no_PR16_vote_K5=int(np.count_nonzero(v5[r]==0)),
           PR16_policy22_K5=int(np.count_nonzero(p['pr16_policy22_K'][r]==5)),
           S72_K5=int(np.count_nonzero(s['s72_K'][r]==5)),
           S79_K5=int(np.count_nonzero(s['s79_K'][r]==5)),
           S80_K5=int(np.count_nonzero(s['s80_K'][r]==5)),
           yourmt3_correct=int(np.count_nonzero(mt['predicted'][order][r]==y[r])),
           PR16_vote5=describe(v5,mask),
           PR16_vote4=describe(v4,mask),
           PR16_network_max5=describe(max5,mask),
           PR16_network_mean5=describe(mean5,mask))
        if np.all(eligible[r]):
            v['harmonic_temporal_proxies']={name:describe(val,mask[selected_idx])
                                           for name,val in local.items()}
        diag[key]=v
    # Among K5 misses, quantify full source candidate vs selection failure.
    lost=groups['true_K5_missed_as_K4']
    aligned=lost & (v5>=1)
    never=lost & (v5==0)
    # No assumption that a candidate is routed into final S72: this is a
    # descriptive diagnostic of candidate bank ability, not actual gate access.
    outcome=dict(
        lost_K5_pred_K4=int(lost.sum()),
        PR16_36_K5_candidate_exists=int(aligned.sum()),
        PR16_36_K5_candidate_absent=int(never.sum()),
        of_lost_yourmt3_correct=int((lost&(mt['predicted'][order]==5)).sum()),
        of_lost_yourmt3_correct_and_PR16_candidate_exists=int((aligned&(mt['predicted'][order]==5)).sum()),
        of_lost_yourmt3_correct_and_PR16_candidate_absent=int((never&(mt['predicted'][order]==5)).sum()),
        while_S72_K5_count=int((lost&(s['s72_K']==5)).sum()),
        while_S79_K5_count=int((lost&(s['s79_K']==5)).sum()),
        while_PR16_policy22_K5_count=int((lost&(p['pr16_policy22_K']==5)).sum()),
        among_lost_with_candidate_votes_median=float(np.median(v5[aligned])) if aligned.any() else None,
    )
    report=dict(status='S393_ORIGINAL_K5_LOSS_STAGE_AND_PHYSICAL_PROXY_DIAGNOSTIC',
      native_events=n,total_trueK5=int((y==5).sum()),
      checksums={key:sha(val) for key,val in
                 [('audio',audio),('truth',truth),('pr16',pr),('s80',s80),('yourmt3',your)]},
      observed_groups=diag,
      PR16_K5_proposal_exists_is_NOT_proof_of_S72_route_access=True,
      lost_stage_inventory=outcome,
      cannot_identify_actual_missing_pitch_from_K_truth=True,
      genuine_tone_separation_not_proven_without_raw_wav_and_note_identity=True,
      S80_train_overlap_not_independent=True,
      pure_causality_not_proven=True,
      model_or_reference_changed=False,
      next_required_test='On WAV with per-pitch ground truth, compare missing-fundamental harmonic trajectory, spectral overlaps and note-onset timing before and after the candidate bank.')
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,dg in diag.items():
        print('S393 CATEGORY',name,json.dumps(dg),flush=True)
    print('S393 CAUSE TRIAGE',json.dumps(outcome),flush=True)
    print('S393 PASS READ-ONLY no causal acoustics proof / no promotion',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('audio','truth','pr','s80','your','out'):p.add_argument('--'+key,type=Path,required=True)
    x=p.parse_args();run(x.audio,x.truth,x.pr,x.s80,x.your,x.out)
