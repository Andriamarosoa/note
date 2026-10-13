"""S364 read-only forensic of prospective S363 12 IDMT chord S80 paired gain.

Audit concentration by each of 12 preregistered audio files,
paired corrections/regressions by true attack class, exact McNemar
conditional event-level p, and resampling/sign permutation by WAV.
Never reassign audio onset labels, change prediction models, optimize
thresholds or promote reference. WAV filename groups != verified song
composition independence; K0 truth still XML-exhaustiveness conditional.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from collections import Counter
from pathlib import Path
import numpy as np
def need(x,msg):
    if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def exact_binom_two_sided(fixes,losses):
    trials=fixes+losses
    if trials==0:return 1.
    b=min(fixes,losses)
    return min(1.,2*sum(math.comb(trials,k) for k in range(b+1))/2**trials)
def confusion(y,b,c,mask):
    i=np.flatnonzero(mask)
    x=(b[i]==y[i]);z=(c[i]==y[i])
    return dict(events=int(len(i)),official_correct=int(x.sum()),
        S80_correct=int(z.sum()),S80_corrections=int((~x&z).sum()),
        S80_regressions=int((x&~z).sum()),
        changed_both_wrong=int(((b[i]!=c[i])&~x&~z).sum()),
        diff_correct=int(z.sum()-x.sum()))
def interval_groups(y,b,c,member,mask,seed=356,n=10000):
    choices=sorted(set(member))
    lists=[np.flatnonzero((member==k)&mask) for k in choices]
    useful=[l for l in lists if len(l)]
    if len(useful)<2:return dict(groups=len(useful),ci95=None)
    rng=np.random.default_rng(seed)
    val=[]
    for _ in range(n):
        draw=rng.integers(0,len(useful),size=len(useful))
        rows=np.concatenate([useful[j] for j in draw])
        val.append(float(np.mean((c[rows]==y[rows]).astype(int)
                     -(b[rows]==y[rows]).astype(int))))
    ci=list(map(float,np.quantile(val,[.025,.975])))
    return dict(groups=len(useful),replicates=n,seed=seed,CI95=ci,
          includes_no_effect=ci[0]<=0<=ci[1])
def signflip_by_wav(y,b,c,member,mask):
    """Exact two-sided per-WAV random-sign test via dynamic programming.

    Unlike enumerating 2**24 sequences, grouped integer convolution
    yields the exact same p-value without 16,777,216 Python loops.
    Requires *exchangeable independent WAV units* -- not established
    for shared musical composition; treat p as conditional only.
    """
    groups=sorted(set(member))
    gains=[]
    for group in groups:
        i=np.flatnonzero((member==group)&mask)
        gains.append(int((c[i]==y[i]).sum()-(b[i]==y[i]).sum()))
    observed=sum(gains)
    dist=Counter({0:1})
    for gain in gains:
        nxt=Counter()
        for val,num in dist.items():
            nxt[val-gain]+=num
            nxt[val+gain]+=num
        dist=nxt
    total=2**len(gains)
    need(sum(dist.values())==total,"signflip exact integer convolution invalid")
    extreme=sum(v for k,v in dist.items() if abs(k)>=abs(observed))
    return dict(n_wav=len(groups),net_improvement=observed,
        two_sided_exact_group_signflip_p=float(extreme/total),
        group_gains=gains,
        groups_are_filename_hashes_not_verified_distinct_music_compositions=True)
def run(source,pred,out):
    need(not out.exists(),'no mutable test report')
    report=json.loads(source.read_text())
    need(report['status']=='S363_FIRST_POSTFROZEN_PROSPECTIVE_DATASET1_CHORDS_ATTACK_EXACTK_DUAL_PROTOCOL'
         and report['native_V88_events']==967 and report['prospective_WAVs']==24
         and report['source_predictions_all_frozen_before_XML']
         and not report['XML_note_transcription_exhaustiveness_verified']
         and not report['negative_K0_truth_independently_certified']
         and not report['official_S80_promotion'],
         'original prospective S363 source or nonpromotion caveats absent')
    with np.load(pred,allow_pickle=False) as z:
        required={'row_id','member','historical_attack_K_if_XML_complete',
                  'acoustic_anchored_K_or_unknown','official_freeze_K',
                  'S80_original_K','S80_twofold_K'}
        need(set(z.files)==required,'original S363 source archive fields drift')
        a={k:z[k].copy() for k in z.files}
    y=a['historical_attack_K_if_XML_complete']
    b=a['official_freeze_K'];c=a['S80_original_K'];strict=a['S80_twofold_K']
    member=a['member'].astype(str)
    need(len(y)==967
         and np.array_equal(np.unique(member),np.unique([
             p['wav_filename_SHA256'] for p in report['per_WAV_source_proofs']])),
         'S364 source not exactly 24 predeclared original recordings')
    total=confusion(y,b,c,np.ones(len(y),bool))
    poly=confusion(y,b,c,y>=2)
    known_positive=confusion(y,b,c,y>=1)
    need(total['events']==967 and total['S80_corrections']-total['S80_regressions']==total['diff_correct']
         and poly['events']==int(np.count_nonzero(y>=2)),
         'S363 source paired ledger does not agree with original SHA-frozen native cohort')
    by_class={str(k):confusion(y,b,c,y==k) for k in range(7)}
    provenance={p['wav_filename_SHA256']:p for p in report['per_WAV_source_proofs']}
    per_wav=[]
    for h in sorted(provenance):
        sel=member==h
        per_wav.append(dict(WAV_filename_SHA256=h,
              original_event_count=int(sel.sum()),
              true_poly_attack_count=int(np.count_nonzero(sel&(y>=2))),
              global_conditional=confusion(y,b,c,sel),
              positive=confusion(y,b,c,sel&(y>=1)),
              poly=confusion(y,b,c,sel&(y>=2)),
              XML_note_count=provenance[h]['XML_note_count'],
              acoustic_anchor_evidence=provenance[h]['acoustic_anchor_evidence']))
    changes=[x for x in per_wav if x['global_conditional']['diff_correct']]
    anchor_stats=dict(
       all_original_XML_note_attacks=sum(x['XML_note_count'] for x in per_wav),
       physical_anchored=sum(x['acoustic_anchor_evidence']['acoustic_peak_anchored'] for x in per_wav),
       unknown_or_unanchored=sum(x['acoustic_anchor_evidence']['unanchored_unknown'] for x in per_wav),
       all_chord_wavs_fail_closed=int(sum(not x['acoustic_anchor_evidence']['unanchored_unknown']==0
           or x['acoustic_anchor_evidence']['anchored_V88_unmatched']>0 for x in per_wav)),
       whole_cohort_audio_protocol_eligible_rows=report['physically_anchored_model_independent_rows'])
    result=dict(status='S364_READ_ONLY_PROSPECTIVE_CHORD_GAIN_DISTRIBUTION_AND_UNCERTAINTY',
       original_S363_report_sha256=sha(source),
       original_S363_predictions_sha256=sha(pred),
       paired_all_native_if_XML_complete=total,
       paired_positive_attack_only=known_positive,
       paired_true_poly_attack_only=poly,
       by_K=by_class,
       per_WAV=per_wav,
       recordings_with_nonzero_net_correct_change=len(changes),
       WAV_group_signflip_all=signflip_by_wav(y,b,c,member,np.ones(len(y),bool)),
       WAV_group_signflip_poly=signflip_by_wav(y,b,c,member,y>=2),
       WAV_cluster_bootstrap_all=interval_groups(y,b,c,member,np.ones(len(y),bool)),
       WAV_cluster_bootstrap_poly=interval_groups(y,b,c,member,y>=2),
       event_level_paired_exact_McNemar_p=exact_binom_two_sided(
           total['S80_corrections'],total['S80_regressions']),
       physical_anchor_quality=anchor_stats,
       source_predictions_unmodified=True,
       alternate_physical_alignment_did_not_yield_scored_events=True,
       independent_XML_K0_exhaustiveness_proven=False,
       independent_song_composition_identity_proven=False,
       reference_promotion_authorized=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()
        if k not in ('per_WAV','by_K')},indent=2),flush=True)
    print('PER-WAV NET GAINS',json.dumps(
        [(i['WAV_filename_SHA256'],i['global_conditional']['diff_correct'],
        i['poly']['diff_correct']) for i in per_wav]),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for x in ('source','pred','out'):ap.add_argument('--'+x,required=True,type=Path)
    q=ap.parse_args();run(q.source,q.pred,q.out)
