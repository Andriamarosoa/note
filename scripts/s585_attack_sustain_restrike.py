"""S585: 28-source four-state pitch head and fixed learned-restrike decoder.

Development is composer-blocked, using exactly S578's 28 exposed sources.
Audio features remain the original 52D; MIDI state is a training TARGET only.
No target MIDI, K prediction or live entry point is modified. Pedal-aware MIDI
activity is a supervision proxy, not a measured acoustic offset.
"""
from __future__ import annotations
import argparse, collections, hashlib, io, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scripts.s569_train_real_23_source_52D_temporal_restrike_note_heads import (
    recover, extension, add_five, temporal52, true_restrike_count)
from scripts.s578_train_28_source_middle12_specialist_vs_global52_composerblocked import (
    load_fifth, component_folds)
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import fetch_midi
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import need, sha
from scripts.s534_independent_real_repeat_notes_dedup_tolerance_audit import note_metrics
from scripts.s536_onset_burst_adaptive_restrike_dedup_loops import emit as old_emit

POLICIES = ('S578_global28', 'S585_state_strict100', 'S585_state_restrike')
STATES = ('inactive_or_resonance', 'key_or_pedal_active', 'new_attack', 'restrike')
RANGES = {'low_40_59': (40,59), 'mid_60_71': (60,71), 'high_72_88': (72,88)}

def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')

def load_training(root):
    a = SimpleNamespace()
    for prefix, man, mid, feat in (('old',469,471,527), ('new',537,542,540),
            ('s545',545,551,549), ('s554',554,559,557),
            ('s561',561,567,565), ('s570',570,575,573)):
        setattr(a, prefix+'_manifest', root/f'S{man}/report.json')
        setattr(a, prefix+'_midi_report', root/f'S{mid}/report.json')
        setattr(a, prefix+'_feature_report', root/f'S{feat}/report.json')
        paths = list((root/f'S{feat}').glob('*FEATURES_PREMIDI.jsonl'))
        need(len(paths)==1, f'Exactly one original feature ledger S{feat}')
        setattr(a, prefix+'_features', paths[0])
    raw,X,rec,Y,O,truth,proof,meta = recover(a)
    raw,X,rec,Y,O,truth,proof,meta = extension(a,meta,raw,rec,Y,O,truth,proof)
    rec,Y,O = add_five(a,meta,raw,Y,O,truth,proof)
    rec,Y,O = load_fifth(a,meta,raw,rec,Y,O,truth,proof)
    need(len(meta)==28, 'S585 same 28 training sources as S578')
    return raw,rec,Y,O,truth,meta

def pedal_intervals(blob):
    """Key release extends through CC64; real later strikes remain separate."""
    import mido
    midi = mido.MidiFile(file=io.BytesIO(blob))
    tempo=500000; t=0.; held=collections.defaultdict(list)
    sustained=collections.defaultdict(list); pedal=collections.defaultdict(bool)
    out=[]
    for m in mido.merge_tracks(midi.tracks):
        t += mido.tick2second(m.time,midi.ticks_per_beat,tempo)
        ch=getattr(m,'channel',0)
        if m.type=='set_tempo': tempo=m.tempo
        elif m.type=='control_change' and m.control==64:
            down=m.value>=64
            if pedal[ch] and not down:
                for p,st in sustained.pop(ch,[]): out.append((st,t,p))
            pedal[ch]=down
        elif m.type=='note_on' and m.velocity>0:
            held[(ch,m.note)].append(t)
        elif m.type=='note_off' or (m.type=='note_on' and m.velocity==0):
            if held[(ch,m.note)]:
                st=held[(ch,m.note)].pop(0)
                if pedal[ch]: sustained[ch].append((int(m.note),st))
                else: out.append((st,t,int(m.note)))
    # EOF is explicitly recorded, not an invented early offset.
    for (ch,p),starts in held.items(): out.extend((st,t,p) for st in starts)
    for notes in sustained.values(): out.extend((st,t,p) for p,st in notes)
    return sorted(out)

def state_targets(raw,rec,Y,truth,meta):
    state=np.zeros(Y.shape,np.int8); valid=np.ones(Y.shape,bool)
    proof={}
    for src,m in sorted(meta.items()):
        blob=fetch_midi(m['url'],m['midi'])
        need(hashlib.sha256(blob).hexdigest()==m['midi_sha'], 'S585 old MIDI SHA changed')
        offset=float(m['offset'])
        intervals=[(a-offset,b-offset,p) for a,b,p in pedal_intervals(blob)]
        ii=np.flatnonzero(rec==src)
        ts=np.asarray([raw[i]['original_V88_candidate_start_s'] for i in ii],float)
        for j in range(49):
            p=j+40
            attacks=np.array(sorted(q['time'] for q in truth[src] if p in q['notes']))
            spans=[(a,b) for a,b,k in intervals if k==p]
            for row,t in zip(ii,ts):
                active=any(a<=t<b for a,b in spans)
                state[row,j]=int(active)
                near=np.abs(attacks-t) if len(attacks) else np.array([])
                if Y[row,j]:
                    need(len(near)>0 and near.min()<=.0800001, 'S585 aligned positive missing real onset')
                    n=int(near.argmin()); at=attacks[n]
                    repeated=(n>0 and at-attacks[n-1]<=.3500001) or any(
                        a<at-.025 and b>at for a,b in spans)
                    state[row,j]=3 if repeated else 2
                elif len(near) and near.min()<=.08:
                    # Another neighboring V88 site may describe the same real strike.
                    # Do not fabricate an unambiguous sustain/background negative.
                    valid[row,j]=False
        proof[src]=dict(MIDI_sha=m['midi_sha'],composer=m['composer'],
            states={name:int(np.sum((state[ii]==k)&valid[ii])) for k,name in enumerate(STATES)},
            ambiguous_neighbor_pitch_sites_excluded=int((~valid[ii]).sum()))
    need(np.array_equal(state>=2,Y.astype(bool)), 'S585 original real note positives must be conserved')
    return state,valid,proof

def fit_state(X,state,valid,rec,train,full=False):
    from sklearn.ensemble import ExtraTreesClassifier
    mask=np.isin(rec,train)[:,None]&valid
    x=X[mask]; y=state[mask]
    counts=np.bincount(y,minlength=4)
    need(np.all(counts>20),'S585 four real musical target states need enough examples')
    # Attack mass stays 1/2, as in binary balanced S578. The other half is
    # divided between explicitly sustained pitches and inactive competitors.
    weights=np.array([.25/counts[0],.25/counts[1],
        .5/(counts[2]+counts[3]),.5/(counts[2]+counts[3])])*len(y)
    model=ExtraTreesClassifier(n_estimators=210 if full else 135,max_depth=19,
        min_samples_leaf=4,max_features=.8,n_jobs=2,random_state=58501)
    model.fit(x,y,sample_weight=weights[y])
    need(list(model.classes_)==[0,1,2,3], 'S585 four state ABI')
    return model

def emit(rows,prob,learned):
    """Original BP preservation/harmonic veto, optional learned rapid restrike."""
    output=[]; last={}; counts=collections.Counter()
    for i in sorted(range(len(rows)),key=lambda z:float(rows[z]['original_V88_candidate_start_s'])):
        row=rows[i]; t=float(row['original_V88_candidate_start_s'])
        bp=sorted(set(map(int,row['old_basicpitch_birth_pitches'])))
        if bp:
            output.append(dict(time=t,notes=bp))
            for p in bp: last[p]=t
            continue
        acoustic=np.asarray(row['per_midi_features_49x11'])
        attack=prob[i,:,2]+prob[i,:,3]; keep=[]
        for j in np.flatnonzero(attack>=.7):
            p=int(j+40)
            rapid=p in last and 0<=t-last[p]<=.100
            if rapid:
                if not (learned and prob[i,j,3]>=.5):
                    counts['rapid_candidate_suppressed']+=1;continue
                counts['rapid_candidate_passed_by_restrike_state']+=1
            harmonic=any(j-d in keep and
                acoustic[j,3]<.8*max(.001,acoustic[j-d,3]) and
                acoustic[j,5]<.8*max(.001,acoustic[j-d,5]) for d in (12,19,24))
            if harmonic: counts['weak_harmonic_suppressed']+=1;continue
            keep.append(int(j))
        if keep:
            notes=[j+40 for j in keep];output.append(dict(time=t,notes=notes))
            for p in notes: last[p]=t
    return output,dict(counts)

def predictions(raw,rec,global_prob,state_prob):
    result={name:{} for name in POLICIES};decisions={}
    for src in sorted(set(rec)):
        ii=np.flatnonzero(rec==src);rows=[raw[i] for i in ii]
        pairs=[(r,dict(raw_only_49_pitch_prob=global_prob[i])) for r,i in zip(rows,ii)]
        result[POLICIES[0]][src]=old_emit(pairs,dict(dedup_ms=100,harmonic_ratio=.8,rearticulation_ratio=0))[0]
        for name,learned in zip(POLICIES[1:],(False,True)):
            result[name][src],stat=emit(rows,state_prob[ii],learned)
            decisions[name+':'+src]=stat
    return result,decisions

def metrics(predicted,truth,meta):
    out={}
    for policy,sources in predicted.items():
        total={str(t):collections.Counter() for t in (30,50,80)}
        songs={}; registers={name:collections.Counter() for name in RANGES}
        for src,gt in sorted(truth.items()):
            guess=sources[src]; sm={}
            for t in (30,50,80):
                s=note_metrics(guess,gt,t/1000)
                total[str(t)].update({k:s[k] for k in ('TP','FP','FN')});sm[str(t)]=s
            for name,(lo,hi) in RANGES.items():
                def subset(g):
                    return [dict(time=x['time'],notes=[p for p in x['notes'] if lo<=p<=hi]) for x in g]
                s=note_metrics(subset(guess),subset(gt),.08)
                registers[name].update({k:s[k] for k in ('TP','FP','FN')})
            songs[src]=dict(composer=meta[src]['composer'],scores=sm)
        def full(c):
            return dict(c,F1=2*c['TP']/max(1,2*c['TP']+c['FP']+c['FN']),
                precision=c['TP']/max(1,c['TP']+c['FP']),recall=c['TP']/max(1,c['TP']+c['FN']))
        out[policy]=dict(fullstream={t:full(c) for t,c in total.items()},
            by_register={n:full(c) for n,c in registers.items()},per_song=songs)
        need(all(sum(c[k] for c in registers.values())==total['80'][k] for k in ('TP','FP','FN')),
             'S585 pitch registers must reconstruct complete metric')
    return out

def train(root,out):
    import joblib
    from sklearn.ensemble import ExtraTreesClassifier
    need(not out.exists(),'S585 immutable training output')
    raw,rec,Y,O,truth,meta=load_training(root)
    X=temporal52(raw);state,valid,labelproof=state_targets(raw,rec,Y,truth,meta)
    folds=component_folds(meta)
    gp=np.full(Y.shape,np.nan,np.float32); sp=np.full(Y.shape+(4,),np.nan,np.float32)
    for i,held in enumerate(folds):
        trainids=sorted(set(meta)-set(held));ix=np.flatnonzero(np.isin(rec,held))
        fit=np.flatnonzero(np.isin(rec,trainids)&O)
        control=ExtraTreesClassifier(n_estimators=135,max_depth=19,min_samples_leaf=4,
            max_features=.8,class_weight='balanced_subsample',random_state=57801,n_jobs=2)
        control.fit(X[fit].reshape(-1,52),Y[fit].reshape(-1))
        gp[ix]=control.predict_proba(X[ix].reshape(-1,52))[:,1].reshape(len(ix),49)
        model=fit_state(X,state,valid,rec,trainids)
        sp[ix]=model.predict_proba(X[ix].reshape(-1,52)).reshape(len(ix),49,4)
        print('S585 COMPOSER BLOCK',i,'train',len(trainids),'held',len(held),flush=True)
    need(np.isfinite(gp).all() and np.isfinite(sp).all(),'S585 complete OOF predictions')
    pred,decisions=predictions(raw,rec,gp,sp); scores=metrics(pred,truth,meta)
    prior=json.loads((root/'S578/report.json').read_text())
    expected=prior['full_noteon_evaluations']['global28']['metrics']
    replay=scores[POLICIES[0]]['fullstream']['80']
    need(all(replay[k]==expected[k] for k in ('TP','FP','FN')),'S578 exact original OOF replay failed')
    model=fit_state(X,state,valid,rec,sorted(meta),full=True)
    out.mkdir(parents=True)
    path=out/'S585_FULL28_four_state_52D.joblib';joblib.dump(model,path,compress=3)
    np.savez_compressed(out/'oof_probabilities.npz',global28=gp,state4=sp,state_target=state,
        eligible_training_target=valid,source=rec,row_id=np.asarray([r['V88_original_event_id'] for r in raw]))
    save(out/'training_source_proof.json',labelproof)
    save(out/'oof_notes.json',pred)
    report=dict(status='S585_28_SOURCE_FOUR_STATE_NOTE_HEAD_TRAINED',
        policies=list(POLICIES),states=list(STATES),training_sources=sorted(meta),
        training_composers={k:m['composer'] for k,m in meta.items()},composer_folds=folds,
        source_count=28,feature_dimensions=52,features_MIDI_free=True,
        genuine_fast_restrikes_under100ms=true_restrike_count(truth),
        ambiguous_neighbor_targets_excluded=int((~valid).sum()),
        state_counts={name:int(np.sum((state==j)&valid)) for j,name in enumerate(STATES)},
        MIDI_pedal_activity_is_supervision_proxy_not_measured_acoustic_offset=True,
        original_S578_OOF_counts_reproduced=True,development_not_new_test=True,
        model_filename=path.name,model_sha256=sha(path),
        OOF_sha256=sha(out/'oof_probabilities.npz'),label_proof_sha256=sha(out/'training_source_proof.json'),
        no_S577_or_sixth_cohort_training=True,no_promotion=True,
        original_S80_and_live_unchanged=True,fullstream_metrics=scores,decoder_counts=decisions)
    save(out/'report.json',report)
    for name in POLICIES: print('S585 DEVELOPMENT',name,json.dumps(scores[name]['fullstream']['80']),flush=True)

def selftest():
    import mido
    mf=mido.MidiFile(ticks_per_beat=500);track=mido.MidiTrack();mf.tracks.append(track)
    track.extend([mido.Message('note_on',note=60,velocity=80,time=0),
        mido.Message('control_change',control=64,value=127,time=10),
        mido.Message('note_off',note=60,time=30),
        mido.Message('note_on',note=60,velocity=80,time=20),
        mido.Message('note_off',note=60,time=30),
        mido.Message('control_change',control=64,value=0,time=110)])
    b=io.BytesIO();mf.save(file=b); spans=pedal_intervals(b.getvalue())
    need(len(spans)==2 and abs(spans[0][1]-.2)<1e-8 and abs(spans[1][0]-.06)<1e-8,
         'Pedal must preserve activity and a genuine 60ms restrike')
    raw=[dict(original_V88_candidate_start_s=t,old_basicpitch_birth_pitches=[],
        per_midi_features_49x11=np.ones((49,11)).tolist()) for t in (.1,.16)]
    q=np.zeros((2,49,4));q[:,:,0]=1
    q[0,20]=[.05,.05,.85,.05];q[1,20]=[.05,.05,.05,.85]
    strict,_=emit(raw,q,False); learned,_=emit(raw,q,True)
    need(sum(len(x['notes']) for x in strict)==1 and sum(len(x['notes']) for x in learned)==2,
         'Learned repeat must restore a real same-pitch strike suppressed by hard100')
    q[1,20]=[.05,.9,.025,.025]
    need(len(emit(raw,q,True)[0])==1,'Sustain state must not emit a new note')
    raw[1]['old_basicpitch_birth_pitches']=[60,64]
    need(emit(raw,q,True)[0][-1]['notes']==[60,64],'Original BP chord protected')
    q=np.random.default_rng(2).dirichlet(np.ones(4),size=(2,49))
    a,_=emit(raw,q,False)
    z,_=old_emit([(r,dict(raw_only_49_pitch_prob=q[i,:,2:].sum(axis=1))) for i,r in enumerate(raw)],
        dict(dedup_ms=100,harmonic_ratio=.8,rearticulation_ratio=0))
    need(a==z,'Strict control must use identical legacy emission semantics')
    print('S585 SELFTEST PASS pedal, fast real repeat, sustain suppression, BP protection, decoder parity')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true')
    p.add_argument('--root',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.self_test:selftest()
    else: train(a.root,a.out)
