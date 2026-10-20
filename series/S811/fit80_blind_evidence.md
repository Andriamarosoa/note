# S811 — Blind input shared identically with three NVIDIA NIM models

Context: factual material only; no previous review, assistant proposal, inferred cause or preferred correction. All code in the excerpts is from the S810 trainer supplied in the conversation, SHA256 `687f731fa255d3aaba33fa07beb04b41d836c3d48d6b8c2e98581a2ecf693059`. It is a locally executed Python trainer, not yet present in GitHub. The code lines below correspond to that original file. Raw recordings, per-track labels, HELD/TEST and CAL32 content are NOT provided.

Source: GuitarSet signed S800 FIT80 OOF, 80 tracks, 5 disjoint leave-one-composition-out folds (BN1, Funk2, Jazz1, Rock3, SS1), decision hop 1024 at 44100 Hz, features [frames,49,23] derived from received or past acoustic audio. S810 has 6 separate fitted sklearn StandardScaler + LogisticRegression models for EACH of 5 held-out groups, 30 saved checkpoints, no final MIDI integration and no production modification. S708 MIDI reference protected. All outputs measured are *standalone shadow axis* statistics on 1024-sample grid, NOT the six official final MIDI event metrics.

## Raw FIT80 evaluated counts (same 80 tracks)

| Axis | Metric | TP | FP | FN | Other counts |
| --- | --- | ---: | ---: | ---: | --- |
| onset | per-pitch/frame positive event | 10002 | 154245 | 1247 | F1 0.11398550394310981 |
| offset | per-pitch/frame positive event | 10479 | 142827 | 647 | F1 0.12745694268755473 |
| frame_pitch | per-pitch/frame occupancy | 132708 | 64190 | 7546 | F1 0.7872294988610479 |
| full_note | four-state frame proxy (onset class only) | 8518 | 94655 | 2731 | onset proxy F1 0.14888745171383125; all-state accuracy 3593452/3753400 |
| exact_k | independent 7-class classifier | — | — | — | correct 58832/76600 = 0.7680417754569191 |
| active_exact_k | independent 7-class classifier, true K>0 | — | — | — | correct 47708/62310 = 0.7656555930027283 |

All counts source: S810 `s810_fit80_trained_six_axes.json`, locally trained source SHA above. Checkpoint hashes present in that JSON. Report in repo: `series/S810/resultat.md`. Archived S809 shadow prior metrics (same FIT80 decision-clock, different methods) are different: active-K 42451/62310 and exact-K 64525/76600. The official S708 CAL32 has NOT been rerun here; these are not official regression numbers.

## Verbatim critical S810 training code and line references

```python
# s810_six_independent_fit80.py lines 14-30
SCHEMA='S810_REAL_FIT80_SIX_TRAINED_INDEPENDENT_AXES_V1'
AXES=('exact_k','active_exact_k','onset','offset','frame_pitch','full_note')
SR=44100
PITCHES=49
FEATURE_COLUMNS={
 'onset':(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,22),
 'offset':(0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,20,21,22),
 'frame_pitch':(0,1,2,3,4,6,7,8,10,11,12,13,14,15,16,17,18,19,20,22),
 'full_note':(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22),
}
TRAIN_CAP=480
COUNT_CAP=160
SEED=810

# lines 33-61
def truth_arrays(notes,ends):
    ends=np.asarray(ends,np.int64)
    now=ends/SR
    active=np.zeros((len(ends),PITCHES),np.uint8)
    count=np.zeros(len(ends),np.uint8)
    onset=np.zeros_like(active)
    release=np.zeros_like(active)
    for n in notes:
        col=int(n['pitch'])-40
        if not 0<=col<PITCHES:raise AssertionError('unexpected guitar pitch')
        begin=float(n['time'])
        end=begin+float(n['duration'])
        on=(now>=begin)&(now<end)
        active[on,col]=1
        count[on]+=1
        birth=int(np.searchsorted(ends,begin*SR,side='left'))
        if 0<=birth<len(ends) and ends[birth]-1024-1e-5 < begin*SR <= ends[birth]+1e-5:
            onset[birth,col]=1
        ending=int(np.searchsorted(ends,end*SR,side='left'))
        if 0<=ending<len(ends) and ends[ending]-1024-1e-5 < end*SR <= ends[ending]+1e-5:
            release[ending,col]=1
    count=np.minimum(count,6)
    states=np.where(active,2,0).astype(np.uint8)
    states[release>0]=3
    states[onset>0]=1
    return dict(count=count,active=active,onset=onset,offset=release,full_note=states)

# lines 76-90
def select_train_rows(f,lab,cap,seed):
    xx=np.asarray(f,np.float32)
    yy=np.asarray(lab,np.uint8)
    rng=np.random.default_rng(seed)
    classes=np.unique(yy)
    limit=max(1,cap//len(classes))
    inds=[]
    for cl in classes:
        candidates=np.flatnonzero(yy==cl)
        if len(candidates)>limit:candidates=rng.choice(candidates,limit,replace=False)
        inds.extend(candidates.tolist())
    inds=np.sort(np.asarray(inds,dtype=int))
    return xx[inds].copy(),yy[inds].copy()

# lines 106-122
    labels=truth_arrays(meta['fit_truth'],decision)
    feats={}
    for axis in AXES:
        if axis in ('exact_k','active_exact_k'):
            features=count_features(x)
            yy=labels['count']
            xs,ys=select_train_rows(features,yy,COUNT_CAP,SEED+31*ix)
        else:
            features=x[:,:,FEATURE_COLUMNS[axis]].reshape(-1,len(FEATURE_COLUMNS[axis]))
            yy=labels['active' if axis=='frame_pitch' else axis].reshape(-1)
            xs,ys=select_train_rows(features,yy,TRAIN_CAP,SEED+31*ix)
        feats[axis]=(xs,ys)

# lines 124-144
def fit_axis(rows,axis,hold):
    selection=[r for r in rows if r['composition']!=hold]
    comps=sorted({r['composition'] for r in selection})
    if len(comps)!=4 or hold in comps:raise AssertionError('leakage into training')
    xs=np.concatenate([r['features'][axis][0] for r in selection])
    ys=np.concatenate([r['features'][axis][1] for r in selection])
    classes=np.unique(ys)
    if len(classes)<2:raise AssertionError('not enough classes')
    model=make_pipeline(StandardScaler(), LogisticRegression(
        C=.18,solver='lbfgs',max_iter=68,
        class_weight='balanced' if axis!='active_exact_k' else None,
        random_state=SEED))
    weights=np.where(ys>0,3.,1.) if axis=='active_exact_k' else np.ones(len(ys))
    model.fit(xs,ys,logisticregression__sample_weight=weights)
    return model,audit

# lines 146-158
def sample_predictions(x,model,axis):
    if axis in ('exact_k','active_exact_k'):
        f=count_features(x);v=model.predict(f).astype(np.uint8)
        return v
    cols=FEATURE_COLUMNS[axis]
    f=x[:,:,cols].reshape(-1,len(cols))
    if axis!='full_note':
        proba=model.predict_proba(f)
        index=list(model.classes_).index(1)
        return (proba[:,index]>=.50).reshape(x.shape[:2]).astype(np.uint8)
    return model.predict(f).reshape(x.shape[:2]).astype(np.uint8)

# lines 160-176
def counts_for_axis(axis,pred,labels):
    if axis=='exact_k':
        return Counter(frames=len(pred),correct=int((pred==labels['count']).sum()))
    if axis=='active_exact_k':
        mask=labels['count']>0
        return Counter(active_frames=int(mask.sum()),correct=int((pred[mask]==labels['count'][mask]).sum()))
    if axis=='frame_pitch':
        target=labels['active'].astype(bool);p=pred.astype(bool)
    elif axis=='full_note':
        target=labels['full_note'];p=pred
        return Counter(correct=int((p==target).sum()),total=int(target.size),
          on_tp=int(((p==1)&(target==1)).sum()),on_fp=int(((p==1)&(target!=1)).sum()),
          on_fn=int(((p!=1)&(target==1)).sum()))
    else:
        target=labels[axis].astype(bool);p=pred.astype(bool)
    return Counter(tp=int((p&target).sum()),fp=int((p&~target).sum()),fn=int((~p&target).sum()))
```

These are bounded excerpts; complete training file remains in S810 ZIP. Every model gets EXACTLY this document plus the same requested formal reasoning protocol. The user has not supplied or opened reserved CAL32/HELD/TEST data for this review.
