"""S118: experimental 527-D S4-owned/S5 acoustic constructor (NOT historical-equivalent).

Reuses S116's original 111D control prefix and forms 416 *candidate*
pitch-agnostic flow/activation descriptors from the original 42x49 trajectory.
The exact original S4-owned 416D code and column order are NOT recovered.
Never use this candidate to promote S80 or report independent accuracy.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

LAGS = (1, 2, 4, 8)
ZONES = ("owned", "dilated2", "before", "after")
EPS = 1e-12
BAD = {"true_K", "true_k", "truth", "target", "label", "labels", "midi", "y"}

def require(value, message):
    if not value:
        raise ValueError(message)

def conc(v):
    total = v.sum()
    return float(((v / total)**2).sum()) if total > EPS else 0.

def pitch15(a):
    v = np.maximum(np.asarray(a, dtype=float), 0.)
    require(v.shape == (49,) and np.isfinite(v).all(), "bad pitch vector")
    ranked = np.sort(v)[::-1]
    c = conc(v)
    peak = ranked[0]
    result = np.r_[ranked[:8], v.sum(), c, 1/c if c > EPS else 0.,
                     [float((v > t*peak).sum()) if peak > EPS else 0.
                      for t in (.02,.05,.10,.20)]]
    require(result.shape == (15,), "pitch15 contract")
    return result

def time5(v):
    v = np.maximum(np.asarray(v, dtype=float), 0.)
    if not len(v):
        return np.zeros(5, float)
    peak = v.max()
    return np.array([v.mean(), v.std(), peak, conc(v),
                     (v > .05*peak).mean() if peak > EPS else 0.])

def regions(mask):
    mask = np.asarray(mask, bool)
    require(mask.shape == (42,), "bad ownership mask")
    broad = mask.copy()
    for d in (1,2):
        broad[d:] |= mask[:-d]
        broad[:-d] |= mask[d:]
    positions = np.flatnonzero(mask)
    before = np.arange(42) < positions[0] if len(positions) else np.zeros(42,bool)
    after = np.arange(42) > positions[-1] if len(positions) else np.zeros(42,bool)
    return dict(owned=mask,dilated2=broad,before=before,after=after)

def owned416(log_sequence, owner_mask):
    log_sequence = np.asarray(log_sequence,dtype=np.float64)
    require(log_sequence.shape == (42,49) and np.isfinite(log_sequence).all()
            and log_sequence.min() >= -1e-6, "original 42x49 log-activation required")
    A = np.maximum(np.expm1(log_sequence),0.)
    chunks=[]
    for zone in ZONES:
        mask=regions(owner_mask)[zone]
        for lag in LAGS:
            diff=(A[lag:]-A[:-lag])[mask[lag:]]
            pos=np.maximum(diff,0.)
            neg=np.maximum(-diff,0.)
            b=np.r_[pitch15(pos.sum(axis=0)),time5(pos.sum(axis=1)),neg.sum()]
            require(b.shape == (21,), "lagged-flow schema drift")
            chunks.append(b)
        activation=A[mask]
        b=np.r_[pitch15(activation.sum(axis=0)),time5(activation.sum(axis=1))]
        require(b.shape == (20,), "activation schema drift")
        chunks.append(b)
    v=np.concatenate(chunks).astype(np.float32)
    require(v.shape == (416,) and np.isfinite(v).all(), "416D nonfinite or drift")
    return v

def selftest():
    m=np.zeros(42);m[12:17]=1
    z,n=owned416(np.zeros((42,49)),m),416
    require(z.shape == (n,) and not z.any(), "zero trajectory drift")
    seq=np.random.default_rng(118).uniform(0,2,(42,49))
    x=owned416(seq,m)
    require(np.isfinite(x).all() and np.array_equal(x,owned416(seq,m)),
            "nonfinite or unstable schema")
    require(not np.array_equal(x,owned416(seq,np.ones(42))), "ownership ignored")
    try:
        owned416(seq,np.ones(41))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid ownership accepted")
    print("S118 selftest PASS: 416D, stability, finite, ownership, invalid-shape guard")

def run(s97,s116,out,s4models=None,s5models=None):
    require(not out.exists(), "refuse overwrite")
    with np.load(s97,allow_pickle=False) as z:
        require(not BAD.intersection(z.files), "ground truth in acoustic input")
        require({"row_id","sequence42x49"}.issubset(z.files), "S97 schema")
        row_id=z["row_id"].astype(str);seq=z["sequence42x49"].copy()
    with np.load(s116,allow_pickle=False) as z:
        require(not BAD.intersection(z.files), "ground truth in control")
        require(set(z.files)=={"row_id","control111","geometry46","freeze_K"},"S116 schema")
        require(np.array_equal(row_id,z["row_id"].astype(str)),"S97/S116 identity mismatch")
        control=z["control111"].copy();geo=z["geometry46"].copy()
    n=len(row_id)
    require(n==57 and len(set(row_id))==n and control.shape==(n,111)
            and geo.shape==(n,46) and seq.shape==(n,42,49),"GOAT dimensions")
    require(np.array_equal(control[:,58:104],geo),"111D geometry prefix mismatch")
    require(np.isfinite(control).all() and np.isfinite(geo).all(), "nonfinite controls")
    require(np.isin(geo[:,:42],[0,1]).all(), "bad binary ownership")
    extension=np.stack([owned416(seq[i],geo[i,:42]) for i in range(n)])
    X=np.column_stack((control,extension)).astype(np.float32)
    require(X.shape==(57,527) and np.isfinite(X).all(), "527D candidate drift")
    out.mkdir(parents=True)
    np.savez_compressed(out/"CANDIDATE_only_527.npz",row_id=row_id,
                        control111=control,extension416=extension,candidate527=X)
    report={"status":"S118_CANDIDATE_527_AUDIO_SCHEMA_NOT_ORIGINAL",
            "event_count":n,"control_dimensions":111,"candidate_extension_dimensions":416,
            "candidate_total_dimensions":527,"ground_truth_or_MIDI_read":False,
            "S4_owned_527_exact_original_equivalence":False,
            "S5_original_527_input_equivalence":False,
            "S80_independent_accuracy":None,"S80_promoted":False}
    if (s4models is None)!=(s5models is None):
        raise ValueError("give both S4 and S5 checkpoint directories")
    if s4models is not None:
        import joblib
        from scripts.s105_s5_frozen_forward import load as s5_load, forward as s5_forward
        preds={}
        for fold in (0,1,2):
            for poly in (1,2):
                path=s4models/f"fold{fold}__owned__poly{poly}.joblib"
                require(path.is_file(), "original S4 owned checkpoint missing")
                state=joblib.load(path)
                model=state["model"]
                require(model.n_features_in_==527
                        and np.array_equal(model.classes_,np.arange(7)),
                        "S4 owned checkpoint contract")
                p=model.predict_proba(X)
                require(p.shape==(n,7) and np.isfinite(p).all()
                        and np.allclose(p.sum(1),1,atol=1e-5),"S4 owned forward")
                preds[f"S4_fold{fold}_poly{poly}"]=np.bincount(p.argmax(1),minlength=7).tolist()
        m=s5_load(s5models)
        p=s5_forward(m,X)
        for j,fold in enumerate((0,1,2,4)):
            preds[f"S5_fold{fold}"]=np.bincount(p[:,j].argmax(1),minlength=7).tolist()
        report["genuine_original_checkpoints_smoke_executed"]=10
        report["NON_VALIDATED_candidate_prediction_histograms"]=preds
        report["warning"]="Forward calls work but feature order is UNPROVEN: these are not meaningful K predictions or scores."
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--s97",type=Path)
    p.add_argument("--s116",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--s4-models",type=Path)
    p.add_argument("--s5-models",type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test: selftest()
    else:
        require(all(v is not None for v in (a.s97,a.s116,a.out)),
                "S97 and S116 and output are required")
        run(a.s97,a.s116,a.out,a.s4_models,a.s5_models)
