"""S147: exact label-free original S80 K0/K2 final stage.
Requires upstream original S72 and S79 decisions, PR16 36 votes/12 probabilities,
S46 193 votes. Historical fixture extraction does NOT copy or read true K.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
FIELDS={"row_id","s72_K","s79_K","pr16_K","pr16_probs","s46_K"}
def req(c,m):
    if not c:raise ValueError(m)
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inputs(p):
    with np.load(p,allow_pickle=False) as z:
        req(set(z.files)==FIELDS,"only six source-only fields allowed")
        x={k:z[k].copy() for k in FIELDS}
    n=len(x["row_id"])
    req(n>0 and len(np.unique(x["row_id"]))==n,"native unique IDs")
    b=x["s72_K"];a=x["s79_K"];p=x["pr16_K"];q=x["pr16_probs"];v=x["s46_K"]
    req(b.shape==a.shape==(n,) and p.shape==(n,36) and
        q.shape==(n,12,7) and v.shape==(n,193),"original shapes")
    req(all(np.isin(z,range(7)).all() for z in (b,a,p,v)),"K0..K6")
    req(np.isfinite(q).all() and (q>=0).all() and
        np.allclose(q.sum(2),1,atol=2e-4),"normalized PR16")
    req(np.array_equal(b[b!=1],a[b!=1]),"S79 can only change K1")
    return x
def tail(b,s79,pr,ps,s46):
    n=len(b)
    avg=ps.mean(1)
    v=np.stack([(pr==k).sum(1) for k in range(7)],1)/36.
    sv=np.stack([(s46==k).sum(1) for k in range(7)],1)/193.
    mix=.7*avg+.2*v+.1*sv
    q0=avg.argmax(1);q2=sv.argmax(1)
    a0=(b==0)&(q0>=2)
    a2=(b==2)&(q2!=b)&(mix[np.arange(n),q2]>=mix[np.arange(n),b]+.4)
    req(not np.any((s79!=b)&(a0|a2)),"overlap with S79 K1 route")
    out=s79.copy();out[a0]=q0[a0];out[a2]=q2[a2]
    return out,a0,a2
def extract(a):
    req(not a.out.exists(),"immutable historical fixture")
    with np.load(a.s80,allow_pickle=False) as z:
        ids=z["global_index"].copy();b=z["s72_K"].copy();k1=z["s79_K"].copy()
    with np.load(a.pr16,allow_pickle=False) as z:
        req(np.array_equal(z["global_index"],ids),"PR16 IDs")
        pr=z["predictions"].copy()
    with np.load(a.pr16_probs,allow_pickle=False) as z:
        req(np.array_equal(z["global_index"],ids),"PR16 probability IDs")
        p=z["probabilities"].copy()
    with np.load(a.s46,allow_pickle=False) as z:
        req(np.array_equal(z["global_index"],ids),"S46 IDs")
        sv=z["predictions"].copy()
    req(len(ids)==59309,"history size")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,row_id=ids,s72_K=b,s79_K=k1,
        pr16_K=pr,pr16_probs=p,s46_K=sv)
    inputs(a.out)
    print("S147 extracted truth-free historic 59309 events",sha(a.out))
def apply(a):
    req(not a.out.exists(),"immutable output")
    x=inputs(a.input)
    k,u,v=tail(x["s72_K"],x["s79_K"],x["pr16_K"],
               x["pr16_probs"],x["s46_K"])
    a.out.mkdir(parents=True)
    dest=a.out/"predictions.npz"
    np.savez_compressed(dest,row_id=x["row_id"],s80_K=k,
                        action_K0=u,action_K2=v)
    report={"status":"S147_LABEL_FREE_S80_K0_K2_TAIL",
            "rows":len(k),"K0_actions":int(u.sum()),"K2_actions":int(v.sum()),
            "input_sha256":sha(a.input),"output_sha256":sha(dest),
            "original_S80_full_external":False,"promoted":False}
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))
def verify(a):
    req(not a.out.exists(),"immutable proof")
    with np.load(a.s80,allow_pickle=False) as z:
        ids=z["global_index"].copy()
        ref={k:z[k].copy() for k in ("s80_K","action_K0","action_K2")}
    with np.load(a.replay,allow_pickle=False) as z:
        req(set(z.files)=={"row_id","s80_K","action_K0","action_K2"} and
            np.array_equal(z["row_id"],ids),"label-free replay alignment")
        errors={k:int(np.sum(z[k]!=ref[k])) for k in ref}
    req(errors=={"s80_K":0,"action_K0":0,"action_K2":0},
        "original tail exact replay failed: "+str(errors))
    req(int(ref["action_K0"].sum())==46 and
        int(ref["action_K2"].sum())==61,"historical action integrity")
    report={"status":"S147_EXACT_ORIGINAL_TAIL_REPLAY",
            "historical_rows":len(ids),"mismatches":errors,
            "K0_actions":46,"K2_actions":61,
            "true_K_read":False,"S80_new_audio_accuracy":None,"promoted":False}
    a.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))
def selftest():
    b=np.array([0,2,2,1,3],np.int8);s79=b.copy();s79[3]=2
    pr=np.zeros((5,36),np.int8);v=np.zeros((5,193),np.int8)
    p=np.zeros((5,12,7),float);p[:,:,0]=1
    p[0,:,0]=0;p[0,:,2]=1;v[1,:]=3;v[2,:]=2
    out,u,w=tail(b,s79,pr,p,v)
    req(out.tolist()==[2,3,2,2,3] and u.sum()==w.sum()==1,
        "S80 K0/K2 math drift")
    print("S147 PASS: original K0/K2 exact source math; no labels")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    sub=p.add_subparsers(dest="op")
    for op,fields in (("extract",("s80","pr16","pr16-probs","s46","out")),
                      ("apply",("input","out")),
                      ("verify",("s80","replay","out"))):
        q=sub.add_parser(op)
        for k in fields:q.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    if a.self_test:selftest()
    elif a.op=="extract":extract(a)
    elif a.op=="apply":apply(a)
    elif a.op=="verify":verify(a)
    else:p.error("choose extract/apply/verify")
