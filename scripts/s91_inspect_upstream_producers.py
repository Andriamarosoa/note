"""S91: inspect nested original producer NPZ schemas, fail closed."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import numpy as np, json, argparse
def inspect(outer, inner):
    with ZipFile(outer) as z:
        if inner not in z.namelist(): return {"found":False}
        b=z.read(inner)
    with np.load(BytesIO(b),allow_pickle=False) as a:
        records=[]
        for key in a.files:
            try:
                v=a[key]
                records.append({"key":key,"shape":list(v.shape),"dtype":str(v.dtype),
                    "sample":v.reshape(-1)[:3].tolist() if v.dtype.kind in "USbiuf" and v.size and v.size<100000 else None})
            except ValueError as exc:
                records.append({"key":key,"error":str(exc)})
    return {"found":True,"bytes":len(b),"arrays":records}
def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    base=Path("analysis/evidence/v273-audit-memory/source-archives")
    with ZipFile(base/"upstream-producers.zip") as z:
        names=z.namelist()
    result={"archive_members":names,"producer":inspect(base/"upstream-producers.zip","specialists/producers.npz"),
            "independent_validation":False,"promotion":False}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,default=str))
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__":main()
