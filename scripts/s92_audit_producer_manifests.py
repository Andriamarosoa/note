"""S92 inspect source manifests only, without reading labels or model outputs."""
from zipfile import ZipFile
from pathlib import Path
import json, argparse
def run(out):
    root=Path("analysis/evidence/v273-audit-memory/source-archives")
    with ZipFile(root/"upstream-producers.zip") as z:
        names=z.namelist()
        manifests={n:json.loads(z.read(n)) for n in names if n.endswith("manifest.json")}
    summary={"status":"S92_SOURCE_MANIFEST_AUDIT","manifests":manifests,
             "is_confirmed_S45":False,"promoted":False,"independent_score":None}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    run(p.parse_args().out)
