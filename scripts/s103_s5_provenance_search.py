"""S103 locate the exact S5 training/inference definitions.

Identify immutable upstream model provenance. Source matches do not imply
runnable external audio model or S80 validation.
"""
from __future__ import annotations
import json,subprocess,argparse
from pathlib import Path
ROOT_COMMIT='854ada718ac9bf1556f00c96b8e1aa90402d6ab3'
NEEDLES=('open_k0k6_series5__prior_corrected','designs_sha256',
        'prior_corrected','tilt1.5','series5/models','series5__raw')
def run(out):
    results={}
    for term in NEEDLES:
        cmd=['git','grep','-n','-F',term,ROOT_COMMIT,'--','scripts','.github/workflows']
        p=subprocess.run(cmd,capture_output=True,text=True)
        if p.returncode not in (0,1):raise RuntimeError(p.stderr)
        matches=[s for s in p.stdout.splitlines() if not s.startswith('analysis/evidence/')][:60]
        results[term]=matches
    data={'status':'S103_S5_SOURCE_PROVENANCE_SEARCH','matches':results,
          'candidate_source_file_count':len({line.split(':')[1] for vv in results.values() for line in vv}),
          'original_model_equivalent_external':False,'S80_promoted':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':data['status'],
       'source_lines':{k:v[:10] for k,v in results.items()}},
       indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
