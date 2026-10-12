"""S99: locate source code behind all 16 original S45 family champions.

Git grep is performed against the ORIGINAL frozen S45 commit. Text
matches are candidates only; presence of code is not inference proof.
"""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
SOURCE_COMMIT='854ada718ac9bf1556f00c96b8e1aa90402d6ab3'

def grep(term):
    p=subprocess.run(['git','grep','-n','-F','--',term,SOURCE_COMMIT,'--','scripts/'],
                     capture_output=True,text=True)
    if p.returncode not in (0,1):raise RuntimeError(p.stderr)
    return p.stdout.splitlines()[:25]

def audit(original,output):
    d=json.loads(original.read_text())
    sources=[]
    families=d['original_families']
    if len(families)!=16:raise ValueError('not all 16 original S45 champion families')
    for fam,item in families.items():
        variant=item['original_variant']
        full=grep(variant)
        short=grep(variant.split('__')[0])
        hits=list(dict.fromkeys(full+short))[:30]
        paths=sorted(set(line.split(':',2)[1] for line in hits if line.count(':')>=2))
        sources.append({'family':fam,'exact_original_variant':variant,
                        'original_vector_sha256':item['original_vector_sha256'],
                        'exact_string_hits':len(full),
                        'prefix_hits':len(short),
                        'candidate_source_paths':paths,
                        'code_hits':hits[:12],
                        'source_execution_verified':False})
    out={'status':'S99_ORIGINAL_S45_SOURCE_LOCATION_AUDIT',
         'pinned_original_commit':SOURCE_COMMIT,
         'families':sources,'families_with_any_code_hit':sum(bool(x['candidate_source_paths']) for x in sources),
         'families_with_executable_external_inference_verified':0,
         'S80_validation':False,'promoted':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':out['status'],
                      'source_hits':{x['family']:x['candidate_source_paths'] for x in sources},
                      'external_producers_verified':0},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--s98',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.s98,a.out)
