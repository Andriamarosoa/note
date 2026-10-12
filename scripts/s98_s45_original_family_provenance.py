"""S98: audit provenance of the 16 ORIGINAL S45 proposal families.

Never infer that a historical prediction vector is an executable
new-audio producer. No audio labels, test outcomes, or promotion decisions.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

def inspect(report_path,output):
    record=json.loads(report_path.read_text())
    if record.get('status')!='completed':
        raise ValueError('incomplete original S44 champion manifest')
    raw=record.get('family_champions',[])
    candidates=[c for c in raw if c.get('kind')=='gross']
    mapping={}
    for c in candidates:
        fam=str(c['family'])
        if fam in mapping: raise ValueError('ambiguous family '+fam)
        mapping[fam]={'original_family':fam,'original_variant':str(c['variant']),
                      'original_vector_sha256':str(c['sha256']),
                      'historical_vector_found':True,
                      'new_audio_source_executable':False,
                      'source_checkpoints_verified':False}
    if len(mapping)!=16: raise ValueError('S45 original 16 families missing: '+str(len(mapping)))
    if any('h9' in fam.lower() or 'yourmt3' in fam.lower() for fam in mapping):
        raise ValueError('prohibited family in historical S45')
    result={'status':'S98_S45_16_ORIGINAL_PROPOSAL_PROVENANCE_MAPPED',
            'original_families':mapping,'families_found':len(mapping),
            'source_runtimes_verified':0,
            'new_audio_proposals_generated':False,
            'can_run_S80_externally':False,'independent_score':None,'promotion':False,
            'note':'A champion is a frozen prediction vector; each source producer needs its actual audio inference pipeline.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'champions':[
         {'family':k,'variant':v['original_variant'],'sha256':v['original_vector_sha256']}
         for k,v in mapping.items()], 'verified_external_producers':0},indent=2))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--s44-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    inspect(a.s44_report,a.output)
