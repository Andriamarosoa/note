"""S104: recover removed S5 527-feature producer from historical git patches.

Search deleted as well as live source; return exact commit references and
matching code lines. A match is NOT validation of new-audio prediction.
"""
import subprocess,json,argparse
from pathlib import Path
TERMS=('designs_sha256','27407','series5/models','527','prior_corrected')
def git(*args):
    p=subprocess.run(['git',*args],capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stderr[-2000:])
    return p.stdout
def main(out):
    result={}
    for needle in TERMS:
        sha=git('log','--all','-G',needle,'--format=%H','--','scripts','.github/workflows').splitlines()
        commits=list(dict.fromkeys(x for x in sha if len(x)==40))[:30]
        details=[]
        for h in commits[:15]:
            changed=git('show','--format=','--name-only',h).splitlines()
            source=[x for x in changed if x.startswith(('scripts/','.github/workflows/'))]
            if source:details.append({'commit':h,'sources':source[:10]})
        result[needle]={'commit_count_capped':len(commits),'evidence':details}
    out.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'S104_DELETED_ORIGINAL_S5_SOURCE_HISTORY_AUDIT','search':result,
            'recovered_executable_training_feature_builder':False,
            'external_S80_validated':False,'promotion':False}
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
