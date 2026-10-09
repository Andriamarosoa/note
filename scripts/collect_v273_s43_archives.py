"""S43: recover every declared historical series decision archive, never weights only.

GitHub Actions artifact collection is deterministic and has an explicit
missing-source manifest. This script does not interpret or train models.
"""
from __future__ import annotations
import argparse,json,os,subprocess,shutil,time
from pathlib import Path

RUNS={
"S09":[37853130781],"S10":[37853361662],"S11":[37853663514],
"S12":[37853979683],"S13":[37854384791],"S14":[37855070118],
"S15":[37855272619],"S16":[37855705158],"S17":[37856164138],
"S18":[37856499212],"S19":[37858017067],"S20":[37858383973],
"S21":[37859454327],"S22":[37859786345],
"S23":[37859970897],"S24":[37862241264],"S25":[37862417522],
"S26":[37862754589],"S27":[37864168525],"S28":[37864237149],
"S29":[37865173636],"S30":[37865974760],"S31":[37866124959],
"S32":[37866372090],"S33":[37866555589],"S34":[37866775260],
"S35":[37868356632],"S36a":[37869563550],
"S36b":[37869664139],"S37":[37870170306],
"S38":[37880386031],"S39":[37880791041],
"S40":[37880762289],"S41":[37881191015]
}
# Audits without predicted vector still count as inspected sources, and
# explicit source provenance is preserved in all manifests.
REPORT_ONLY={'S10','S23','S33','S39'}
ALLOWED_FILES={'predictions.npz','prediction.npz',
               'report.json','report.md',
               'all-variant-decisions.npz',
               'all-candidate-decisions.npz',
               'appended-candidates.npz'}
MAX_UNZIPPED=150*1024*1024

def shell(argv,timeout=180):
    p=subprocess.run(argv,text=True,capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError((p.stderr or p.stdout).strip()[-1000:])
    return p.stdout

def recover(a):
    out=a.output
    out.mkdir(parents=True,exist_ok=True)
    manifest=dict(version='S43-archive-retrieval-v1',created_by='GitHub Actions',
      branch='codex/v273-open-k0-k6',required_series=list(RUNS),
      report_only_sources=list(REPORT_ONLY),sources=[])
    for series,runs in RUNS.items():
        dest=out/series
        dest.mkdir()
        item=dict(series=series,run_ids=runs,downloaded_artifacts=[],
                  prediction_archives=[],reports=[],status='pending')
        try:
            for run in runs:
                api=f'/repos/{a.repo}/actions/runs/{run}/artifacts?per_page=100'
                arts=json.loads(shell(['gh','api',api],timeout=60)).get('artifacts',[])
                ready=[z for z in arts if not z.get('expired')]
                if not ready:
                    item.setdefault('warnings',[]).append(f'run {run} has no unexpired artifacts')
                    continue
                for artifact in ready:
                    name=artifact['name']
                    target=dest/name
                    target.mkdir()
                    try:
                        shell(['gh','run','download',str(run),'--repo',a.repo,
                               '--name',name,'--dir',str(target)],
                              timeout=600)
                        item['downloaded_artifacts'].append(dict(
                            id=artifact['id'],name=name,
                            size_bytes=artifact.get('size_in_bytes'),
                            run_id=run))
                        files=list(target.rglob('*'))
                        for file in files:
                            if not file.is_file():continue
                            kind=file.name
                            if kind in ALLOWED_FILES:
                                if file.stat().st_size>MAX_UNZIPPED:
                                    item.setdefault('warnings',[]).append(
                                        f'oversize extracted {file.name}')
                                    file.unlink()
                                elif kind.endswith('.npz'):
                                    item['prediction_archives'].append(str(file.relative_to(out)))
                                else:item['reports'].append(str(file.relative_to(out)))
                            else:
                                file.unlink()  # discard large stored model/checkpoint weights
                        # Remove empty folders, keep provenance and reports.
                    except Exception as e:
                        item.setdefault('warnings',[]).append(
                            f'artifact {name}: {type(e).__name__}: {str(e)[:240]}')
            item['status']='downloaded' if item['prediction_archives'] else (
                'report_only' if item['reports'] else 'missing_or_unavailable')
        except Exception as e:
            item['status']='error'
            item['error']=f'{type(e).__name__}: {str(e)[:400]}'
        manifest['sources'].append(item)
        (out/'source_manifest.json').write_text(
            json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(dict(series=series,status=item['status'],
            files=item['prediction_archives'], warnings=item.get('warnings',[]))),
            flush=True)
    good=sum(bool(x['prediction_archives']) for x in manifest['sources'])
    assert good>=15,('not enough recoverable original predictions; '
                      'mark failure rather than claim comprehensive audit')
    print(f'S43 archive retrieval: {good}/{len(RUNS)} source series with predictions',
           flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repo',default=os.environ.get('GITHUB_REPOSITORY',
                                                    'Andriamarosoa/note'))
    recover(p.parse_args())
