"""Keep trained parameters, provenance and the historical inventory beyond CI retention.

The outcome memory preserves all native decisions. This companion archive keeps
the available weights and reports, without duplicating large prediction tensors.
It is not a claim that every historical model has an executable adapter.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

from scripts.build_v273_variant_memory import CORE_SOURCES
from scripts.verify_v273_selector_repair_artifacts import digest, require


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();require(not args.output.exists(),'refusing to overwrite source archive')
    args.output.mkdir(parents=True)
    manifest=dict(scope='Available weights, reports and calibration files from 17 measured source archives; two producer caches; fetched Git history inventory.',
        large_prediction_tensors='Referenced by SHA256 in variants/registry.json; native decisions are preserved in variants/all-variant-decisions.npz.',
        complete_historical_model_backup=False,files={})
    bundles=[('model-replay-bundles.zip',[(name,d) for name,d,_ in CORE_SOURCES],{'predictions.npz'}),
             ('upstream-producers.zip',[('specialists','group127-cached-results/producer-cache'),('learned_corrector','catalogue-results/corrector-cache')],set())]
    for filename,directories,excluded in bundles:
        members={};output=args.output/filename
        with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
            for name,directory in directories:
                source=args.artifacts_root/directory;require(source.is_dir(),'missing source '+str(source))
                files=[f for f in sorted(source.rglob('*')) if f.is_file() and f.name not in excluded]
                require(files,'empty source '+name)
                for f in files:
                    path=name+'/'+f.relative_to(source).as_posix();data=f.read_bytes()
                    info=zipfile.ZipInfo(path,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
                    archive.writestr(info,data,compresslevel=9)
                    members[path]=dict(source=f.relative_to(args.artifacts_root).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        with zipfile.ZipFile(output) as archive:
            require(set(archive.namelist())==set(members),'archive members')
            for name,info in members.items():
                data=archive.read(name);require(len(data)==info['bytes'] and hashlib.sha256(data).hexdigest()==info['sha256'],'unchanged archived source')
        manifest['files'][filename]=dict(bytes=output.stat().st_size,sha256=digest(output),members=members,round_trip_verified=True)
    history=args.artifacts_root/'audit-memory-results/history'
    source=history/'history-head-registry.json';data=source.read_bytes();entries=json.loads(data)
    summary=json.loads((history/'summary.json').read_text())
    require(len(entries)==summary['entries'] and not summary['unreadable_refs'],'history inventory completeness within fetched refs')
    require(len({e['head_id'] for e in entries})==len(entries),'history entry identities')
    branch_entries=[e for e in entries if e['head_id'].startswith('HIST-')]
    require(all(len(e['source_blob_sha'])==40 for e in branch_entries),'branch blob provenance')
    output=args.output/'history-head-registry.json.gz';output.write_bytes(gzip.compress(data,compresslevel=9,mtime=0))
    require(gzip.decompress(output.read_bytes())==data,'unchanged history inventory')
    manifest['files'][output.name]=dict(bytes=output.stat().st_size,sha256=digest(output),decompressed_sha256=digest(source),
        entries=len(entries),branch_file_entries=len(branch_entries),round_trip_verified=True)
    output=args.output/'history-summary.json';output.write_bytes((history/'summary.json').read_bytes())
    manifest['files'][output.name]=dict(bytes=output.stat().st_size,sha256=digest(output))
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({name:{k:v for k,v in info.items() if k!='members'} for name,info in manifest['files'].items()}),flush=True)


if __name__=='__main__':
    main()
