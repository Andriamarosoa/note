"""Append every candidate and losslessly preserve model state in manageable git blobs."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
from scripts.evaluate_v273_regression_loops import read, dump
from scripts.verify_v273_selector_repair_artifacts import require, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1];work=a.workspace.resolve();a.output=a.output.resolve()
    require(not a.output.exists(),'refusing to replace candidate history');a.output.mkdir(parents=True)
    sources=[work/'regression-loops-prepared']+[work/('regression-loops-series'+str(i)) for i in range(1,5)]
    copied=[]
    for i,source in enumerate(sources):
        target=a.output/('prepared' if i==0 else 'series'+str(i));target.mkdir()
        for file in sorted(source.iterdir()):
            if file.name=='models.joblib':continue
            if file.name=='models.json':
                dest=target/'models.json.gz';dest.write_bytes(gzip.compress(file.read_bytes(),mtime=0))
                require(gzip.decompress(dest.read_bytes())==file.read_bytes(),'lossless linear model archive')
            else:
                dest=target/file.name;shutil.copyfile(file,dest)
            copied.append(dict(source='prepared/'+file.name if i==0 else 'series'+str(i)+'/'+file.name,
                               source_sha256=digest(file),stored=str(dest.relative_to(a.output)),stored_sha256=digest(dest)))
    model_archive=[]
    for i in (3,4):
        original=work/('regression-loops-series'+str(i))/'models.joblib'
        compact=work/('regression-loops-models-series'+str(i)+'.joblib')
        old=joblib.load(original);new=joblib.load(compact)
        require(joblib.hash(old)==joblib.hash(new),'identical complete fitted estimator state')
        payload=compact.read_bytes();parts=[]
        for j,start in enumerate(range(0,len(payload),3500000)):
            part=a.output/f'series{i}'/f'models-xz.joblib.part{j:02d}'
            part.write_bytes(payload[start:start+3500000]);parts.append(dict(file=str(part.relative_to(a.output)),bytes=part.stat().st_size,sha256=digest(part)))
        require(hashlib.sha256(b''.join((a.output/v['file']).read_bytes() for v in parts)).hexdigest()==digest(compact),'complete model shards')
        model_archive.append(dict(series=i,original_serialization_sha256=digest(original),
            compact_sha256=digest(compact),compact_bytes=len(payload),joblib_state_hash=joblib.hash(old),
            numerical_and_estimator_state_unchanged=True,serialization_bytes_changed=True,parts=parts))
    dump(a.output/'storage-manifest.json',dict(files=copied,model_archives=model_archive,
        restore='Decompress models.json.gz; concatenate each ordered models-xz.joblib.part* into a .joblib file. joblib.load restores the complete fitted state.'))
    verify=json.loads((work/'regression-loops-final-verification.json').read_text())
    dump(a.output/'verification.json',verify)
    old1=repo/'analysis/evidence/v273-audit-memory/variants/all-variant-decisions.npz'
    old2=repo/'analysis/evidence/v273-contextual-risk/preserved-candidates.npz'
    previous_registry=repo/'analysis/evidence/v273-contextual-risk/candidate-registry.json'
    require(json.loads(previous_registry.read_text())['total_candidates']==46,'previous memory count')
    d1,d2=read(old1),read(old2);matrices=[d1['predictions'],d2['predictions']]
    names=list(map(str,d1['variant_ids']))+list(map(str,d2['variant_ids']));new_entries=verify['entries']
    for i in range(1,5):
        d=read(a.output/f'series{i}'/'predictions.npz')
        require(np.array_equal(d['global_index'],d1['global_index']),'history alignment')
        matrices.append(d['predictions']);names.extend('regression_loops_s'+str(i)+'__'+str(k) for k in d['variant_ids'])
    matrix=np.column_stack(matrices).astype(np.int8);y=d1['true_K'];b=d1['frozen_baseline_K']
    require(len(names)==len(set(names))==208 and matrix.shape==(59309,208),'complete unique candidate identifiers')
    require(np.array_equal(matrix[:,:41],d1['predictions']) and np.array_equal(matrix[:,41:46],d2['predictions']),'old predictions intact')
    values={};aliases={}
    for i,name in enumerate(names):
        sha=hashlib.sha256(matrix[:,i].copy().tobytes()).hexdigest();aliases[name]=values.get(sha);values.setdefault(sha,name)
    for entry in new_entries:
        j=names.index(entry['variant_id']);entry['prediction_values_sha256']=hashlib.sha256(matrix[:,j].copy().tobytes()).hexdigest()
        entry['identical_prediction_alias_of']=aliases[entry['variant_id']]
    common={k:d1[k] for k in ('global_index','true_K','frozen_baseline_K','fold','eligible_global_index')}
    np.savez_compressed(a.output/'all-candidate-decisions.npz',**common,variant_ids=np.array(names),predictions=matrix,
        outcomes_vs_freeze=(matrix==y[:,None]).astype(np.int8)-(b==y)[:,None].astype(np.int8))
    old_union=(b!=y)&(matrix[:,:46]==y[:,None]).any(1);union=(b!=y)&(matrix==y[:,None]).any(1)
    registry=dict(previous_registry='../v273-contextual-risk/candidate-registry.json',previous_registry_sha256=digest(previous_registry),
        previous_native_memories=[dict(path=str(v.relative_to(repo)),sha256=digest(v)) for v in (old1,old2)],
        previous_candidates=46,added_candidates=new_entries,total_candidates=208,distinct_prediction_vectors=len(values),
        all_candidates_file='all-candidate-decisions.npz',all_candidates_sha256=digest(a.output/'all-candidate-decisions.npz'),
        original_columns_unchanged=True,all_negative_and_duplicate_candidates_retained=True,active_heads_added=0,
        union_initial_errors_corrected=int(union.sum()),previous_union=int(old_union.sum()),
        additional_union=int((union&~old_union).sum()),oracle_union_not_prediction=True,
        downstream_ready=False,independent_validation=False,promotion=False,
        downstream_requirement='Regenerate nested predictions for each consumer; preserve the documented supervised calibration split.')
    dump(a.output/'candidate-registry.json',registry)
    files=[]
    for file in sorted(a.output.rglob('*')):
        if file.is_file():files.append(dict(path=str(file.relative_to(repo)),bytes=file.stat().st_size,sha256=digest(file),
            git_blob_sha1=hashlib.sha1(b'blob '+str(file.stat().st_size).encode()+b'\0'+file.read_bytes()).hexdigest()))
    dump(work/'regression-loops-publish-files.json',files)
    print(json.dumps(dict(files=len(files),bytes=sum(v['bytes'] for v in files),candidates=208,
        distinct=len(values),old_union=int(old_union.sum()),new_union=int(union.sum()))),flush=True)


if __name__=='__main__':main()
