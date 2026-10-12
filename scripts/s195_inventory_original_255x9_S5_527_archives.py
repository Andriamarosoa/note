"""S195 authenticated binary-schema inventory of the original 255x9 and S5 527D evidence.

Read only archive indexes/array headers. Do NOT open truth, train, infer,
or claim parity. Exact available source fields and provenance decide next gate.
"""
from __future__ import annotations
import argparse,hashlib,io,json,zipfile
from pathlib import Path
from numpy.lib import format as fmt

ROOT=Path('analysis/evidence')
NPZ=(
 'v273-regression-loops/prepared/inputs.npz',
 'v273-contextual-risk/factor-swaps.npz',
 'v273-contextual-risk/probability-evidence.npz',
 'v273-contextual-risk/preserved-candidates.npz',
 'v273-audit-memory/variants/context-fold-0-part-0.npz',
 'v273-audit-memory/variants/context-fold-1-part-0.npz',
 'v273-audit-memory/variants/all-variant-decisions.npz',
 'v273-regression-loops/series1/probabilities.npz',
)
ZIPS=(
 'v273-audit-memory/source-archives/upstream-producers.zip',
 'v273-audit-memory/source-archives/model-replay-bundles.zip',
)
def need(c,m):
    if not c:raise ValueError(m)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def npy_header(stream):
    version=fmt.read_magic(stream)
    if version==(1,0):shape,order,dtype=fmt.read_array_header_1_0(stream)
    else:shape,order,dtype=fmt.read_array_header_2_0(stream)
    return {'shape':list(shape),'dtype':str(dtype),'fortran':bool(order)}
def inventory(path):
    with zipfile.ZipFile(path) as z:
        result={}
        for member in z.infolist():
            if not member.filename.endswith('.npy'):continue
            with z.open(member) as f:head=npy_header(f)
            result[member.filename[:-4]]=head
        return result
def probe(root,out):
    need(not out.exists(),'immutable provenance')
    base=root/ROOT;datasets=[]
    for rel in NPZ:
        p=base/rel;need(p.is_file(),'missing native archived '+rel)
        contents=inventory(p)
        datasets.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,
          'sha256':sha(p),'arrays':contents,'counts':len(contents),
          'contains_exact_255_group_proposal':any(len(v['shape'])>=2 and v['shape'][1]==255
              and 'proposal' in k.lower() for k,v in contents.items()),
          'contains_exact_255x9_audits':any(len(v['shape'])==3 and v['shape'][1:]==[255,9]
              for v in contents.values()),
          'contains_saved_527D_design':any(len(v['shape'])>=2 and v['shape'][-1]==527
              for v in contents.values())})
    zipcontents={}
    for rel in ZIPS:
        p=base/rel;need(p.is_file(),'missing original archive '+rel)
        with zipfile.ZipFile(p) as z:
            entries=[]
            for m in z.infolist():
                record={'name':m.filename,'bytes':m.file_size,'compressed':m.compress_size}
                if m.filename.endswith('.npz'):
                    with z.open(m) as f:blob=f.read()
                    with zipfile.ZipFile(io.BytesIO(blob)) as nested:
                        record['array_headers']={}
                        for inner in nested.infolist():
                            if inner.filename.endswith('.npy'):
                                with nested.open(inner) as f:
                                    record['array_headers'][inner.filename[:-4]]=npy_header(f)
                entries.append(record)
        zipcontents[rel]={'bytes':p.stat().st_size,'sha256':sha(p),'members':entries}
    available=[d for d in datasets if d['contains_exact_255_group_proposal'] or d['contains_exact_255x9_audits'] or d['contains_saved_527D_design']]
    original_regression={d['path']:d['arrays'] for d in datasets if '/prepared/' in d['path']}
    report={'status':'S195_ORIGINAL_255x9_AND_S5_527_BINARY_HEADERS_FORENSIC',
       'num_versioned_binary_arrays_examined':sum(d['counts'] for d in datasets),
       'input_archive_inventories':datasets,'source_ZIP_inventories':zipcontents,
       'actual_255_9_527D_matrices_found_in_sampled_versioned_sources':[d['path'] for d in available],
       'historical_255_group_context_7x20_derived_only':
          any(v.get('shape')==[7493,7,20] for x in original_regression.values() for v in x.values()),
       'source_root':str(root),
       'MIDI_or_truth_array_values_accessed':False,'all_data_are_headers_or_member_names':True,
       'original_527_builder_identified':False,
       'original_255_proposal_audits_available_from_this_inventory':False,
       'S80_promoted':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    compact={key:report[key] for key in
      ('status','num_versioned_binary_arrays_examined',
       'actual_255_9_527D_matrices_found_in_sampled_versioned_sources',
       'historical_255_group_context_7x20_derived_only','S80_promoted')}
    compact['files']=[{'name':Path(d['path']).name,'folder':str(Path(d['path']).parent),
                       'arrays':{k:v['shape'] for k,v in d['arrays'].items()}}
                      for d in datasets]
    compact['archives']={k:[{'name':v['name'],'arrays':
                            {a:h['shape'] for a,h in v.get('array_headers',{}).items()}}
                              for v in item['members']] for k,item in zipcontents.items()}
    print(json.dumps(compact,indent=2),flush=True)
def selftest():
    import numpy as np
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as d:
        p=Path(d)/'test.npz'
        np.savez_compressed(p, proposals=np.zeros((4,255),np.int8),
                            audits=np.zeros((4,255,9),np.float32))
        x=inventory(p)
        need(x['proposals']['shape']==[4,255] and
             x['audits']['shape']==[4,255,9],'ZIP numpy headers')
    print('S195 PASS 255x9 schema without loading any values/labels')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(a.out is not None,'output required')
        probe(a.root,a.out)
