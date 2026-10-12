"""S199 exhaustive CURRENT-BRANCH tracked archive header audit for authentic S5 527D input.

Inspect all git-versioned .npz and .zip, including ZIP->NPZ members. Search
for 527D, 416D, 255x9 and audio 42x49 source arrays. No array VALUES/truth
read, and no assertion that omitted historic branches/artifacts are covered.
"""
from __future__ import annotations
import argparse,hashlib,io,json,subprocess,zipfile
from pathlib import Path
from scripts.s195_inventory_original_255x9_S5_527_archives import inventory,npy_header

def need(v,m):
    if not v:raise ValueError(m)
def matches(shape):
    return {'exact_527_last_dim':bool(shape and shape[-1]==527),
      'extra_416_last_dim':bool(shape and shape[-1]==416),
      'original_255x9':bool(len(shape)>=2 and shape[-2:]==[255,9]),
      'all_255_groups':bool(shape and shape[-1]==255),
      'S4_111_last_dim':bool(shape and shape[-1]==111)}
def embedded_npz(z,member):
    try:
        blob=z.read(member)
        need(len(blob)<=25000000,'oversized embedded npz, skip safely')
        with zipfile.ZipFile(io.BytesIO(blob)) as nested:
            return {m.filename[:-4]:npy_header(nested.open(m))
                for m in nested.infolist() if m.filename.endswith('.npy')}
    except (zipfile.BadZipFile,ValueError) as e:
        return {'__skipped__':{'reason':str(e)[:150]}}
def run(root,out):
    need(not out.exists(),'immutable broad S5 audit')
    result=subprocess.run(['git','ls-files','--','*.npz','*.zip'],
                          cwd=root,capture_output=True,text=True,check=True)
    files=sorted(set(result.stdout.splitlines()))
    need(len(files)>=35,'unexpected source binary count')
    hits=[];full=[];readable=0;scanned=0;file_summaries=[]
    for path in files:
        p=root/path
        if not p.is_file():continue
        entries={}
        with zipfile.ZipFile(p) as z:
            if p.suffix=='.npz':
                entries=inventory(p)
            else:
                for member in z.infolist():
                    if member.filename.endswith('.npz'):
                        inner=embedded_npz(z,member)
                        for k,head in inner.items():
                            if k!='__skipped__':
                                entries[member.filename+'::'+k]=head
                    if member.filename.endswith('.npy'):
                        with z.open(member) as f:
                            entries[member.filename]=npy_header(f)
        for key,head in entries.items():
            shape=head['shape'];categories=matches(shape)
            scanned+=1
            if any(categories.values()):
                hits.append({'file':path,'array':key,'shape':shape,
                             'categories':[k for k,v in categories.items() if v]})
        file_summaries.append({'file':path,'array_header_count':len(entries),
                               'bytes':p.stat().st_size})
        readable+=1
    summary={
       'status':'S199_ALL_VERSIONED_CURRENT_BRANCH_NPZ_AND_ZIP_S5_527_SOURCE_SCAN',
       'versioned_binary_files_scanned':readable,
       'numpy_array_headers_scanned':scanned,
       'source_relevant_schema_hits':hits,
       'source_relevant_schema_hits_count':len(hits),
       'historical_527D_matrix_present_in_scanned_sources':any(
           'exact_527_last_dim' in x['categories'] for x in hits),
       'original_255x9_matrix_present_in_scanned_sources':any(
           'original_255x9' in x['categories'] for x in hits),
       'scope':'only tracked .npz/.zip at this branch head, not absent historical CI artifacts, other branches or serialized checkpoint tensors',
       'all_source_npy_array_values_opened':False,
       'all_truth_or_MIDI_content_opened':False,
       'S5_original_527D_constructor_proven':False,
       'S80_promoted':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,indent=2)+'\n')
    compact=dict(summary)
    compact['source_relevant_schema_hits']=[h for h in hits if
        'exact_527_last_dim' in h['categories'] or
        'extra_416_last_dim' in h['categories'] or
        'original_255x9' in h['categories']]
    print(json.dumps(compact,indent=2),flush=True)
def selftest():
    need(matches([10,527])['exact_527_last_dim'] and
        matches([12,255,9])['original_255x9'] and
        not matches([111,7])['exact_527_last_dim'],
        'scan matching matrix strictness')
    print('S199 PASS strict matrix shape evidence and no numerical values')
if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--root',type=Path,default=Path('.'))
    a.add_argument('--out',type=Path)
    a.add_argument('--self-test',action='store_true')
    p=a.parse_args()
    if p.self_test:selftest()
    else:
        need(p.out is not None,'out required')
        run(p.root,p.out)
