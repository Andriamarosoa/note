"""S203 external original GitHub RELEASE archive search for missing owned527 code/design.

S199 only scanned tracked Git blobs. Here inspect previously published,
immutable series41/43/44/45/46 tarball *member names*, Python source matches
and .NPZ array HEADERS (never labels/values). No trained model changes.
"""
from __future__ import annotations
import argparse,hashlib,io,json,re,tarfile,zipfile
from pathlib import Path
from scripts.s195_inventory_original_255x9_S5_527_archives import npy_header
PATTERNS={
 'owned527_builder':re.compile(r'(?:def\s+\w*(?:owned|feature|design)\w*\s*\(|extra416|owned527|designs_sha256|zone_masks)',re.I),
 's5_source_semantics':re.compile(r'(?:(?:527|416).{0,120}(?:features|construct|owned|design))',re.I),
}
def check(q,m):
    if not q:raise ValueError(m)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def headers(buf):
    found={}
    with zipfile.ZipFile(io.BytesIO(buf)) as z:
        for m in z.infolist():
            if m.filename.endswith('.npy'):
                with z.open(m) as fp:found[m.filename[:-4]]=npy_header(fp)
    return found
def examine(root,out):
    check(not out.exists(),'immutable release audit')
    files=sorted(root.rglob('*.tar.gz'))
    check(len(files)==5,'five pinned archival releases required, no skips')
    results=[]
    for file in files:
        member_count=0;noteworthy=[];matches=[];arrays=[];codefiles=0
        with tarfile.open(file,'r:gz') as tar:
            for m in tar.getmembers():
                if not m.isfile():continue
                member_count+=1
                name=m.name
                suffix=Path(name).suffix.lower()
                if any(k in name.lower() for k in ('s4','s5','series4','series5','owned','527','416','feature','input','design')):
                    noteworthy.append({'member':name,'bytes':m.size})
                if suffix in ('.py','.ipynb') and m.size<400000:
                    codefiles+=1
                    txt=tar.extractfile(m).read().decode('utf8','replace')
                    for key,pattern in PATTERNS.items():
                        hits=pattern.findall(txt)
                        if hits:
                            matches.append({'source_file':name,'pattern':key,'occurrences':len(hits),
                                            'example':hits[:3]})
                if suffix=='.npz' and m.size<30000000:
                    try:
                        h=headers(tar.extractfile(m).read())
                        for k,x in h.items():
                            d=x['shape']
                            if d and (d[-1] in (416,527) or
                                      (len(d)>1 and d[-2:]==[255,9])):
                                arrays.append({'archive_member':name,'array':k,'shape':d})
                    except (OSError,ValueError,zipfile.BadZipFile) as e:
                        arrays.append({'member':name,'skipped':str(e)[:150]})
        results.append({'tag_bundle':file.name,'sha256':sha(file),'bytes':file.stat().st_size,
          'members':member_count,'Python_or_notebook_files_scanned':codefiles,
          'interesting_members':noteworthy[:150],'relevant_source_code_matches':matches[:100],
          'authentic_416_527_data_shape_matches':arrays})
    report={'status':'S203_GITHUB_RELEASES_SOURCE_AND_ARRAY_HEADER_FORENSIC',
       'release_tarballs_scanned':len(files),
       'archives':results,
       'source_signatures_found':sum(len(x['relevant_source_code_matches']) for x in results),
       '527D_or_416D_original_matrices_found':sum(len(x['authentic_416_527_data_shape_matches']) for x in results),
       'source_file_recoverable_original_S5_owned_builder_proven':False,
       'archive_data_values_or_ground_truth_accessed':False,
       'no_new_audio_processed':True,'S80_promoted':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in (
       'status','release_tarballs_scanned','source_signatures_found',
       '527D_or_416D_original_matrices_found','S80_promoted')},indent=2),flush=True)
    for x in results:
        print(json.dumps({k:x[k] for k in (
          'tag_bundle','members','Python_or_notebook_files_scanned',
          'relevant_source_code_matches','authentic_416_527_data_shape_matches')},ensure_ascii=False),flush=True)
def selftest():
    check(PATTERNS['owned527_builder'].search('def owned527(x):') is not None,
          'original source syntax literal')
    print('S203 PASS exact source signature search and no annotation value access')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        check(a.input and a.out,'downloaded tagged release tarballs required')
        examine(a.input,a.out)
