"""S110 recover S5 archived feature/design schema without reading labels.

Metadata only. All files are SHA256-verified against immutable Series5 storage
manifest. No inference or promotion.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from zipfile import ZipFile
from numpy.lib import format as fmt

def header(fp):
    ver=fmt.read_magic(fp)
    return (fmt.read_array_header_1_0(fp) if ver==(1,0) else fmt.read_array_header_2_0(fp))
def run(root,out):
    manifest=json.loads((root/'storage-manifest.json').read_text())
    rows=[]
    for record in manifest['files']:
        f=root/record['path']
        if not f.exists(): raise FileNotFoundError(str(f))
        digest=hashlib.sha256(f.read_bytes()).hexdigest()
        if digest!=record['sha256']:raise ValueError(f'SHA drift: {f}')
        meta={'path':record['path'],'bytes':record['bytes'],'sha256_verified':True}
        if f.suffix=='.npz':
            with ZipFile(f) as z:
                arrays=[]
                for k in z.namelist():
                    if k.endswith('.npy'):
                        with z.open(k) as p:
                            shape,fortran,dtype=header(p)
                        arrays.append({'key':k[:-4],'shape':list(shape),'dtype':str(dtype)})
                meta['arrays']=arrays
        rows.append(meta)
    active=[{'path':x['path'],**arr} for x in rows for arr in x.get('arrays',[]) if 527 in arr['shape']]
    result={'status':'S110_SERIES5_ORIGINAL_ARCHIVE_SCHEMA_VERIFIED',
       'verified_files':len(rows),'archive_inventory':rows,
       'saved_527D_arrays':active,
       'authentic_audio_to_527D_design_builder_verified':False,
       'independent_S80_accuracy':None,'promotion':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'verified_files':len(rows),
       'schemas':[{'path':x['path'],'arrays':x.get('arrays',[])} for x in rows if 'arrays' in x],
       'saved_527D_arrays':active},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)
