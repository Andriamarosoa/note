"""S109: inventory original Series3 source archive for any S5 527D matrix.

Read only file/array headers and provenance hashes, never annotation values.
"""
from __future__ import annotations
import argparse,hashlib,io,json,tarfile,zipfile
from pathlib import Path
from numpy.lib import format as npformat

def header(f):
    version=npformat.read_magic(f)
    shape,fortran,dtype=(npformat.read_array_header_1_0(f) if version==(1,0)
        else npformat.read_array_header_2_0(f))
    return {'shape':list(shape),'dtype':str(dtype),'fortran':fortran}

def inventory(root,out):
    m=json.loads((root/'storage-manifest.json').read_text())
    chunks=[]
    for p in m['parts']:
        path=root/Path(p['path']).name
        data=path.read_bytes()
        if len(data)!=p['bytes'] or hashlib.sha256(data).hexdigest()!=p['sha256']:
            raise ValueError('S3 shard integrity mismatch: '+path.name)
        chunks.append(data)
    complete=b''.join(chunks)
    if len(complete)!=m['archive_bytes'] or hashlib.sha256(complete).hexdigest()!=m['archive_sha256']:
        raise ValueError('S3 assembled tar checksum failure')
    entries=[]
    with tarfile.open(fileobj=io.BytesIO(complete),mode='r:') as tar:
        for item in tar:
            desc={'path':item.name,'bytes':item.size}
            if item.isfile() and item.name.endswith('.npz'):
                with zipfile.ZipFile(io.BytesIO(tar.extractfile(item).read())) as z:
                    desc['arrays']={}
                    for field in z.namelist():
                        if field.endswith('.npy'):
                            with z.open(field) as f:desc['arrays'][field[:-4]]=header(f)
            if item.isfile() and item.name.endswith('.json') and item.size<50000:
                val=json.loads(tar.extractfile(item).read())
                desc['json_keys']=list(val) if isinstance(val,dict) else []
            entries.append(desc)
    candidates=[{'path':e['path'],'name':key,'shape':a['shape']} for e in entries
        for key,a in e.get('arrays',{}).items() if 527 in a['shape']]
    result={'status':'S109_S3_ARCHIVE_SCHEMA_INVENTORY',
        'verified_shards':len(chunks),'total_entries':len(entries),
        'entries':entries,'527D_candidates':candidates,
        'authentic_S5_design527_rebuilt':False,'S80_promoted':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],
        'verified_shards':len(chunks),'total_entries':len(entries),
        'paths':[e['path'] for e in entries],'527D_candidates':candidates},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    inventory(a.root,a.out)
