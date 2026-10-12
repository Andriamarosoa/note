"""S182 probe official Zenodo large-archive HTTP Range without downloading dataset.

Only small bytes at archive tail are transferred. No XML/MIDI, held-out WAV,
or model data opened. No archive validation/accuracy claim.
"""
from __future__ import annotations
import argparse,json,urllib.request
from pathlib import Path
from scripts.s177_preregister_external_polyphony_source import API,audit
def main(out):
    if out.exists():raise ValueError('immutable transfer probe')
    with urllib.request.urlopen(
      urllib.request.Request(API,headers={'Accept':'application/json','User-Agent':'S80-range-audit/1.0'}),
      timeout=45) as response:
        data=audit(json.load(response))
    url=data['source_url'];total=int(data['size_bytes']);take=65536
    req=urllib.request.Request(url,headers={'Range':f'bytes={total-take}-{total-1}',
                                            'Accept-Encoding':'identity',
                                            'User-Agent':'S80-label-blind-range-audit/1.0'})
    with urllib.request.urlopen(req,timeout=90) as response:
        status=response.status
        cr=response.headers.get('Content-Range')
        accept=response.headers.get('Accept-Ranges')
        if status==206:
            b=response.read(take+1)
            ok=len(b)==take and cr==f'bytes {total-take}-{total-1}/{total}'
        else:
            # Do not read an unbounded full archive on a server that ignores Range.
            b=b'';ok=False
    report={'status':'S182_OFFICIAL_EXTERNAL_ZIP_RANGE_PROBE',
       'record_id':data['record_id'],'zip_expected_size':total,
       'source_manifest_MD5_published':data['expected_md5'],
       'server_http_status':status,
       'response_content_range':cr,
       'accept_ranges_header':accept,
       'exact_64KiB_tail_delivered':ok,
       'server_http_range_supported':ok,
       'bytes_read':len(b),
       'raw_WAV_read':False,'annotations_read':False,
       'S80_independent_ExactK_scored':False,'S80_promoted':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();main(args.out)
