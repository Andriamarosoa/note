"""S188 inspect dataset2 note-level XML/WAV pairing by ZIP names only.

Proposes a NEW clean polyphonic Exact-K benchmark; no label content opened.
The existing dataset4 chord labels are not presumed to encode K cardinality.
"""
from __future__ import annotations
import argparse,json,urllib.request,zipfile,hashlib,collections
from pathlib import PurePosixPath,Path
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP

def need(q,m):
    if not q:raise ValueError(m)
def run(out):
    need(not out.exists(),'immutable index')
    with urllib.request.urlopen(urllib.request.Request(API,headers={
      'Accept':'application/json','User-Agent':'S80-S188-annotations-unopened'}),timeout=45) as r:
        source=audit(json.load(r))
    reader=HttpRangeZIP(source['source_url'],source['size_bytes'])
    with zipfile.ZipFile(reader) as z:
        infos=[i for i in z.infolist() if '/dataset2/' in ('/'+i.filename.lower()) and not i.is_dir()]
        counts=dict(collections.Counter(PurePosixPath(i.filename).suffix.lower() for i in infos))
        wav=[i.filename for i in infos if i.filename.lower().endswith('.wav')]
        xml=[i.filename for i in infos if i.filename.lower().endswith('.xml')]
        txt=[i.filename for i in infos if i.filename.lower().endswith('.txt')]
        need(len(wav)>0 and len(xml)>0,'note event benchmark lacks recordings or XML')
        allxml=set(xml)
        matches=[]
        for x in wav:
            p=PurePosixPath(x)
            direct=str(p.with_suffix('.xml'))
            pref=[y for y in xml if PurePosixPath(y).stem==p.stem]
            matches.append({'wav':x,'same_directory_XML':direct if direct in allxml else None,
                            'same_stem_XML_across_folders':pref[:3]})
        sample=sorted(wav,key=lambda w:hashlib.sha256(w.encode()).hexdigest())[:16]
        report={'status':'S188_DATASET2_AUDIO_XML_FILENAME_INDEX_UNOPENED',
          'dataset2_file_count':len(infos),'dataset2_extensions':counts,
          'dataset2_wav_count':len(wav),'dataset2_XML_count':len(xml),
          'dataset2_txt_count':len(txt),
          'dataset2_wav_filename_samples':wav[:18],
          'dataset2_xml_filename_samples':xml[:18],
          'hash_predeclared_new_AUDIO_only_selection':sample,
          'sample_crosswalk':[m for m in matches if m['wav'] in sample],
          'all_WAV_exact_same_dir_XML_pairs':sum(x['same_directory_XML'] is not None for x in matches),
          'all_WAV_matching_any_XML_stem':sum(bool(x['same_stem_XML_across_folders']) for x in matches),
          'zip_index_range_bytes':reader.network_bytes,
          'XML_content_read':False,'audio_content_read':False,
          'labels_or_correctness_used_to_select_files':False,
          'exact_K_accuracy_measured':False,'S80_promoted':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
def selftest():
    samples=['a.wav','b.wav']; xml=['a.xml','b.xml']
    need(all(a.rsplit('.',1)[0]+'.xml' in xml for a in samples),'paired names')
    print('S188 PASS no labels, deterministic candidate WAV selection')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path);p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(a.out is not None,'output needed');run(a.out)
