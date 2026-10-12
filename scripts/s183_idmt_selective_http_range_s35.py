"""S183 fast HTTP-RANGE ZIP reader for true independent IDMT guitar WAV inference.

Use *official* HTTPS Zenodo API, pinned archive identity/length and partial
byte-range reads; ZipFile verifies CRC32 of every extracted WAV. The published
1.33GB MD5 is NOT recalculated because entire archive is not downloaded.
No audio files or annotated XML/MIDI are released or opened.
"""
from __future__ import annotations
import argparse,hashlib,io,json,urllib.request,wave,zipfile
from pathlib import Path
import numpy as np
from scripts.s177_preregister_external_polyphony_source import API,ORIGINAL_MD5,audit
from scripts.s178_idmt_audio_only_native_s35_heads import (
    decode_wav,acoustic_onsets,group_wav_names,views,predict_original_s35,sha_bytes)
from scripts.extract_v273_harmonic_trajectory import _frames,features_from_spectrogram
from scripts.loop_v273_birth_flow import flow_profile
from scripts.v273_extended_head_bank import audio_temporal_views

class HttpRangeZIP:
    """Random access validated HTTP 206 windows. Never issue full archive GET."""
    def __init__(self,url,size,fetch=None,window=1024*1024):
        if not url.startswith('https://zenodo.org/') or not (1_000_000_000<size<2_000_000_000):
            if fetch is None:raise ValueError('untrusted external ZIP source')
        self.url=url;self.size=size;self.fetch=fetch;self.window=window
        self.pos=0;self.start=-1;self.cache=b''
        self.network_bytes=0;self.range_requests=0
    def seek(self,offset,whence=0):
        new=int(offset if whence==0 else self.pos+offset if whence==1 else self.size+offset)
        if new<0 or new>self.size:raise ValueError('invalid bounded ZIP seek')
        self.pos=new;return new
    def tell(self):
        return self.pos
    def seekable(self):
        return True
    def readable(self):
        return True
    def _receive(self,start,end):
        if self.fetch is not None:
            data=self.fetch(start,end)
        else:
            req=urllib.request.Request(self.url,
              headers={'Range':f'bytes={start}-{end}',
                       'Accept-Encoding':'identity','User-Agent':'S80-range-zip-audio-only/1.0'})
            with urllib.request.urlopen(req,timeout=120) as response:
                if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':
                    raise ValueError('Zenodo HTTP Range changed during ZIP access')
                data=response.read(end-start+2)
        if len(data)!=end-start+1:
            raise ValueError('partial/truncated/unbounded ZIP Range response')
        self.network_bytes+=len(data);self.range_requests+=1
        return data
    def read(self,n=-1):
        if n<0:raise ValueError('unbounded ZIP read forbidden')
        remaining=min(n,self.size-self.pos)
        chunks=[]
        while remaining:
            if not (self.start<=self.pos<self.start+len(self.cache)):
                start=(self.pos//self.window)*self.window
                end=min(start+self.window,self.size)-1
                self.cache=self._receive(start,end)
                self.start=start
            offset=self.pos-self.start
            step=min(remaining,len(self.cache)-offset)
            chunks.append(self.cache[offset:offset+step])
            self.pos+=step;remaining-=step
        return b''.join(chunks)
def need(ok,m):
    if not ok:raise ValueError(m)
def run(models,out,max_tracks=10,max_events=10):
    need(not out.exists(),'immutable result')
    with urllib.request.urlopen(urllib.request.Request(
       API,headers={'Accept':'application/json','User-Agent':'S80-label-blind-range-research/1.0'}),
       timeout=60) as response:
        src=audit(json.load(response))
    reader=HttpRangeZIP(src['source_url'],int(src['size_bytes']))
    summaries=[];seqs=[];filenames=[];starts=[];selected=[]
    with zipfile.ZipFile(reader,'r') as z:
        chosen,candidates=group_wav_names(z,max_tracks)
        for name in chosen:
            info=z.getinfo(name)
            need(44100*2<=info.file_size<=44100*120*3,
                 'unexpected IDMT WAV size')
            payload=z.read(name)  # ZipFile enforces CRC32 for this WAV; no other member read
            signal=decode_wav(payload)
            positions=acoustic_onsets(signal,max_events)
            for sample in positions:
                spec,times=_frames(signal,int(sample))
                d,activ=features_from_spectrogram(spec,times,return_states=True)
                need(len(d)==58 and activ.shape==(42,49),'original S97 acoustic contract')
                summaries.append([d[k] for k in sorted(d)])
                seqs.append(np.log1p(activ))
                filenames.append(name);starts.append(int(sample))
            selected.append({'member_name_sha256':sha_bytes(name.encode()),
                 'wav_sha256':sha_bytes(payload),'ZIP_member_crc32':str(hex(info.CRC)),
                 'events':len(positions),'wav_seconds':round(len(signal)/44100,3),
                 'uncompressed_audio_bytes':len(payload)})
            print(json.dumps({'wav_name_sha256':sha_bytes(name.encode()),
                              'label_free_events':len(positions)}),flush=True)
    need(len(summaries)>=max_tracks,'too few usable audio-only events in deterministic IDMT selection')
    summary=np.asarray(summaries,np.float32)
    seq=np.asarray(seqs,np.float32)
    flow=flow_profile(seq)
    morph,phys=audio_temporal_views(seq)
    probabilities,proofs=predict_original_s35(models,views(summary,flow,morph,phys))
    names=tuple(probabilities)
    ids=np.asarray([sha_bytes((name+':'+str(sample)).encode()) for name,sample in zip(filenames,starts)])
    need(len(set(ids.tolist()))==len(ids),'duplicate IDMT event IDs')
    out.mkdir(parents=True)
    np.savez_compressed(out/'S178_IDMT_audio_only_S35_source_heads.npz',
      row_id=ids,summary58=summary,flow245=flow,morphology490=morph,
      physics294=phys,sequence42x49=seq,source_head_ids=np.asarray(names),
      source_probabilities=np.stack([probabilities[h] for h in names],axis=1),
      source_predicted_K=np.column_stack([probabilities[h].argmax(1) for h in names]).astype(np.int8))
    report={'status':'S183_IDMT_RANGE_ONLY_REAL_AUDIO_28_ORIGINAL_S35_EXPERTS',
      'original_corpus_MD5_PUBLISHED_NOT_RECALCULATED':src['expected_md5'],
      'original_zip_total_size':src['size_bytes'],
      'range_requests':reader.range_requests,
      'range_bytes_read':reader.network_bytes,
      'full_zip_downloaded':False,
      'wav_files_chosen':len(selected),
      'dataset4_total_wav_candidates':candidates,
      'native_unseen_audio_events':len(ids),
      'wav_CRC32_verified_by_Python_ZipFile':True,
      'wav_SHA256_provenance':selected,
      'source_pretrained_genuine_S35_checkpoint_count':len(proofs),
      'source_pretrained_S35_checkpoint_hashes':proofs,
      'original_pretrained_head_class_histogram_K0to6':
         {h:np.bincount(probabilities[h].argmax(1),minlength=7).tolist() for h in names},
      'source_S35_historical_training_reused_unchanged':True,
      'original_vote32_inputs_available':False,
      'missing_original_S1_S4_255_model_sources':True,
      'external_WAV_bytes_released':False,
      'XML_MIDI_or_annotations_accessed':False,
      'ExactK_ground_truth_accuracy_measured':False,
      'original_S80_promoted':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in
      ('status','range_requests','range_bytes_read','wav_files_chosen',
       'native_unseen_audio_events','source_pretrained_genuine_S35_checkpoint_count',
       'original_pretrained_head_class_histogram_K0to6','original_S80_promoted')},indent=2),flush=True)
def selftest():
    rng=np.random.default_rng(3183)
    x=(rng.normal(0,50,88200).astype('<i2')).tobytes()
    s=io.BytesIO()
    with wave.open(s,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(44100);w.writeframes(x)
    raw=s.getvalue();buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('dataset4/test.wav',raw)
    source=buffer.getvalue()
    r=HttpRangeZIP('https://example.invalid',len(source),
        fetch=lambda a,b:source[a:b+1],window=1024)
    with zipfile.ZipFile(r,'r') as z:
        sample=z.read('dataset4/test.wav')
    need(sample==raw and r.network_bytes<len(source)+3*1024,
         'bounded random ZIP range or exact CRC mismatch')
    need(len(decode_wav(sample))==88200,'sample decoder failed')
    print('S183 PASS simulated ZipFile central directory and compressed WAV CRC with bounded range reads')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--models',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--max-tracks',type=int,default=10)
    p.add_argument('--max-events',type=int,default=10)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(a.models and a.out,'checkpoint source/output required')
        run(a.models,a.out,a.max_tracks,a.max_events)
