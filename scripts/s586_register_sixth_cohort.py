"""S586: choose five new composers from CSV only, excluding 33 prior sources."""
import argparse,csv,hashlib,io,json,urllib.request
from pathlib import Path
from scripts.s469_preregister_MAESTRO_v3_third_source_filename_only import URL,AUDIO_MIRROR
from scripts.s577_preregister_next_composer_cohort_after28_audio_unopened import components
from scripts.s585_attack_sustain_restrike import need,sha,save

def choose(rows,names,artists):
    excluded=set().union(*(components(x) for x in artists));eligible=[]
    for r in rows:
        w=r.get('audio_filename','');m=r.get('midi_filename','');c=r.get('canonical_composer','')
        try:d=float(r['duration'])
        except (KeyError,ValueError):continue
        if (r.get('split')=='train' and 28<=d<=1000 and c and w.endswith('.wav')
            and m.endswith('.midi') and w[:-4]==m[:-5] and '..' not in w+m
            and not w.startswith('/') and not m.startswith('/')
            and w not in names and not (components(c)&excluded)):
            eligible.append(r)
    eligible.sort(key=lambda r:hashlib.sha256(('S586-four-state-after33-v1/'+r['audio_filename']).encode()).hexdigest())
    selected=[];seen=set()
    for r in eligible:
        c=r['canonical_composer']
        if components(c)&seen:continue
        seen.update(components(c));w=r['audio_filename']
        selected.append(dict(composer=c,composition=r['canonical_title'],official_MAESTRO_split='train',
            audio_filename=w,MIDI_filename_path_ONLY_not_opened=r['midi_filename'],
            filename_SHA256=hashlib.sha256(w.encode()).hexdigest(),source_duration_seconds=float(r['duration']),
            audio_excerpt_start_seconds=15.,audio_excerpt_duration_seconds=12.))
        if len(selected)==5:break
    need(len(selected)==5,'S586 fewer than 5 eligible unused composer components; do not relax exclusions')
    return selected,len(eligible)

def run(root,out):
    need(not out.exists(),'S586 metadata cannot overwrite a registered cohort')
    previous=[];proof={}
    for step in (469,537,545,554,561,570,577):
        path=root/f'S{step}/report.json';m=json.loads(path.read_text())
        rows=m.get('selected_sources',m.get('chosen_metadata_only'))
        need(len(rows)==(3 if step==469 else 5),'S586 previous cohort incomplete')
        previous.extend(rows);proof[f'S{step}']=sha(path)
    names={r['audio_filename'] for r in previous};artists={r['composer'] for r in previous}
    need(len(names)==33,'S586 must exclude all 33 exposed source filenames')
    with urllib.request.urlopen(URL,timeout=75) as response: data=response.read(1000000)
    digest=hashlib.sha256(data).hexdigest()
    need(digest=='cb88a64a12971a995dd41852aa3a33879501cf70d01b9a6feb637f4ad8d8ddff','S586 official metadata changed')
    chosen,eligible=choose(list(csv.DictReader(io.StringIO(data.decode('utf-8-sig')))),names,artists)
    out.mkdir(parents=True)
    save(out/'report.json',dict(status='S586_PREREGISTERED_FIVE_MORE_TRAIN_SPLIT_COMPOSERS_AFTER_S544_NO_WAV_MIDI',
        chosen_metadata_only=chosen,official_metadata_URL=URL,official_metadata_sha256=digest,
        source_audio_mirror=AUDIO_MIRROR,source_WAV_files_not_read=True,source_MIDI_files_not_read=True,
        five_next_distinct_composers=5,MAESTRO_original_split_used='train',NOT_official_MAESTRO_test_split=True,
        previous_source_count=33,previous_names=sorted(names),previous_composers=sorted(artists),previous_manifests_SHA=proof,
        original_previous_S469_3_S537_5_S545_5_S554_5_S561_5_S570_5_S577_5_components_excluded=True,
        co_composer_COMPONENTS_fully_excluded_and_pairwise_disjoint=True,
        filenames_frozen_before_audio_model_inference=True,eligible_tracks=eligible,
        NOT_guaranteed_never_used_elsewhere_in_repo=True,no_promotion=True))
    print('S586 METADATA ONLY',json.dumps(chosen),flush=True)

def selftest():
    rows=[dict(split='train',duration=100,audio_filename=f'2000/{c}.wav',
        midi_filename=f'2000/{c}.midi',canonical_composer=c,canonical_title='x') for c in 'ABCDEFG']
    one,_=choose(rows,{'2000/A.wav'},{'A','B'})
    two,_=choose(rows[::-1],{'2000/A.wav'},{'A','B'})
    need(one==two and {x['composer'] for x in one}==set('CDEFG'),'Metadata selection/exclusion determinism')
    print('S586 SELFTEST PASS metadata-only exclusions')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:selftest()
    else:run(a.root,a.out)
