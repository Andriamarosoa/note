"""S377 audited original 20-stage S80 source-only rebase to 8 untouched WAVs.

Reuses SAME original S359 exact deterministic text-only code transform,
replaces external 24-WAV S358 SHA reference with NEW 8-WAV S374,
changes only explicit cohort count and WAV selection to original
S372 SHA-frozen eight names. Never touches any XML or true-K.
One original 133-model/family weight is NOT refitted or replaced.
This emits ORIGINAL named 477 source modules; Actions copies them
under derived _<newN> file stems. Every compiled source has SHA proof.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s359_deterministic_rebase_original_S80_stages_24_IDMT import NAMES,rewrite
def need(x,msg):
    if not x:raise ValueError(msg)
def sha(data):return hashlib.sha256(data).hexdigest()
def adapt(original,n):
    content=rewrite(original,n)
    content=content.replace('scripts.s358_original_freeze_V88_24_IDMT_chords_PREXML',
                             'scripts.s374_original_freeze_V88_last8_unseen_chords_PREXML')
    content=content.replace('S358','S374').replace('s358','s374')
    content=content.replace('S357','S372').replace('s357','s372')
    content=content.replace('next_twentfour','final_eight')
    content=content.replace('S374_TRUE_OFFICIAL_FREEZE_ON_UNEXPOSED_DATASET1_24_WAV_UNLABELLED',
                            'S374_TRUE_OFFICIAL_FREEZE_ON_UNEXPOSED_DATASET1_8_WAV_UNLABELLED')
    content=content.replace('len(selected)==24','len(selected)==8')
    content=content.replace("evidence['count_WAV']==24","evidence['count_WAV']==8")
    content=content.replace('ref_report["count_WAV"]==24','ref_report["count_WAV"]==8')
    content=content.replace('WAV_count=24','WAV_count=8')
    content=content.replace('["WAV_count"]==24','["WAV_count"]==8')
    need('next_twentfour' not in content and
         's358_original_freeze_V88_24_IDMT' not in content and
         'S358_TRUE_OFFICIAL_FREEZE' not in content,
         'S377 generated source still depends on already exposed 24 WAVs')
    return content
def run(s374,report,source,out):
    need(not out.exists(),'immutable compiled original source provenance')
    evidence=json.loads(report.read_text())
    need(evidence['status']=='S374_TRUE_OFFICIAL_FREEZE_ON_UNEXPOSED_DATASET1_8_WAV_UNLABELLED'
         and evidence['count_WAV']==8
         and evidence['sealed_original_predictions_SHA256']==sha(s374.read_bytes())
         and not evidence['XML_MIDI_annotated_contents_read'],
         'authentic sealed source-only S374 eight new native events not verified')
    with np.load(s374,allow_pickle=False) as z:
        n=len(z['row_id'])
        need(n==evidence['original_native_event_count'] and n>=8 and
             len(set(map(str,z['member'])))==8,
             'S377 original eight recording native source IDs drifted')
    out.mkdir(parents=True)
    ledger=[]
    for fn in NAMES:
        original=(source/fn).read_bytes()
        rendered=adapt(original.decode(),n)
        compile(rendered,str(out/fn),'exec')
        (out/fn).write_text(rendered)
        ledger.append(dict(original_module=fn,original_SHA256=sha(original),
                           source_rebased_SHA256=sha(rendered.encode())))
    summary=dict(status='S377_20_ORIGINAL_TRAINED_S80_MODULES_REBASED_8_UNTOUCHED_WAVS',
       original_native_events=n,original_source_WAV_count=8,
       pinned_official_S374_SHA256=sha(s374.read_bytes()),
       unchanged_original_s359_generator_required=True,
       original_trained_models_refit=False,
       real_external_XML_read=False,
       all_20_original_modules_compile=True,
       archive_members=ledger,official_promotion=False)
    (out/'generation_report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='archive_members'},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('s374','report','source','out'):
        p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args();run(args.s374,args.report,args.source,args.out)
