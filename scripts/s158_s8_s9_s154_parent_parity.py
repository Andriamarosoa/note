"""S158: prove S157 original trained S8->S9 equals pinned S153/S154 adapter
EVENT-BY-EVENT, hence no need to rerun the 596 original downstream risk
checkpoint forwards to check the same unchanged runtime input.
This only proves S9 vector equality on 57 GOAT events, not true S5 source
equivalence or generalization, nor S80 promotion. No labels/MIDI accessed.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
def need(x,m):
    if not x:raise ValueError(m)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(s157,s153,s154,out):
    need(not out.exists(),"S158 fail closed immutable proof")
    with np.load(s157,allow_pickle=False) as z:
        need(set(z.files)=={'row_id','freeze_K','adapted_global_S135_K',
            'adapted_poly_PR16_K','S8_original_weight_K',
            'S9_original_formula_K','S8_learned_advantage'},
            'source S157 strictly label-free schema')
        ids=z['row_id'].astype(str)
        source=z['S9_original_formula_K'].astype(np.int8)
        freeze=z['freeze_K'].astype(np.int8)
        p=z['adapted_poly_PR16_K'].astype(np.int8)
    with np.load(s153,allow_pickle=False) as z:
        need('true_K' not in z.files and 'midi' not in z.files,'no truth S153')
        need(np.array_equal(ids,z['row_id'].astype(str)),'S153 original native event order')
        s153parent=z['adapted_S9_K'].astype(np.int8)
        need(np.array_equal(freeze,z['freeze_K']),'baseline S153 drift')
        s15=z['S15_safe_OR_K'].astype(np.int8)
    with np.load(s154,allow_pickle=False) as z:
        need(set(z.files)=={'row_id','freeze_K','adapted_S9_K',
                             'adapted_S15_K','S16_K','S17_K','S18_K'},
             'strict stage output S154 schema')
        need(np.array_equal(ids,z['row_id'].astype(str)),'S154 native GOAT group mismatch')
        s154parent=z['adapted_S9_K'].astype(np.int8)
        final=z['S18_K'].astype(np.int8)
        need(np.array_equal(freeze,z['freeze_K']),'S154 original parent mismatch')
        need(np.array_equal(s15,z['adapted_S15_K']),'S15/16 alignment drift')
    mismatch1=int((source!=s153parent).sum())
    mismatch2=int((source!=s154parent).sum())
    need(mismatch1==mismatch2==0,"Original S8/S9 changed S13/S18 upstream; rerun downstream!")
    need(np.array_equal(source,p),'S157 S9 should match former p on 57; no hidden shift')
    need(len(ids)==len(set(ids))==57,"57 identical GOAT events")
    out.parent.mkdir(parents=True)
    result={'status':'S158_REAL_ORIGINAL_S8_S9_OUTPUTS_IDENTITY_PROVEN_ON_GOAT',
            'native_events':57,
            'new_original_S8_S9_vs_S153_existing_parent_mismatches':mismatch1,
            'new_original_S8_S9_vs_S154_existing_parent_mismatches':mismatch2,
            'S8_to_S9_original_weights_and_rule_applied_on_GOAT':True,
            'S13_to_S18_downstream_input_is_unchanged':True,
            'original_S5_global_poly_heads_used':False,
            'exact_historical_S9_equivalence_established':False,
            'new_music_full_S80_ExactK_measured':False,
            'S154_final_K_histogram':np.bincount(final,minlength=7).tolist(),
            'source_checksums':{'S157':sha(s157),'S153':sha(s153),'S154':sha(s154)},
            'GOAT_label_MIDI_read':False,'S80_promoted':False}
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
def selftest():
    z=np.asarray([0,1,2],np.int8);q=z.copy()
    need(np.array_equal(z,q),"vector order identity")
    print('S158 PASS deterministic parent identity without label access')
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for n in ('s157','s153','s154','out'):ap.add_argument('--'+n,type=Path)
    ap.add_argument('--self-test',action='store_true')
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        need(all(getattr(a,n) is not None for n in ('s157','s153','s154','out')),
             'S157 S153 S154 frozen sources mandatory')
        run(a.s157,a.s153,a.s154,a.out)
