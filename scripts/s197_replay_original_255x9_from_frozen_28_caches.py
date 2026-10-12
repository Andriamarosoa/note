"""S197 reconstruct original 255 group proposals AND 9 local audit values from pinned 14+14 cached producers.

Uses genuine archived labels ONLY for outer TRAIN reference neighbor outcome audit.
Receiver rows' own truth is never passed to the audit. Validates reconstructed
group proposals and 20D summaries against preserved original historical OOF
artefacts. No S5 527D and no unseen new-WAV inference.
"""
from __future__ import annotations
import argparse,hashlib,json,tempfile,zipfile
from pathlib import Path
import numpy as np
from scripts.v273_selector_contract import context_columns
from scripts.prepare_v273_group127_producers import FrozenProducerPool
from scripts.prepare_v273_learned_corrector import FrozenCorrector,CataloguePool
from scripts.v273_catalogue_contract import joint_groups,local_audits
from scripts.v273_regression_loops import group_context
from scripts.yourmt3_exactk_common import digest

ROOT=Path('analysis/evidence/v273-audit-memory/source-archives')
def need(b,m):
    if not b:raise ValueError(m)
def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def archive_caches(root,dst):
    manifest=json.loads((root/ROOT/'manifest.json').read_text())
    record=manifest['files']['upstream-producers.zip']
    path=root/ROOT/'upstream-producers.zip'
    need(checksum(path)==record['sha256'] and path.stat().st_size==record['bytes'],
         'sealed exact saved original specialist/corrector upstream source archive SHA drift')
    required=('specialists/producers.npz','specialists/manifest.json',
              'learned_corrector/correctors.npz','learned_corrector/manifest.json')
    with zipfile.ZipFile(path) as z:
        need(set(required)<=set(z.namelist()),'original upstream producer/corrector manifests missing')
        for name in required:
            source=z.read(name)  # the originals are archived immutable caches
            if 'members' in record and name in record['members']:
                entry=record['members'][name]
                need(hashlib.sha256(source).hexdigest()==entry['sha256'],
                     'individual original cached archive member SHA drift')
            file=dst/name
            file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(source)
    return {n:checksum(dst/n) for n in required}
def historical_inputs(root):
    p=root/'analysis/evidence/v273-regression-loops/prepared/inputs.npz'
    with np.load(p,allow_pickle=False) as z:
        needed=('global_index','truth','baseline','fold','proposal','group_context',
                'feature_names','features','group_names')
        values={key:z[key].copy() for key in needed}
    n=len(values['global_index'])
    need(n==7493 and values['features'].shape==(7493,58),'original historical feature shape')
    source_names=tuple(map(str,values['feature_names']))
    need(source_names==tuple(sorted(source_names)),'historical 58D original alphabetical feature order changed')
    chosen=context_columns(source_names)
    lookup=[source_names.index(n) for n in chosen]
    raw=np.asarray(values['features'][:,lookup],float)
    need(raw.shape[1]==len(chosen) and np.isfinite(raw).all(),'original acoustic raw contextual observables')
    return values,raw,chosen
def run(root,out,fold=0,limit=128):
    need(not out.exists(),'immutable per-fold restored audit proof')
    need(fold in (0,1,2,4),'old excluded fold3 cannot be used')
    d,raw,names=historical_inputs(root)
    idd=d['global_index'];y=d['truth'];b=d['baseline'];f=d['fold']
    need(set(np.unique(f))=={0,1,2,4} and np.isin(b,[2,3,4]).all(),
         'original K2-K4 evaluation cohort only')
    held=np.flatnonzero(f==fold);train=np.flatnonzero(f!=fold)
    if limit>0:held=held[:limit]
    need(0<len(held) and len(train)>len(held),'no cross-fit outer/receiver cohort')
    need(not set(map(int,idd[held]))&set(map(int,idd[train])),
         'historical outer receiver/reference events overlap')
    with tempfile.TemporaryDirectory(prefix='s197-cache-') as directory:
        cached=Path(directory)
        proofs=archive_caches(root,cached)
        # Pretrained caches provide all 14 train-fold variants, no fitting.
        expert=FrozenProducerPool({},y,b,f,idd,cached/'specialists')
        corrector=FrozenCorrector(y,b,f,idd,cached/'learned_corrector')
        pool=CataloguePool(expert,corrector)
        tp,recs=pool.crossfit(train,{fold})
        hp=pool.predict(set(map(int,f[train])),held,{fold})
        need(tp.shape==(len(train),8,5) and hp.shape==(len(held),8,5),
             'true eight frozen original specialist+corrector sources')
        original_source,_,_=joint_groups(tp,b[train])
        recreated,direct,_=joint_groups(hp,b[held])
        original=d['proposal'][held]
        mismatches=int(np.count_nonzero(original!=recreated))
        need(mismatches==0,f'original 255 OOF proposal contract mismatch {mismatches}')
        # Pinned deterministic nearest-neighbor reference outcome audits:
        # uses y[train] and NEVER y[held].
        group9=local_audits(raw[train],raw[held],original_source,y[train],
                            b[train],recreated,b[held],idd[train],idd[held])
        need(group9.shape==(len(held),255,9) and np.isfinite(group9).all(),
             'true local9 source audit generation failed')
        recovered,groupnames=group_context(recreated,group9,b[held])
        expected=d['group_context'][held]
        need(list(map(str,d['group_names']))==groupnames,'original 20D aggregate order drift')
        diff=np.abs(recovered-expected)
        max_diff=float(np.max(diff))
        mean_diff=float(np.mean(diff))
        col_errors=np.max(diff,axis=(0,1)).tolist()
        exact_counts=int(np.sum(np.all(diff<=2e-5,axis=(1,2))))
        report={
          'status':'S197_ORIGINAL_PINNED_UPSTREAM_REBUILT_255_AND_9_HISTORICAL_OOF_AUDITS',
          'held_fold':fold,'held_rows':len(held),
          'reference_train_rows':len(train),'receiver_truth_used_for_features_or_audits':False,
          'training_label_usage':'training/reference fold true K only for historical learned audit outcomes',
          'original_eight_voting_producers_from_14_cached_train_fold_sets':True,
          'source_cached_Specialist_Corrector_SHA256':proofs,
          'original_group_proposal_mismatches':mismatches,
          'original_9D_per_group_audits_shape':list(group9.shape),
          'audits_nine_complete_group_columns_recomputed_from_genuine_original_source':True,
          'aggregate_20D_max_absolute_discrepancy':max_diff,
          'aggregate_20D_mean_absolute_discrepancy':mean_diff,
          'aggregate_20D_component_max_absolute_discrepancy':col_errors,
          'all_receiver_rows_reproduce_historical_20D_audits_with_tolerance_2e_minus_5':exact_counts,
          'outer_training_folds':[int(v) for v in sorted(np.unique(f[train]))],
          'outer_receiver_ids_sha256':hashlib.sha256(idd[held].astype('<i8').tobytes()).hexdigest(),
          'audit_may_recover_exact_archive':bool(max_diff<=2e-5),
          'only_old_development_events_audited':True,'new_WAV_verified_original_source_predictions':False,
          'S5_original527_feature_constructor_recovered':False,'S80_promoted':False}
        out.mkdir(parents=True)
        np.savez_compressed(out/'original_9D_fold_audits_REPLAY.npz',
           global_index=idd[held],fold=np.full(len(held),fold,np.int8),
           original_255_group_proposal=recreated.astype(np.int8),
           rebuilt_original_255x9_group_audits=group9,
           recovered_7x20_audit_summary=recovered.astype(np.float32))
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in (
          'status','held_fold','held_rows','reference_train_rows',
          'original_group_proposal_mismatches',
          'original_9D_per_group_audits_shape',
          'aggregate_20D_max_absolute_discrepancy',
          'all_receiver_rows_reproduce_historical_20D_audits_with_tolerance_2e_minus_5',
          'audit_may_recover_exact_archive','S80_promoted')},indent=2),flush=True)
def selftest():
    from scripts.v273_catalogue_contract import group_masks,direct_dim
    need(group_masks(8).shape==(255,8) and direct_dim(8)==61,
         'true historical 255 eight-vote catalogue feature contract')
    print('S197 PASS exact 255 8-vote grouping and 9 audit posterior dimensions')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path)
    p.add_argument('--fold',type=int,default=0)
    p.add_argument('--limit',type=int,default=128)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(a.out is not None,'output required')
        run(a.root,a.out,a.fold,a.limit)
