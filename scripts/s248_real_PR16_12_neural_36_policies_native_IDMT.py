"""S248 genuine original 12 PR16 frozen neural heads and 36 original PR16
policies on EXACT 1034 unseen IDMT V88 native event IDs.

Source S234 waveform summary58/sequence42x49, S236 original true V88
ownership geometry46, and original official S235 frozen freeze_K. Uses
original trained fold-0 normalizers and 12 original 58D/2162D network
weight snapshots. Implements original S82 numpy_forward + original S85
original_pr16_policies unchanged; no XML/MIDI, no new training or score.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s82_external_pr16_neural_inference import numpy_forward
from scripts.s85_external_pr16_policies import original_pr16_policies,ALPHAS
def must(v,msg):
    if not v:raise ValueError(msg)
def load(p,keys):
    with np.load(p,allow_pickle=False) as z:
        must(set(z.files)==set(keys),'S248 only original label-free evidence fields')
        return {k:z[k].copy() for k in z.files}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(s234file,s236file,model_dir,out):
    must(not out.exists(),'immutable S248 reference neural forward cannot overwrite')
    a=load(s234file,{'row_id','member','start_samples','incumbent_freeze_K',
       'summary58','sequence42x49','flow245','morphology490','physics294'})
    b=load(s236file,{'row_id','member','start_samples','end_samples',
       'incumbent_freeze_K','original_PR16_geometry46',
       'original_S5_input_owned527'})
    ids=a['row_id'];n=len(ids)
    must(n==1034 and np.array_equal(ids,b['row_id']) and
         np.array_equal(a['member'],b['member']) and
         np.array_equal(a['start_samples'],b['start_samples']) and
         np.array_equal(a['incumbent_freeze_K'],b['incumbent_freeze_K']),
         'true original 1034 V88 native official incumbent PR16 feature IDs mismatch')
    x58=a['summary58'].astype(np.float32)
    s42=a['sequence42x49'].astype(np.float32)
    g46=b['original_PR16_geometry46'].astype(np.float32)
    must(x58.shape==(n,58) and s42.shape==(n,42,49) and
         g46.shape==(n,46) and np.isfinite(x58).all() and
         np.isfinite(s42).all() and np.isfinite(g46).all(),
         'same original S82 PR16 58D 42x49 46D observations required')
    x2162=np.column_stack((x58,s42.reshape(n,-1),g46)).astype(np.float32)
    must(x2162.shape==(n,2162),'original PR16 2162D source input shape')
    provenance=[];network=[];names=[]
    for view,raw in (('summary58',x58),('trajectory',x2162)):
        scaler=model_dir/(view+'-normalizer.npz')
        must(scaler.is_file(),'original trained PR16 normalizer absent')
        with np.load(scaler,allow_pickle=False) as z:
            must(set(z.files)=={'mean','scale'},'original PR16 mean/scale fields changed')
            mean=z['mean'];scale=z['scale']
        must(mean.shape==scale.shape==(raw.shape[1],) and np.all(scale>0),
             'original PR16 normalizer feature shape')
        transformed=raw.copy().astype(np.float32)
        transformed-=mean.astype(np.float32)
        transformed/=scale.astype(np.float32)
        np.clip(transformed,-6,6,out=transformed)
        for weight in (1,1.5):
            for epoch in (4,8,12):
                key=f'{view}__poly{weight:g}__epoch{epoch}'
                file=model_dir/(key+'.weights.npz')
                must(file.is_file(),'original PR16 checkpoint missing '+key)
                with np.load(file,allow_pickle=False) as z:
                    weights=[z['w'+str(i)].copy() for i in range(len(z.files))]
                must(len(weights)==6,'original PR16 trained network 3 dense layers')
                q=numpy_forward(transformed,weights)
                must(q.shape==(n,7) and np.isfinite(q).all() and
                     np.allclose(q.sum(1),1,atol=1e-5),
                     'real original 12 neural head outputs invalid')
                network.append(q);names.append(key)
                provenance.append(dict(name=key,network_SHA256=sha(file),
                    original_normalizer_SHA256=sha(scaler)))
    probs=np.stack(network,axis=1)
    incumbent=b['incumbent_freeze_K'].astype(np.int8)
    choices=original_pr16_policies(probs,incumbent)
    must(probs.shape==(n,12,7) and choices.shape==(n,36) and
         np.isin(choices,range(7)).all(),'exact original 12 neural x 3 policy classes mismatch')
    vote=np.column_stack([(choices==k).sum(axis=1) for k in range(7)])
    must(np.all(vote.sum(axis=1)==36),'PR16 36-policy vote mass')
    out.mkdir(parents=True)
    file=out/'pr16_external_36_policies.npz'
    np.savez_compressed(file,member=b['member'],start_samples=b['start_samples'],
         end_samples=b['end_samples'],incumbent_freeze_K=incumbent,
         pr16_policy_K=choices,pr16_vote_count=vote,
         pr16_policy22_K=choices[:,22],
         pr16_network_probabilities=probs)
    manifest=dict(status='S248_GENUINE_ORIGINAL_PR16_12_MLP_36_POLICY_ON_IDMT_NATIVE',
       original_reference_native_event_count=n,
       original_trained_network_checkpoints=12,
       original_12_normalizer_reuse=True,
       original_36_policy_definitions_exact=True,
       original_S85_alpha_values=list(ALPHAS),
       original_neural_head_SHA256=provenance,
       PR16_policy22_changed_vs_freeze=int(np.count_nonzero(choices[:,22]!=incumbent)),
       original_PR16_policy_event_changes_vs_freeze=int(np.count_nonzero(choices!=incumbent[:,None])),
       output_SHA256=sha(file),no_original_weights_changed=True,
       native_event_identity_matches_S234_S236=True,
       models_retrained=False,XML_MIDI_content_read=False,
       S69_S80_not_yet_replayed=True,independent_accuracy_measured=False,
       S80_promoted=False)
    (out/'report.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='original_neural_head_SHA256'},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('s234','s236','models','out'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    run(a.s234,a.s236,a.models,a.out)
