"""Verify an eight-state training step and unchanged historical model defaults."""
import argparse
from pathlib import Path
import numpy as np
from causal_note.count_states import encode_states, decode_probabilities, state_metrics
from scripts.rebuild_v273_sources import digest, write_json
from scripts.v273_high_pitch_native import build as historical_build
from scripts.v273_silence_state_model import build_model
from scripts.v273_spectral_normalization import layer_hashes
from scripts.train_v273_window_pair import weight_hash


def run(output):
    import tensorflow as tf
    if output.exists():raise FileExistsError(output)
    if tf.__version__!='2.15.1':raise RuntimeError('TensorFlow 2.15.1 required')
    tf.config.experimental.enable_op_determinism()
    old=historical_build();old_hash=weight_hash(old);old_layers=layer_hashes(old)
    assert old.output_shape==(None,7)
    model=build_model();new_layers=layer_hashes(model)
    assert model.output_shape==(None,8)
    assert all(new_layers[n]==v for n,v in old_layers.items() if n!='cardinality')
    assert 'cardinality' not in new_layers and 'count_state' in new_layers
    initial=weight_hash(model)
    rng=np.random.default_rng(9302026)
    x={t.name.split(':')[0]:rng.uniform(0,1,(8,*t.shape.as_list()[1:])).astype(np.float32)
       for t in model.inputs}
    x['candidate_mask'].fill(1)
    truth=np.arange(-1,7);encoded=encode_states(truth)
    before=model(x,training=False).numpy()
    decode_probabilities(before)
    log=model.train_on_batch(x,encoded,return_dict=True)
    assert all(np.isfinite(v) for v in log.values()) and weight_hash(model)!=initial
    after=model(x,training=False).numpy();prediction=decode_probabilities(after)
    synthetic_metrics=state_metrics(truth,prediction)
    assert model.count_params()==249379
    assert weight_hash(historical_build())==old_hash
    output.mkdir(parents=True)
    report=dict(status='passed',tensorflow=tf.__version__,numpy=np.__version__,
        states=list(range(-1,7)),encoded_classes=list(range(8)),parameters=249379,
        initial_sha256=initial,historical_seven_class_hash=old_hash,
        shared_encoder_initialization_unchanged=True,historical_defaults_unchanged=True,
        all_eight_classes_finite_training_step=True,synthetic_metrics=synthetic_metrics,
        real_data_training_performed=False,output_corrector=False,
        state_builder_sha256=digest('src/causal_note/count_states.py'),
        model_builder_sha256=digest('scripts/v273_silence_state_model.py'),
        script_sha256=digest(__file__))
    write_json(output/'report.json',report)
    print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
