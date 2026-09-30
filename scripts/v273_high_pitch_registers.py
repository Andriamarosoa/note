"""Evaluation-only MIDI strata for the fixed inner validation population."""
import json
from pathlib import Path
import zipfile
import numpy as np

from scripts.audit_v273_control_inputs import nearest_assignment
from scripts.rebuild_v273_sources import digest, write_json
from scripts.v273_window_experiment import require

ANNOTATION_SHA256 = '8daa02e6417ccca1685feb44b135e95928ad7037e5032ecb326b5791856fda99'


def assigned_pitches(groups, events):
    """events are (onset_sample, MIDI value); pitches never enter model inputs."""
    flat=np.concatenate(groups).astype(np.int64)
    owners=np.repeat(np.arange(len(groups)),[len(g) for g in groups])
    order=np.argsort(flat,kind='stable')
    rows=[[] for _ in groups]
    unassigned=0
    for onset,pitch in events:
        require(np.isfinite(pitch) and 0<=pitch<=127,'invalid annotation MIDI')
        assigned=nearest_assignment(int(onset),flat[order],owners[order])
        if assigned is None:
            unassigned+=1
        else:
            rows[assigned[0]].append(int(np.rint(pitch)))
    require(all(len(row)<=6 for row in rows),'unexpected raw K>6')
    pitches=np.full((len(groups),6),-1,np.int16)
    for i,row in enumerate(rows):
        pitches[i,:len(row)]=sorted(row)
    return pitches,np.array([len(row) for row in rows],np.int32),unassigned


def masks(k,pitches):
    k=np.asarray(k);p=np.asarray(pitches)
    require(p.shape==(len(k),6),'invalid MIDI metadata shape')
    np.testing.assert_array_equal((p>=0).sum(1),k)
    bass=np.any((p>=0)&(p<48),axis=1)
    high=np.any(p>=72,axis=1)
    mid=np.any((p>=48)&(p<72),axis=1)
    octave=np.zeros(len(k),bool)
    for i in range(6):
        for j in range(i+1,6):
            d=p[:,j]-p[:,i]
            octave|=(p[:,i]>=0)&(p[:,j]>=0)&(d>0)&(d%12==0)
    return dict(any_bass=bass,any_high=high,only_bass=bass&~mid&~high,
        only_mid=mid&~bass&~high,only_high=high&~bass&~mid,
        mixed_registers=(bass.astype(int)+mid.astype(int)+high.astype(int))>1,
        octave_pair=octave,k3_bass=(k==3)&bass,k3_no_bass=(k==3)&~bass)


def prepare(annotations,geometry,cache,validation,output):
    require(digest(annotations)==ANNOTATION_SHA256,'annotation archive changed')
    report=json.loads((geometry/'report.json').read_text())
    members=np.asarray(cache['members'][validation])
    positions={int(g):i for i,g in enumerate(validation)}
    pitches=np.full((len(validation),6),-1,np.int16)
    counts=np.full(len(validation),-1,np.int32)
    allowed=set(members.astype(str));seen=set();unassigned=events_total=0
    with zipfile.ZipFile(annotations) as archive:
        for record in report['tracks']:
            member=record['member']
            if member not in allowed:continue
            require(member not in seen,'duplicate track');seen.add(member)
            path=geometry/record['timing_file']
            require(digest(path)==record['timing_sha256'],'candidate timing changed')
            with np.load(path,allow_pickle=False) as z:
                ids,flat,offset=z['global_index'],z['full_candidate_samples'],z['full_candidate_offsets']
                require(str(z['member'][0])==member,'wrong timing member')
            np.testing.assert_array_equal(cache['members'][ids],np.repeat(member,len(ids)))
            groups=[flat[a:b] for a,b in zip(offset[:-1],offset[1:])]
            np.testing.assert_array_equal([g[0] for g in groups],cache['cluster_start_samples'][ids])
            doc=json.loads(archive.read(member));events=[]
            for a in doc['annotations']:
                if a['namespace']=='note_midi':
                    events.extend((round(d['time']*44100),float(d['value'])) for d in a['data'])
            p,k,missing=assigned_pitches(groups,events)
            np.testing.assert_array_equal(k,cache['exact'][ids])
            where=np.array([positions[int(g)] for g in ids])
            pitches[where]=p;counts[where]=k
            unassigned+=missing;events_total+=len(events)
    require(seen==allowed and len(seen)==50,'incomplete validation annotation inventory')
    np.testing.assert_array_equal(counts,cache['exact'][validation])
    masks(counts,pitches)
    np.savez_compressed(output,global_index=validation,member=members,k=counts,pitches_midi=pitches)
    notes=pitches[pitches>=0]
    return dict(path=Path(output).name,sha256=digest(output),rows=len(validation),tracks=len(seen),
        notes=int(len(notes)),midi_min=int(notes.min()),midi_max=int(notes.max()),
        note_register_counts=dict(bass=int((notes<48).sum()),mid=int(((notes>=48)&(notes<72)).sum()),high=int((notes>=72).sum())),
        annotated_events=events_total,unassigned_annotations=unassigned,
        k_histogram=np.bincount(counts,minlength=7).tolist(),labels_recomputed=True,
        evaluation_only=True,annotation_sha256=ANNOTATION_SHA256,
        meaning='assigned new note attacks, not all sounding notes; nearest MIDI semitone; bass<C3, high>=C5')
