"""Annotation-conditioned dictionary probe; never an inference-time feature."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from causal_note.guitarset import index_guitarset,SAMPLE_RATE
from causal_note.guitarset_acoustics import load_rich_annotations
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_internal_residual_acoustics import match_frequencies,decompose
from scripts.v273_residual_audit import require,write_json,sha256_file


def run(a):
    rows=[json.loads(s) for s in a.cases.read_text().splitlines()]
    rows=[r for r in rows if r['true_K']==3]
    require(all(r['fold'] in (0,1,2,4) for r in rows),'outer fold')
    wanted={r['recording_id'] for r in rows}
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    results=[]
    for member in sorted(wanted):
        track=tracks[member]
        rich=load_rich_annotations(track.annotation_zip,member)
        pcm=decode_pcm16_mono_wav(track.audio_zip,track.audio_member)
        samples=np.asarray(pcm.samples,np.float64)/32768
        for r in [q for q in rows if q['recording_id']==member]:
            start=r['start_sample'];end=start+h.WINDOW
            freq,x=h.transition_spectrum(samples,start)
            annotated=np.array([n['frequency_hz'] for n in r['owned_notes']])
            contour_f0=[];contour_counts=[];cents=[]
            for n in r['owned_notes']:
                values=[p.frequency_hz for p in rich.contours_by_slot[n['slot']]
                        if p.voiced and max(start,n['onset_sample'])<=p.sample<min(end,n['offset_sample'])]
                contour_counts.append(len(values))
                v=float(np.median(values)) if values else None
                contour_f0.append(v)
                cents.append(float(abs(1200*np.log2(v/n['frequency_hz']))) if v is not None else None)
            base=r['decomposition']['pool_f0']
            pool=[*base,*annotated.tolist()]
            pair,trip,x2=decompose(freq,x,pool)
            selected=np.array(pool)[list(trip[1])]
            matched,_=match_frequencies(annotated,selected)
            have=[v for v in contour_f0 if v is not None]
            contour_match,_=match_frequencies(have,base)
            results.append({'row_id':r['row_id'],'fold':r['fold'],'recording_id':member,'group':r['group'],
                            'annotation_f0':annotated.tolist(),'contour_f0':contour_f0,
                            'contour_counts':contour_counts,'contour_annotation_cents':cents,
                            'contour_pool_matches':contour_match,'contour_note_count':len(have),
                            'augmented_selected':selected.tolist(),'augmented_owned_matches':matched,
                            'augmented_residual':float(trip[0]/(x2+1e-12)),
                            'annotated_candidates_selected':sum(i>=len(base) for i in trip[1]),
                            'original_pool_matches':r['measurements']['pool_owned_matches']})
        print(member,len(results),flush=True)
    summary={}
    for group in ['K3_regressed','K3_preserved']:
        rr=[r for r in results if r['group']==group]
        ds=[c for r in rr for c in r['contour_annotation_cents'] if c is not None]
        summary[group]={'rows':len(rr),'contour_available_notes':len(ds),'total_notes':3*len(rr),
                        'median_contour_annotation_cents':float(np.median(ds)),
                        'notes_contour_deviation_over_55_cents':sum(c>55 for c in ds),
                        'contour_all_three_in_original_pool':sum(r['contour_note_count']==3 and r['contour_pool_matches']==3 for r in rr),
                        'augmented_triplet_all_three':sum(r['augmented_owned_matches']==3 for r in rr),
                        'mean_augmented_matches':float(np.mean([r['augmented_owned_matches'] for r in rr]))}
    write_json(a.output,{'status':'completed','outer_fold_3_used':False,'diagnostic_only':True,
                        'source_cases_sha256':sha256_file(a.cases),'summary':summary,'cases':results})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cases','dataset','output']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
