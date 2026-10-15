"""Restore original NOTE checkpoints and validation ledgers with GitHub CLI.

Needs `gh` authenticated for repository Actions access. No training or model
conversion. Original model and ledger SHA256 are checked by the validator.
"""
import argparse
from pathlib import Path
import subprocess

ARTIFACTS = (
    (38675146653, 's578-28-real-composer-midi-trained-global52-and-mid12-52D-before-S577-WAV', 'S578'),
    (38678912773, 's585-28-source-four-state-training', 'S585'),
    (38679314946, 's586-s589-sixth-audio-only', 'sixth'),
    (38679314946, 's590-sixth-three-policies-PREMIDI', 'S590'),
    (38681794058, 's592-s595-urmp-AUDIO-ONLY-NO-NOTES', 'urmp'),
    (38681794058, 's596-urmp-predictions-BEFORE-ANNOTATIONS', 'S596'),
    (38681794058, 's597-urmp-independent-noteon-verdict', 'S597'),
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--models-only', action='store_true')
    a = p.parse_args()
    if a.out.exists():
        p.error('Use a new output directory; original artifacts will not be overwritten')
    a.out.mkdir(parents=True)
    for run, name, folder in ARTIFACTS[:2] if a.models_only else ARTIFACTS:
        subprocess.run(['gh', 'run', 'download', str(run), '--repo', 'Andriamarosoa/note',
                        '--name', name, '--dir', str(a.out/folder)], check=True)
    from causal_note.recent_native import RecentHeads
    print('Original trained weights verified:', RecentHeads(a.out).hashes)


if __name__ == '__main__':
    main()
