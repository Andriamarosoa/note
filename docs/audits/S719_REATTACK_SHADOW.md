# S719 — Expérience ciblée : nouvelles attaques sur une note déjà active

S719 est un **candidat shadow uniquement**, jamais une modification
de la référence `GuidedSelector` S708. Il répond au blocage démontré
dans le code : une fois l'Onset émis, `armed=False` exige normalement
que la probabilité descende sous `onset*0.4` pour permettre la
réattaque. Une nouvelle montée nette peut être ignorée si ce retour
sous le seuil ne s'est pas produit.

## Modification instrumentée

`ShadowReattackSelector` conserve toutes les conditions existantes
et n'ajoute qu'un réarmement **avec preuve acoustique actuelle** :
- Onset >= `policy.onset` (S708 = 0.75).
- Sustain >= `policy.sustain_on` (S708 = 0.50).
- Release < 0.50.
- Montée Onset depuis le **bloc causal précédent** >= `rise`.
- Temps depuis la dernière attaque même hauteur >= `min_gap_ms`.
- Le pitch était effectivement désarmé ; aucun événement supplémentaire
  sur un signal Onset continuellement élevé.

Les essais CAL explorent seulement `rise=0.10, 0.20, 0.30`
et `min_gap_ms=90, 150`. L'ancien S708 reste la 7e option.
Le modèle S708, la tête K, les caractéristiques harmoniques, S707
et tous les paramètres de la politique historique sont figés.

## Validation sans régression

Lancement : [workflow S719](https://github.com/Andriamarosoa/note/actions/workflows/note-s719-reattack-cal.yml).

La sélection est faite sur CAL **uniquement**, à partir des NoteOn/NoteOff
réellement émis avec S707 et des six scores :
`onset_f1`, `exact_k`, `active_exact_k`, `offset_accuracy`,
`full_note_f1`, `frame_pitch_f1`.

Une politique n'est éligible que si aucune des six métriques ne
régresse face au témoin CAL, aucun Offset historiquement correct
n'est perdu et aucune attaque correcte de note longue n'est perdue.
Sinon `S708_frozen` est conservé. La meilleure politique CAL
est figée avant de regarder les diagnostics HELD/DEV ; ces
ensembles sont historiquement exposés, pas un test inédit.

Il n'y a **ni fusion, ni promotion, ni branche LIVE modifiée**
avant reproduction indépendante, et le rapport final signale
explicitement `promoted=false`.

## Limites

Le signal Onset sur une hauteur peut monter en raison d'une
harmonique d'une AUTRE corde. S719 n'affirme pas distinguer
ces cas avec certitude. La faiblesse des fondamentales graves
MIDI40–52 et les accords K2–K5 restent prioritaires.
Le sélecteur shadow est volontairement expérimental : toute
régression en F1, NoteOff, exact K, latence ou événement de
réattaque le disqualifie.
