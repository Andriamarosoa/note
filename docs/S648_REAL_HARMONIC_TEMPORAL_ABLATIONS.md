# S648 — Audits de compétition harmonique et de nouveauté temporelle

## Population, preuve et verdict

GitHub CI **réussi** :
https://github.com/Andriamarosoa/note/actions/runs/39001321416

Artefact `S648_REAL_GUITARSET_HARMONIC_RESIDUAL_LOOPS/report.json`.
Aucun nouveau modèle NOTE officiel, aucune modification Exact-K ni du LIVE.
Les 36 enregistrements GuitarSet TRAIN originaux (9 compositions)
entraînent les alternatives ; sélection des seuils et de la famille
par validation croisée **GroupKFold** sur compositions ; puis scoring
sur les 12 pistes originales GuitarSet DEV annotées. TEST non consulté.

## Pistes étudiées

1. `control41` : sélection S641 (native + vue ×2) identique ;
2. `competitive65` : information statique de compétition et de
   fondamentale indépendante ;
3. `temporal41` : nouveautés propres au pitch, à l'octave et aux
   voisins dans les 4 et 12 hops **précédents** (env. 23/70ms) ;
4. `temporal65` : combinaison physique et temporelle ;
5. `tree_temporal65` : petit classifieur non linéaire sur 125 features.

Chaque candidat est un NoteOn réellement proposé par le Bass Rescue
sans garde-fou. Le label positif de TRAIN signifie que l'événement
sauve une annotation auparavant ratée par S614 ; les doublons
et fausses attaques sont négatifs. Les caractéristiques ne reçoivent
aucune annotation à l'inférence ; seuls le signal natif et les
probabilités NOTE/K sont utilisés.

## Résultats DEV GuitarSet

| Politique | TP | FP | FN | F1 | overlap NoteOn TP | K3+ TP |
|---|---:|---:|---:|---:|---:|---:|
| S614 gelé | 998 | 585 | 582 | 63,10465% | 697 | 550 |
| Bass0,80 sans sélection | 1044 | 834 | 536 | 60,38172% | 740 | 586 |
| **S641 témoin control41** | **1009** | **594** | **571** | **63,39931%** | **707** | **557** |
| competitive65 | 1003 | 586 | 577 | 63,30073% | 702 | 552 |
| temporal41 | 1004 | 594 | 576 | 63,18439% | 703 | 553 |
| temporal65 | 1004 | 587 | 576 | 63,32387% | 702 | 552 |
| tree_temporal65 | 1010 | 603 | 570 | 63,26339% | 707 | 557 |

Les 5-fold TRAIN portent sur **761 propositions effectivement émises
en rescue, dont seulement 50 vrais nouveaux NoteOn**.
Seuils sélectionnés sur TRAIN avec utilité TP − 0,5·FP :
- S641 témoin `control41` : 0,90, 17 TP / 20 FP hors-composition ;
- `competitive65` : 0,97, 9 / 7 ;
- `temporal41` : 0,92, 16 / 17 ;
- `temporal65` : 0,95, 10 / 13 ;
- **`tree_temporal65`** : 0,70, 19 / 20 et utilité 9 ; gagnant TRAIN.

Le meilleur gagnant TRAIN (tree_temporal65) **régresse** sur DEV
par rapport au contrôle S641 : une seule vraie correction
supplémentaire mais 9 FP de plus, malgré un bon score TRAIN.
**Rejeté, aucune promotion.**

## Interprétation et limites

La nouveauté temporelle seule ne suffit pas à déterminer si des
composantes augmentent en raison d'une vraie fondamentale ou à cause
de l'attaque d'une autre note. Les gabarits des partiels entre notes
superposées se recouvrent et la solution reste très déséquilibrée :
50 TP nouveaux parmi 761 candidats TRAIN.
DEV a été examiné historiquement : ce n'est pas une preuve
indépendante de généralisation.
Piste distincte testée en S649 : séparer soutien **exclusif**
et **partagé** des partiels à partir d'une FFT causale de la fenêtre
originale, puis vérifier le gain en vrai NoteOn F1.
