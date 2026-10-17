# S662–S664 — NOTE Offset : objectif 80 %

## Résultats vérifiés sur GuitarSet historique

Métrique : NoteOff correct à ±50 ms parmi les 25 395 NoteOn appariés
par hauteur et début ±50 ms, sur 190 pistes GuitarSet.

| Variante | Offsets corrects | Précision | MAE |
|---|---:|---:|---:|
| S425 | 16 543 | 65,1427 % | 87,60 ms |
| S662 résidu ×0,5 | 16 306 | 64,2095 % | 87,13 ms |
| S663 classement durée | 16 641 | 65,5286 % | 87,55 ms |
| S664 acoustique + classement | **16 777** | **66,0642 %** | **86,48 ms** |

S664 gagne 234 notes complètes nettes vs S425; par fold de composition
les gains sont +70, +59, +44, +61. Plusieurs réglages de pénalité
ont été comparés sur ces folds : le gagnant est encore exploratoire.

## Faisabilité et limites

L'objectif 80 % exige 20 316 corrects, encore **3 539** notes de plus.
L'oracle qui sélectionne avec l'annotation la meilleure des seules trois
durées S420/S422/S425 atteint 17 368/25 395 (68,3914 %).
Un oracle diagnostique sur durées interpolées et étirées atteint
21 857/25 395 (86,0681 %). Cet oracle utilise la vérité au choix et
n'est absolument PAS un score d'inférence.

S662 : régression résiduelle apprise sur autres compositions, rejetée.
S663 : classement probabiliste par durée candidate.
S664 : ajoute les trajectoires spectrotemporelles MIC S411; alignement
S411/S425 par identifiant événement et MIDI, et non simple position.

K original S80 de 59 309 événements, 33 475 notes MIDI et leurs Onsets
inchangés. Aucune promotion, aucun résultat strict LIVE nouveau. S411
utilise une STFT centrée et les sorties S425 peuvent avoir 5 secondes
de contexte futur. La cohorte historique GuitarSet a été exposée; les
modèles amont S425 ne sont pas intégralement nested. Les scores ne
constituent donc pas une validation prospective indépendante.

## Suite structurelle requise

Une nouvelle tête temporelle pour chaque note active doit estimer
une probabilité de NoteOff par bloc à partir du spectre strictement
passé, des notes concurrentes, du K prédit, de la décroissance et
de la mémoire de survie. Elle doit traiter les passages K0 sans
nouvel Onset et ne jamais antidater les événements. Mesurer les
notes longues ≥500 ms et ≥1s, les accords K2/K3+, la précision
NoteOff ±50 ms, le F1 de note complète et la latence microphone.
Ne pas promouvoir sans test indépendant et sans protection Exact-K.

## Preuves

- S425 : https://github.com/Andriamarosoa/note/actions/runs/38513725784
- S662 : https://github.com/Andriamarosoa/note/actions/runs/39023125833
- S663 : https://github.com/Andriamarosoa/note/actions/runs/39023274305
- S664 : https://github.com/Andriamarosoa/note/actions/runs/39023531830
