# S638 — Les notes réellement superposées de GuitarSet

## Périmètre vérifié

- Run réussi: https://github.com/Andriamarosoa/note/actions/runs/38936304609
- Branche: `codex/s638-guitarset-polyphonic-overlap-audit`.
- 12 pistes audio **GuitarSet originales** (284,294 s), avec annotations MIDI.
- DEV seulement, compositions BN3/Funk3/Rock1; TEST non lu. DEV est
  historiquement exposé, donc diagnostic et non généralisation indépendante.
- NOTE S614 gelé + unique tête native K, même trame 256, **zéro lookahead**.
- Trois vues: ×1 original; ×2 fréquence pour les graves; ÷2 pour les aigus.
- Annotation utilisée EXCLUSIVEMENT après prédiction, pour scoring TP/FP/FN.
- `onset_overlap`: autre hauteur annotée déjà active au début, en
  acceptant une tolérance de 30 ms entre attaques d'accord, et >=20 ms
  de recouvrement réel. `any_overlap`: recouvrement >=20 ms à n'importe
  quel moment de la note. Polyphonie K2/K3+ calculée sur annotations,
  non à partir de la tête K prédite.
- FP attribué à un passage polyphonique si >=2 hauteurs annotées sont
  actives à l'instant de la fausse attaque (±30 ms autour des bornes).
  Cette attribution n'utilise pas le futur du moteur LIVE.

## Rappel notes jouées simultanément, et F1 global

| Variante | TP note avec overlap dès attaque / 1147 | Rappel overlap | TP K2 / 237 | TP K3+ / 910 | FP pendant passage poly | F1 global |
|---|---:|---:|---:|---:|---:|---:|
| NOTE S614 original | 697 | 60,77% | 147 | 550 | 416 | 63,10% |
| Bass ×2 @0,80 | 740 | **64,52%** | 154 | 586 | 603 | 60,38% |
| Bass ×2 @0,90 | 717 | 62,51% | 152 | 565 | 474 | 62,59% |
| High ÷2 @0,80 | 697 | 60,77% | 147 | 550 | 489 | 58,86% |
| Both @0,80 | 740 | 64,52% | 154 | 586 | 676 | 56,65% |
| Both @0,80 + octave-veto/K | 721 | 62,86% | 150 | 571 | 545 | 59,67% |

### Corrections et régressions au niveau VRAIE note

À seuil Bass 0,80, sur les 1147 notes annotées avec superposition dès
l'attaque :
- **44 notes vraies nouvellement retrouvées**, 1 auparavant correcte perdue
  => **+43 TP nets sur les accords**, rappel +3,75 points.
- Parmi ces 44 récupérées, **37 sont K3+**, 7 sont K2; la perte
  concerne K3+.
- 3 autres notes isolées nouvellement retrouvées, 0 perdue.
- 416 -> 603 faux NoteOn apparaissent durant de vrais passages
  polyphoniques (**+187**); au total 585 -> 834 FP (+249).
- F1 global 63,1046% -> 60,3817% (**−2,723 points**).

Bass 0,90: 20 notes nouvellement retrouvées avec une attaque superposée,
0 perdue (K2 +5, K3+ +15); mais 416 -> 474 FP dans les passages poly
(+58) et 585 -> 655 FP au total (+70). F1 global −0,516 point.

High ÷2 @0,80: **0** vrai NoteOn de plus, +228 FP globaux.
GuitarSet DEV contient seulement 1 note E5+, donc ne permet pas
d'exclure un intérêt pour les aigus sur URMP/une autre source.

## Lecture

- **L'hypothèse physique est partiellement validée**: la transposition ×2
  aide à détecter des graves au sein d'accords complexes.
- Mais elle n'a PAS une précision suffisante et diminue le F1 global.
  Les taux de rappel ci-dessus ne sont PAS des F1 propres aux accords.
- Les 910 notes en contexte K3+ représentent l'essentiel des cas
  difficiles, où le correcteur Bass0,80 gagne +36 TP nets
  (37 récupérées / 1 perdue).
- Une tête sélective, entraînée uniquement sur compositions TRAIN,
  devra choisir sur base d'évidence native, fondamentale physique,
  compétition octave et K, et apprendre l'abstention.
- Ne pas activer `guard` dans le LIVE et ne pas promouvoir S638.
  Après entraînement et contrôle DEV, vérifier sur corpus indépendants,
  notamment URMP et IDMT pour la diversité instrumentale.

Le rapport complet est l'artefact
`S638_GUITARSET_REAL_OVERLAP_TP_FN_FP/report.json`
du run GitHub cité. Les métriques par piste y sont conservées.
