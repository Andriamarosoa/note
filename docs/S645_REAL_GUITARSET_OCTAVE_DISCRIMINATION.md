# S643 → S644 → S645 — distinguer une vraie note à l'octave de son harmonique

## Question expérimentale

La raie spectrale de **E3 (164,8 Hz)** est présente dans une vraie note
**E2 (82,4 Hz)** même si E3 n'est pas jouée : c'est son harmonique 2.
Comment distinguer `E2` de `E2+E3` dans l'architecture native NOTE
sans détruire ses détections polyphoniques ?

La réponse testée : **une tête de comparaison binaire** basée sur le
spectre original à plusieurs résolutions, les rapports entre partiels
pairs et impairs et leur évolution temporelle.

**ATTENTION :** les résultats de S643, S644, S645 concernent cette seule
classification binaire, PAS le F1 de transcription MIDI du réseau NOTE.

## S643 — Expérience contrôlée, vraies prises guitaristiques mixées

- Code : `scripts/s643_octave_duet_head.py`.
- Run réussi :
  https://github.com/Andriamarosoa/note/actions/runs/38998327282
- GuitarSet **TRAIN** uniquement : 162 extraits de notes relativement
  isolées, issus de 9 compositions d'entraînement.
- Le protocole partage les 9 compositions en deux ensembles totalement
  disjoints (4/5). Une paire est constituée de deux vraies prises de
  guitares enregistrées indépendamment, dont on fait la somme **au
  niveau PCM** : ce ne sont PAS des accords effectivement joués ensemble.
- 200 paires de test dans les deux directions, 400 observations
  binaires équilibrées (200 note seule, 200 duo).
- Tests sous provenance distincte A→B et B→A :
  - A→B : F1 88,04%, TP81 FP9 FN13 TN85.
  - B→A : F1 79,22%, TP101 FP48 FN5 TN58.
  - Total hors composition : **TP182 FP57 FN18 TN143, F1 82,92%**.
- En ajout décalé de 120ms, un seul sens de validation :
  **F1 94,38%, TP84 FP0 FN10 TN94**.
  Attention : résultat plus favorable mais pas équilibré sur les deux
  sens de provenance.

## S644 — Transfert sans adaptation aux vrais morceaux GuitarSet DEV

- Code : `scripts/s644_natural_octave_transfer.py`.
- Run réussi :
  https://github.com/Andriamarosoa/note/actions/runs/38998555912
- S643 réentraîné uniquement sur TRAIN; gelé et testé sur les **vrais
  audios microphone GuitarSet DEV**, 12 enregistrements.
- Une octave annotée positive : note supérieure commencée dans ±30ms
  du début de la note grave et soutenue ≥80ms.
  Les octaves commencées avant ou ajoutées plus tard sont exclues.
- 361 attaques évaluables : 28 positives, 333 négatives.
- **TP11 FP15 FN17 TN318 → précision 42,31%, rappel 39,29%, F1 40,74%**.
- Le 91,14% accuracy serait trompeur du fait du déséquilibre.
- Les 28 vrais duos proviennent de contextes **K3+** et 20 des 28
  se situent dans un seul enregistrement `04_Rock1-90-C#_comp`.
- La distribution réelle change complètement le verdict S643 :
  les mélanges artificiels ne reproduisent pas suffisamment les
  accords naturels multi-hauteurs, le timbre et les résonances.

## S645 — Réentraîner exclusivement sur les VRAIS accords naturels

- Code : `scripts/s645_learn_real_octave.py`.
- Run réussi :
  https://github.com/Andriamarosoa/note/actions/runs/38998824657
- `36 GuitarSet TRAIN` → **1466 attaques réelles**, dont **199
  vraies attaques simultanées à l'octave**.
- 5-fold `GroupKFold` par composition entière, choix de la famille
  de modèle et du seuil uniquement avec les résultats OOF TRAIN.
- Architecture sélectionnée : `StandardScaler + LogisticRegression`
  sur 173 caractéristiques harmoniques multi-résolution, `C=0.2`,
  `class_weight=balanced`.
- Seuil fixé avant DEV : **0,80**.
- F1 hors-composition sur TRAIN : **53,44%**
  (TP97 FP67 FN102 TN1200).

### Évaluation sur GuitarSet DEV naturel

| Variante | TP | FP | FN | TN | F1 | Précision | Rappel |
|---|---:|---:|---:|---:|---:|---:|---:|
| S644 tête synthétique → réel | 11 | 15 | 17 | 318 | 40,74% | 42,31% | 39,29% |
| **S645 tête réentraînée sur vrai** | **19** | **14** | **9** | **319** | **62,30%** | **57,58%** | **67,86%** |
| S645 sous-ensemble caractéristiques | 18 | 17 | 10 | 316 | 57,14% | 51,43% | 64,29% |
| S645 gradient boosting | 24 | 28 | 4 | 305 | 60,00% | 46,15% | 85,71% |

**Contexte K3+ uniquement** (229 attaques, 28 duos) :
- TP19, FP7, FN9, TN194 ;
- **F1 70,37%, précision 73,08%, rappel 67,86%**.

Les seuils et la famille ont été choisis par TRAIN, pas sur DEV.
Le meilleur sur DEV n'est donc pas sélectionné artificiellement.

### Limites qui restent majeures

1. **E2/E3 spécifiquement** n'a pas de vrai couple positif annoté
   exploitable dans ce DEV selon notre critère ; le seul contexte E2
   évalué est négatif et S645 le classe correctement.
   Les chiffres mesurés concernent plusieurs paires d'octaves, pas une
   validation directe des accords E2/E3.
2. Seuls 28 duos naturels positifs apparaissent sur DEV, concentrés
   sur un petit nombre de morceaux. Intervalle de confiance large ;
   DEV a été historiquement étudié.
3. Les features observées jusqu'à **190 ms après l'attaque réelle**.
   Elles sont causales vis-à-vis du micro, mais incompatibles à ce stade
   avec une émission strictement immédiate d'un NoteOn.
4. Le classifieur S645 n'a PAS été greffé au décodeur MIDI NOTE :
   il ne produit que `octave_present` à des attaques déjà connues
   via annotation dans cet audit. En production, il faudra générer
   les mêmes candidats sans connaître les notes vraies.
5. Les 36 compositions TRAIN et 12 DEV sont originales ; aucun TEST
   scellé n'a été évalué, aucune performance de généralisation sur
   instruments autres que GuitarSet n'est attestée.
6. Le S614, le S641 et le LIVE NOTE original restent **inchangés**.

## Prochaine étape technique

- Réutiliser ce classifieur comme **tête de sélection candidate**
  alimentée par les attaques NOTE proposées sans annotations, en
  régularisant les décisions en competition octave.
- Réduire progressivement la durée d'observation après attaque :
  190ms -> 90ms -> 40ms -> 0ms avec un audit strict de la baisse
  de précision, du temps CPU et des files micro.
- Ne pas forcer un veto octave global : les vraies octaves sont des
  notes superposées légitimes. Utiliser la décision probabiliste
  d'abstention par candidat.
- Mesurer l'effet final sur **TP/FP/FN/F1 NoteOn Global** et sur K2/K3+
  sur des compositions indépendantes et un autre instrument, sans
  prétendre qu'un F1 de classification binaire = un F1 MIDI.
