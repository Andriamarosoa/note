# S650 — 4 têtes harmoniques × 18 sélections combinées pour NOTE

## Question et réponse

Les précédents audits S648 et S649 ajoutaient les pistes au **même vecteur
de caractéristiques** et apprenaient une seule tête. S650 conserve
**4 têtes entraînées indépendamment** et apprend leurs seuils avant
de tester **18 sélections distinctes** AND, OR, majorité,
consensus, veto/rattrapage et fusion de probabilités.

**Quelques combinaisons dépassent S641 sur GuitarSet DEV**, mais la
combinaison désignée par TRAIN ne le dépasse pas. **Aucune promotion.**

## Références, protocole et disponibilité

- CI **réussi** : https://github.com/Andriamarosoa/note/actions/runs/39002881964
- Script `scripts/s650_harmonic_head_selection_combos.py`
- Tests `test/test_s650_harmonic_head_selection_combos.py`
- Rapport complet `S650_GUITARSET_HARMONIC_HEAD_SELECTION_COMBOS/report.json`.
- 36 pistes GuitarSet TRAIN (9 compositions différentes),
  puis 12 GuitarSet DEV déjà exposées.
- Ensemble TRAIN réel : **761 candidats émis par le rescue sans
  garde**, dont **50 corrections inédites** ; les autres sont
  négatifs. Le classifieur ne reçoit jamais les annotations
  à l'inférence.
- 5-fold CV par composition entière : chaque tête est entraînée
  sur les autres compositions, puis émet des probabilités OOF.
  Tous les seuils et choix de sélection sont déterminés sur TRAIN
  avant de lire DEV.
- 0 milliseconde de future lookahead supplémentaire : FFT/historique
  original uniquement, sans antidater des NoteOn.
- S614 et sa tête K, probas native, logique `CausalSelection`
  restent gelées. Production LIVE non modifiée.
- Ceci ajoute des **têtes de recherche** au-dessus de la chaîne
  S614/S641 ; ce n'est **pas** un réentraînement des nombreuses
  têtes historiques de la référence S80.

## Têtes individuelles

- **B** : S641 originale, 41D (note native, bass x2, posterior K,
  indices harmoniques de base).
- **T** : B + nouveauté acoustique sur ~23–70 ms de passé (temporel).
- **E** : B + compétition, f0 native et support harmonique
  partagé/exclusif (spectre causal).
- **J** : B + T + E en vecteur fusionné ; cette tête sert de
  comparaison indépendante.

Les seuils individuels OOF TRAIN retenus :
B **0,90** ; T **0,95** ; E **0,95** ; J **0,92**.

## Réelles sélections OOF — TRAIN uniquement

Utilité de recherche prédéfinie : **TP − 0,5 FP** sur événements
candidats, abstention autorisée.

| Sélection | TP nouveaux | FP candidats | Utilité |
|---|---:|---:|---:|
| B seule, témoin S641 | 17 | 20 | 7 |
| T seule | 10 | 13 | 3,5 |
| E seule | 12 | 13 | 5,5 |
| J seule | 12 | 19 | 2,5 |
| B ET E | **12** | **8** | **8** |
| B ET (T OU E) | 12 | 10 | 7 |
| B OU (T ET E) | 17 | 23 | 5,5 |
| Majorité 2/3 B T E | 12 | 13 | 5,5 |
| Unanimité B T E | 8 | 5 | 5,5 |
| B OU J | 20 | 30 | 5 |
| Fusion pondérée B60 T20 E20 | 10 | 8 | 6 |
| Fusion pondérée B40 T30 E30 | 17 | 21 | 6,5 |
| Fusion pondérée B40 T20 E20 J20 | 16 | 21 | 5,5 |

TRAIN a donc désigné **B ET E** (gain d'utilité +1 vs B).
Remarque importante : sur petit effectif positif, ce gain
n'est pas une preuve statistique solide.

## Résultats sur les vrais NoteOn GuitarSet DEV

| Sélection | TP | FP | FN | F1 (%) | TP K3+ |
|---|---:|---:|---:|---:|---:|
| S614 officiel | 998 | 585 | 582 | 63,10465 | 550 |
| Bass ×2 naïf | 1044 | 834 | 536 | 60,38172 | 586 |
| **B = S641** | **1009** | **594** | **571** | **63,39931** | **557** |
| T | 1004 | 587 | 576 | 63,32387 | 552 |
| E | 1003 | 586 | 577 | 63,30073 | 552 |
| J | 1005 | 590 | 575 | 63,30709 | 553 |
| B ET T | 1004 | 586 | 576 | 63,34385 | 553 |
| **B ET E — gagnant TRAIN** | **1002** | **586** | **578** | **63,25758** | **552** |
| B ET J | 1005 | 587 | 575 | 63,36696 | 554 |
| **B ET (T OU E)** | **1006** | **586** | **574** | **63,43001** | **554** |
| B ET T ET E | 1000 | 586 | 580 | 63,17119 | 551 |
| **B OU (T ET E)** | **1010** | **594** | **570** | **63,44221** | **557** |
| B OU J | 1009 | 597 | 571 | 63,33961 | 556 |
| T ET E | 1001 | 586 | 579 | 63,21440 | 551 |
| **Majorité 2/3 B T E** | **1007** | **586** | **573** | **63,47305** | **554** |
| Unanimité 3/3 B T E | 1000 | 586 | 580 | 63,17119 | 551 |
| Pondération 60/20/20 | 1003 | 587 | 577 | 63,28076 | 553 |
| Pondération 40/30/30 | 1008 | 596 | 572 | 63,31658 | 556 |
| Pondération 40/20/20/20 | 1007 | 595 | 573 | 63,29353 | 555 |
| Produit géométrique BTE | 1007 | 596 | 573 | 63,27364 | 555 |

**Trois politiques sont supérieures à S641 sur le DEV précédemment
examiné :**
- **Majorité 2/3** : +0,073745 point F1 vs S641 ; 2 TP
  de moins et 8 FP de moins.
- **B OU (T ET E)** : +0,042902 point F1 ; 1 TP de
  plus, aucun FP de plus que S641.
- **B ET (T OU E)** : +0,030704 point F1 ; 3 TP
  de moins et 8 FP de moins.

**Mais B ET E gagnant TRAIN régresse** :
- −7 TP et −8 FP vs S641, F1 −0,141733 point.
- Le fait que la majorité gagne uniquement en examinant DEV
  **interdit sa promotion sur ce résultat**.

## Audit par piste : où agit la majorité ?

Différences majoritaire (2/3) vs S641 :
- `BN3-154-E_comp` : +0 TP, **−4 FP** ;
- `Funk3-98-A_comp` : +0 TP, **−1 FP** ;
- `Rock1-90-C#_comp` : +0 TP, **−1 FP** ;
- `Funk3-112-C#_comp` : **−2 TP et −2 FP** ;
- `Rock1-130-A_comp` : **−1 TP et 0 FP** ;
- `Rock1-130-A_solo` : **+1 TP et 0 FP**.

Donc le petit gain DEV se concentre sur des corrections de quelques
pistes. La majorité protège des faux doublons/harmoniques mais peut
empêcher de vraies nouvelles fondamentales en accords complexes,
notamment K3+ (557 → 554).

## Conséquence pour l'architecture têtes/sélections

Il existe une **complémentarité mesurée**, pas une supériorité
généralisée de la fusion :
- strict AND améliore la précision mais perd du rappel ;
- OR récupère du rappel mais expose au surcomptage ;
- vote majoritaire équilibre les deux sur DEV, pas sur TRAIN ;
- interpolation simple des probabilités peut être moins robuste
  qu'une sélection logique à cas spécifiques.

Il faut maintenant apprendre un **routeur conditionnel** :
- caractéristiques de morphologie de l'attaque (nouveauté),
  notes actives, registre, et tête K native ;
- classes d'erreurs **par tête et par sélection**, avec audit de
  corrections/régressions, pour réserver le consensus aux erreurs
  harmoniques et le rattrapage aux notes graves masquées ;
- entrainement/choix sur TRAIN avec groupes compositions et test
  indépendant, sans lire DEV pour ajuster les règles.

Une vraie reconstrution H0/H1 (résidu après ajustement conjoint
des partiels) n'a **pas encore** été testée en S650, uniquement
les features spectrales S649. Il ne faut pas confondre le réseau
de votes expérimentaux avec une intégration définitive aux têtes
Exact-K historiques.

**Aucune promotion S650 ; le meilleur candidat préalablement
désigné reste S641 à 63,39931% DEV exploratoire.**
