# S652 — Résultats réels du Tone simulé et combinaisons de vues NOTE

## Verdict

**Le filtrage peut aider, mais aucune promotion.**

GitHub Actions SUCCESS: https://github.com/Andriamarosoa/note/actions/runs/39005694346
Source: `scripts/s652_tone_filter_multiview.py`
Tests: `test/test_s652_tone_filter_multiview.py`
Artefact: `S652_GUITARSET_TONE_FILTER_MULTIVIEW_REAL/report.json`.

Le score exploratoire le plus élevé sur GuitarSet DEV (**63,5978% F1**)
vient de la **moyenne des trois vues** ; elle dépasse le S641
expérimental (**63,3993%**) de ~0,199 point. **MAIS elle régresse
sur TRAIN calibration** et perd des notes superposées. Cette
performance DEV ne justifie pas de sélectionner ni de promouvoir
le modèle. Le protocole fixé sur TRAIN retient `S614_original`.

La campagne comporte **36 pistes GuitarSet TRAIN**, réparties en
9 compositions ; les 5 premières compositions triées par nom
calibrent la politique, les 4 dernières sont contrôle indépendant
de politique ; puis 12 GuitarSet DEV historiquement exposées.
TEST GuitarSet/IDMT/URMP demeure fermé.

## Tone vs pitch shifting

Nous n'avons PAS mesuré le potentiomètre d'une Fender Stratocaster.
Aucune transposition +2, aucun delay, aucun pitch-shift ni
échantillon futur. Les traitements numériques sont des
**proxies de réponse fréquentielle causalement appliqués** :
- original PCM inchangé ;
- passe-bas `scipy.signal.sosfilt`, Butterworth 1 ordre fc 2800Hz
  (simule grossièrement une légère atténuation aigus) ;
- passe-bas `sosfilt`, fc 1100Hz (atténuation plus forte).

Les 3 flux passent chacun dans le même frontend NOTE
`CausalPitchFeatures` et les mêmes têtes `NativeLiveHeads`
gelées. **Toutes les sélections utilisent K original non filtré** :
l'inférence Exact-K officielle et son état causal ne sont pas
modifiés. Les K des signaux filtrés sont calculés seulement pour
audit. Les moyennes/votes ne sont pas rétrodatés.

## Mesure NoteOn GuitarSet DEV ±80 ms

| Politique | TP | FP | FN | F1 % | TP avec overlap | TP K3+ | E2 TP /9 | E3 TP /77 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S614 natif | 998 | 585 | 582 | 63.1046 | 697 | 550 | 0 | 44 |
| Filtre doux seul | 993 | 598 | 587 | 62.6301 | 689 | 545 | 1 | 44 |
| Filtre fort seul | 921 | 538 | 659 | 60.6120 | 618 | 483 | 0 | 42 |
| Moyenne natif + doux | 997 | 572 | 583 | 63.3217 | 696 | 547 | 1 | 43 |
| **Moyenne des 3** | **974** | **509** | **606** | **63.5978** | **671** | **530** | **1** | **44** |
| Vote 2/3 | 991 | 603 | 589 | 62.4449 | 687 | 544 | 0 | 44 |
| Veto original & doux | 962 | 518 | 618 | 62.8758 | 665 | 525 | 0 | 42 |
| Veto original & fort | 879 | 411 | 701 | 61.2544 | 588 | 459 | 0 | 39 |
| Original OU doux (0.70) | 1010 | 609 | 570 | 63.1447 | 705 | 555 | 0 | 44 |
| Original OU doux (0.85) | 1001 | 587 | 579 | 63.1944 | 698 | 550 | 0 | 44 |
| Original OU fort (0.70) | 1019 | 654 | 561 | 62.6499 | 712 | 562 | 0 | 45 |
| Original OU fort (0.85) | 1006 | 595 | 574 | 63.2506 | 700 | 552 | 0 | 44 |
| Original OU filtres confirmés | 1001 | 586 | 579 | 63.2144 | 698 | 550 | 0 | 44 |

S641 est une autre chaîne expérimentale, distincte des 13
politiques ici, avec F1 historique reproduit précédemment de
**63,3993%** (1009 TP /594 FP /571 FN).

### Les vrais gains/régressions par enregistrement (moyenne 3)

Vs S614 :
- `Funk3-112-C#_comp` : **−23 FP et −3 TP** ;
- `Rock1-130-A_comp` : **−19 FP et −5 TP** ;
- `Rock1-90-C#_comp` : **−15 FP et +3 TP** ;
- `Funk3-98-A_comp` : **−12 FP et −13 TP** ;
- `BN3-119-G_comp` : **0 FP éliminé et −6 TP**.

**Le filtrage ne doit PAS s'activer uniformément.**
Selon les accords, la réponse fréquentielle élimine des fausses
harmoniques ou masque de vraies fondamentales à l'octave.

### Cas graves E2 et E3

Sur 12 pistes DEV il y a **9 annotations E2** et **77 E3**.
S614 : **0/9 E2**, **44/77 E3**.
Filtre doux : **1/9 E2**, **44/77 E3**.
Moyenne native+doux : **1/9 E2**, **43/77 E3**.
Moyenne trois vues : **1/9 E2**, **44/77 E3**.
Filtre fort : **0/9 E2**, **42/77 E3**.

Ainsi, un filtrage doux retrouve au moins **une vraie E2** que
l'original a manquée dans ce corpus, mais cela ne prouve pas
que le comportement soit reproductible sur une Stratocaster.
Ces comptes ne représentent pas une comparaison équilibrée
« E2 seul » / « E2+E3 simultanés » : les annotations seules
ne garantissent pas des prises isolées, et seulement 9 E2
apparaissent en DEV.

## K natif vs K diagnostic des sons filtrés

Mesure auxiliaire sur 1 576 attaques annotées
(lecture 30ms après l'attaque, cible nombre de hauteurs distinctes
actives, K plafonné à 6). **Ce n'est pas le score historique S80 Exact-K**.
Toutes les sorties de sélection emploient **K original**.

| Tête K de diagnostic | K correct total /1576 | K1 correct /441 | K2 correct /234 | K3 correct /458 | K4 correct /385 |
|---|---:|---:|---:|---:|---:|
| Signal original | 668 | **310** | 49 | 183 | 124 |
| Tone doux 2800Hz | 674 | 282 | 64 | 197 | **129** |
| Tone fort 1100Hz | 671 | 257 | **79** | **210** | 122 |

La réponse filtrée peut **favoriser les détections K2/K3**
lorsque les notes sont superposées, mais au coût de **confondre
davantage de vrais K1**. C'est une direction intéressante pour
une tête de sélection **conditionnelle par K** plutôt qu'un
filtrage général. Aucun modèle K n'est promu.

## Pourquoi le résultat DEV 63,5978 ne doit pas être promu

Calibrage TRAIN : 5 compositions entières (frozen split) :
- S614 original **77,8057% F1** ;
- Tone doux **76,8195%** ;
- moyenne trois vues **76,8000%** ;
- confirmation par les filtres **77,7987%**.
Aucune famille ne dépasse le baseline natif d'au moins
**+0,4 point** ; le sélecteur déterminé AVANT DEV reste S614.

Contrôle interne 4 autres compositions TRAIN :
- original **70,5360%** ;
- Tone doux **72,0588%** ;
- moyenne trois vues **71,7915%** ;
- veto original+doux **72,1897%**.
Cette variabilité explique la piste du **routage
conditionnel suivant le registre, les accords et le timbre**,
mais il serait trompeur de choisir l'optimum d'un autre split
à postériori.

## Pistes suivantes

1. Une **tête Tone-impact** doit prédire à quel moment une
   atténuation des aigus aide sans effacer une véritable
   fondamentale : entraînement avec validation imbriquée par
   composition et coût des vrais NoteOn perdus.
2. Mettre la preuve multi-vues au niveau des **sélections par
   K1/K2/K3+**, pas modifier arbitrairement l'Exact-K original.
3. Préférer un **consensus acoustique dynamique** (énergie des
   partiels propres à une vraie note) à la moyenne uniforme,
   car elle perd 26 TP superposés et 20 TP K3+.
4. Mesurer le vrai potentiomètre de Stratocaster dans une prise
   A/B simultanée dry/filtered, sans confondre jeu du guitariste,
   gain, effets, attack timing et noise floor.
5. Vérifier sur instruments et compositions réellement inconnus
   puis CPU/microphone strict LIVE avant promotion.

**Statut : S652 terminé, succès CI, aucun modèle officiel ni LIVE changé.**
