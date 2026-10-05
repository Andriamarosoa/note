# Exact-K : audit du critère de reconstruction

Audit du 6 octobre 2026 sur `codex/v273-failure-clustering`, après le checkpoint
`83dda55`. **Le solveur numérique retrouve une combinaison connue, mais les
combinaisons correspondant aux notes annotées décrivent moins bien les spectres
réels que d'autres fréquences. Deux corrections physiques simples ont été testées
sans annotations et rejetées. La base reste conservée.**

Le [protocole](v273-reconstruction-criterion-protocol.md) fixe successivement le
diagnostic, la pondération log-fréquence puis le gabarit Hann avant leurs
évaluations. Les [preuves](evidence/v273-reconstruction-criterion/) contiennent
les 488 traces diagnostiques, les 1 666 spectres d'action, les features, modèles,
scores, fréquences choisies et rejeux.

## Périmètre conservé

- Folds internes **0, 1, 2 et 4 uniquement** ; aucun chargement du fold 3.
- 488 cas acoustiques figés : 272 vrais K3 et 216 vrais K2.
- Tests Exact-K sur les mêmes 1 666 lignes audio uniques, 122 enregistrements et
  845 lignes VAL d'action valides, avec tous les autres vrais K comptabilisés.
- Base neuronale, routage B_low, grille F0, saillance, pool-64, LR, seuil 0,5 et
  populations inchangés.
- Les annotations servent uniquement au diagnostic des fréquences. Aucune
  fréquence annotée n'entre dans une feature ou une décision d'inférence.

Il s'agit du chemin résiduel sur **audio normal**. Le projet utilise aussi le
chemin compressé ; ces résultats ne prouvent aucune cause ni aucun gain pour ce
second chemin. Ces folds internes ont déjà été inspectés et ne constituent pas
un test final intact.

## Comparaison libre contre combinaison attendue

Pour chaque cas dont le pool-64 couvre toutes les notes, l'audit compare :

1. la combinaison choisie librement par le NNLS ;
2. la meilleure combinaison du même pool qui apparie toutes les notes attendues
   à 55 cents, utilisée seulement comme oracle de diagnostic.

Sur les 272 K3, le pool est complet dans 267 cas.

| Gabarit | 0 note retrouvée | 1 | 2 | 3 | Écart oracle − libre médian |
|---|---:|---:|---:|---:|---:|
| Gaussien, `1/sqrt(h)` | 37 | 167 | 63 | 0 | 0,08512 |
| Gaussien, `1/h²` | 33 | 98 | 105 | 31 | 0,03469 |

L'écart est exprimé en résidu quadratique normalisé. La combinaison attendue a
un résidu plus élevé puisque le choix libre est le minimum global. Avec `1/h²`,
l'écart médian est **0,06143** dans les 124 K3 du groupe historiquement dégradé,
contre **0,01748** dans les 143 K3 historiquement préservés. Cela mesure une
association dans les groupes figés ; ce n'est pas une cause unique du changement
d'Exact-K.

Le contrôle synthétique construit exactement un mélange des colonnes attendues.
Le solveur retrouve les notes dans **267/267 cas**, pour les deux gabarits, avec
un résidu maximal de `7,11e-15`. Les résidus et fréquences du contrôle réel
reproduisent les sorties archivées avec un écart maximal de **0**. Il n'y a donc
pas ici de défaut numérique du solveur ou de rejeu.

## Quelles fréquences remplacent les notes ?

Les catégories ci-dessous sont **non exclusives**. Une composante peut être à la
fois proche d'un harmonique attendu et d'un harmonique étranger. La proximité
n'identifie pas la source physique du son.

| Relation d'une composante K3 non appariée | `1/sqrt(h)` (508 composantes) | `1/h²` (400 composantes) |
|---|---:|---:|
| Voisine d'une note attendue, 55–150 cents | 223 | 132 |
| Harmonique entier d'une note attendue | 27 | 157 |
| Sous-harmonique entier d'une note attendue | 166 | 23 |
| Fondamentale étrangère active | 34 | 25 |
| Harmonique d'une note étrangère active | 21 | 89 |
| Sous-harmonique d'une note étrangère active | 79 | 14 |
| Aucune relation mesurée ci-dessus | 92 | 83 |

La pente historique favorise surtout des voisins et sous-harmoniques. La pente
`1/h²` déplace une partie du problème vers les harmoniques. Les relations
étrangères et les cas non classés montrent qu'aucune règle unique « supprimer
les harmoniques » ne couvre la population.

## Correction 1 : erreur uniforme en log-fréquence

Le coût historique additionne les erreurs sur des bins uniformes en Hz. Le
contrôle préenregistré utilise `w(f)=65/f`, équivalent à une mesure uniforme en
`log(f)`, avec pool-64 et gabarit `1/h²` inchangés. Aucun exposant n'a été ajusté.

| Critère | Fold 0 | Fold 1 | Fold 2 | Fold 4 | Corrections | Régressions | Autres K | Net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linéaire Hz, contrôle | +4 | −2 | −9 | +9 | 137 | 135 | 192 | **+2** |
| Log-fréquence | −2 | −5 | −9 | +11 | 146 | 151 | 213 | **−5** |

La pondération réduit les composantes K3 proches d'un harmonique attendu de 160
à 145 sur l'ensemble des 272 cas, mais augmente les voisins à 55–150 cents de
137 à 161. Elle améliore le nombre de notes retrouvées dans 19 K3 et le dégrade
dans 26 ; les triplets complets passent de 31 à 30. Pour K2, les couples complets
passent de 47 à 53, sans transfert en gain Exact-K global.

Ses nets FIT tournants sont −67, −85, −82 et −79. Le bras est rejeté dans les
quatre folds ; la politique sélectionnée s'abstient partout.

## Correction 2 : réponse exacte de la fenêtre Hann

Le second contrôle garde le pool-64 et `1/h²`, puis remplace la gaussienne de
18 Hz par la réponse de puissance phase-moyennée exacte de la fenêtre Hann de
2 048 échantillons. Aucun paramètre de forme n'est ajusté.

| Gabarit | Fold 0 | Fold 1 | Fold 2 | Fold 4 | Corrections | Régressions | Autres K | Net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gaussien, contrôle | +4 | −2 | −9 | +9 | 137 | 135 | 192 | **+2** |
| Hann exact | −9 | −4 | −10 | +7 | 125 | 141 | 186 | **−16** |

Le Hann améliore le nombre de notes retrouvées dans 29 K3 et le dégrade dans
25, mais les triplets complets diminuent de 31 à 29. Les couples K2 complets
passent de 47 à 50. Ses nets FIT sont −90, −54, −69 et −91 : rejet dans les
quatre folds et abstention de la politique sélectionnée.

## Vérifications et décision

- **29 tests** couvrent les solveurs, les contrôles, les rangs/NMS, la séparation
  FIT/VAL, la mesure log-fréquence et la réponse Hann.
- Le solveur accéléré Hann reproduit le solveur exhaustif de référence sur les
  problèmes contrôlés ; il ne change que le temps de calcul.
- Rejeu des 16 modèles log/Hann sur les quatre folds : écart maximal de probabilité **0**,
  décisions et abstentions identiques, sans audio et sans réajustement.
- Le rejeu intégral recalcule les 488 diagnostics depuis les spectres archivés,
  réagrège les deux audits de fréquences et vérifie les 1 666 spectres.
- Le contrôle GitHub précédent [`37367957693`](https://github.com/Andriamarosoa/note/actions/runs/37367957693)
  est terminé avec succès. Le run `37366044960` avait été annulé avant toute
  étape ; il ne constitue pas un échec scientifique.

**Décision : rejeter les deux corrections et conserver la base.** Le +2 du bras
gaussien reste descriptif et non sélectionnable ; aucune variante n'est promue.

Les preuves établissent maintenant une limite précise du modèle statique : même
avec les bonnes fréquences disponibles et un solveur exact, le coût préfère
souvent d'autres composantes sur le spectre positif post-moins-pré. Les deux
modifications simples du poids fréquentiel et de la forme des pics ne résolvent
pas ce problème. La prochaine étape doit tester une information indépendante du
même coût statique, par exemple la stabilité temporelle des composantes sur deux
fenêtres postérieures, avec protocole FIT identique et sans annotations en
inférence. Cette proposition reste une hypothèse à contrôler.

## Reproduction

Les archives précédentes fournissent le pool et les annotations diagnostiques ;
la nouvelle archive fournit les spectres des 1 666 lignes et toutes les sorties.
Le rejeu complet n'utilise ni audio ni entraînement :

```bash
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B \
  scripts/replay_v273_reconstruction_followups.py \
  --candidate /chemin/candidate --cases /chemin/acoustic/cases.jsonl \
  --criterion /chemin/criterion --log-guard /chemin/log \
  --log-coverage /chemin/log-coverage --hann-guard /chemin/hann \
  --hann-coverage /chemin/hann-coverage --output /chemin/replay.json
```

Dépendances : Python 3.12.14, NumPy 1.26.4, SciPy 1.17.1 et
scikit-learn 1.4.2.
