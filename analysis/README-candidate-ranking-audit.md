# Exact-K : classement des candidats et contrôle du pool élargi

Audit du 5 octobre 2026 sur `codex/v273-failure-clustering`, à partir du commit
`aad8dfb`. **La limite de huit candidats perd beaucoup de fréquences attendues.
Corriger cette couverture ne suffit pas à rendre le correcteur 3→2 utile.**
Aucun modèle n'est promu ; la base reste conservée.

Le [protocole](v273-candidate-ranking-protocol.md) a été écrit en deux étapes :
classement et NMS avant leur mesure complète, puis contrôle factoriel après
constat de la faible capacité. Les [preuves](evidence/v273-candidate-ranking/)
contiennent les traces, spectres intermédiaires, scores, modèles et rejeu.

## Population et contrôles

- Même cohorte de 488 cas : 125 K3 dégradés, 147 K3 préservés, 108 K2 corrigés,
  108 K2 manqués par le correcteur historique. Ces groupes restent figés ; ils
  ne sont pas redéfinis avec les nouvelles prédictions.
- Folds internes **0, 1, 2 et 4 uniquement**. Aucun chargement du fold 3.
- Comparaison des correcteurs sur 1 666 lignes audio uniques de FIT/VAL,
  122 enregistrements, et 845 lignes VAL d'action valides, autres vrais K inclus.
- Spectre positif post-moins-pré, fenêtre Hann 2 048 échantillons à 44,1 kHz,
  FFT 8 192, grille de 240 F0, NMS 55 cents, base et routage B_low figés.
- Appariement univoque à 55 cents avec les annotations, réservé au diagnostic.
  Les fréquences annotées ne sont jamais données au correcteur.

Il s'agit du **chemin résiduel sur audio normal**. Le projet utilise normal et
compressé ; ces observations ne démontrent ni une cause ni un gain dans le
chemin compressé. Les folds internes ont déjà été examinés : pas de nouvelle
validation indépendante intacte.

## Où les fréquences disparaissent-elles ?

Le classement complet après NMS est poursuivi sans s'arrêter à huit. Pour les
816 notes des 272 vrais K3, avec saillance historique `1/sqrt(h)` :

| Situation de la note | Nombre |
|---|---:|
| Un candidat correspondant figure dans les huit premiers | 312 |
| Premier candidat correspondant après la huitième place | 499 |
| Tous les voisins à 55 cents supprimés par NMS | 5 |
| Hors de la grille | 0 |

Il n'y a pas de conflit supplémentaire d'appariement univoque dans cette
population. Ces nombres expliquent une perte de couverture du pool ; ils ne
constituent pas une attribution causale du surcomptage du réseau.

| Pente de saillance | Pool 8 | Pool 16 | Pool 32 | Pool 64 | Toute la NMS |
|---|---:|---:|---:|---:|---:|
| 0,5, historique | 3 | 15 | 172 | 267 | 267 |
| 1 | 10 | 24 | 180 | 262 | 262 |
| 2 | 8 | 54 | 192 | 266 | 266 |

Chaque cellule compte les cas où **les trois fréquences K3** sont présentes,
sur 272. Avant toute NMS, la grille complète couvre 272/272. La NMS élimine
entièrement une fréquence dans 5, 10 ou 6 cas selon la pente. Le pool de huit
n'interroge donc qu'une petite partie du classement nécessaire aux notes faibles.

Pour les 216 vrais K2, la couverture historique passe de **34/216 à 213/216**
avec 64 candidats. Les traces détaillées donnent, pour chaque note, le rang
brut, le rang après NMS, les ex æquo et l'identité des candidats bloquants.

Le pool historique utilise en moyenne 4,43 bins dominants de fondamentale pour
huit candidats sur les K3. Cela signale une redondance du score ; ce nombre de
bins n'est pas un nombre de sources musicales.

## Les bonnes candidates sont-elles ensuite choisies ?

Contrôle factoriel : pool 8 ou 64 × pente du gabarit gaussien 0,5 ou 2.
La pente de saillance reste historique, ainsi que tous les autres paramètres.

| Pool | Pente du gabarit | K3 : pool complet /272 | K3 : triplet complet /272 | K2 : meilleur couple complet /216 |
|---|---:|---:|---:|---:|
| 8 | 0,5 | 3 | 0 | 19 |
| 8 | 2 | 3 | 1 | 19 |
| 64 | 0,5 | 267 | 0 | 23 |
| 64 | 2 | 267 | 31 | 47 |

Le défaut de couverture est donc réel, mais **la présence des bonnes fréquences
ne garantit pas leur sélection**. Avec le gabarit historique, aucun des 272
triplets n'est complet, même dans le pool élargi. Le gabarit modifié en retrouve
31, ce qui reste très loin de la couverture de 267. Le critère de reconstruction
et l'adéquation des gabarits aux spectres réels restent à examiner ; ce tableau
ne permet pas de les réduire à une cause unique.

## Exact-K sur les mêmes cas

Même petite LR à deux résidus, pondération équilibrée, C=1 et seuil 0,5. Pour
chaque fold externe, les variantes sont classées uniquement sur les trois folds
restant dans FIT, avec abstention si aucun net FIT n'est strictement positif.
Les résultats VAL des bras fixes sont descriptifs et ne servent pas à choisir.

| Pool / pente | Fold 0 | Fold 1 | Fold 2 | Fold 4 | Net total |
|---|---:|---:|---:|---:|---:|
| 8 / 0,5, contrôle historique stable | −4 | +7 | −19 | +1 | **−15** |
| 8 / 2, contrôle déjà mesuré | −1 | −4 | −22 | +10 | **−17** |
| 64 / 0,5 | −11 | +4 | −23 | −1 | **−31** |
| 64 / 2 | +4 | −2 | −9 | +9 | **+2** |

Le bras 64 / 2 produit 137 corrections, 135 régressions et 192 actions sur
d'autres vrais K. Son +2 descriptif n'est pas une correction validée : deux
folds régressent et il ne satisfait pas la sélection prévue sur FIT.

**Tous les bras sont rejetés par FIT dans les quatre folds.** Les meilleurs nets
FIT sont respectivement −66, −40, −53 et −68. La politique sélectionnée
s'abstient partout : zéro action, zéro correction, zéro régression, net zéro.
Cette conservation de la base n'est pas un gain d'Exact-K.

## Vérifications et décision

- Les deux bras pool-8 reproduisent exactement les résidus et comptes de l'audit
  précédent : écart maximal des résidus **0**.
- Les 24 tests passent : référence NNLS, dictionnaires, NMS, rangs, classement
  stable, monotonie du résidu avec pool élargi et séparation FIT/VAL.
- Rejeu des saillances à partir des 488 spectres sauvegardés, des traces NMS et
  des appariements. Le rejeu ne redécode pas l'audio.
- Rejeu des 16 modèles figés sur les 845 lignes VAL : écart maximal de
  probabilité **0**, décisions et sélection identiques, sans réentraînement.
- Empreintes des entrées, modèles, scores et archive consignées dans les preuves.

**Décision : garder la base, ne pas activer le pool élargi comme correcteur.**
La capacité et la pondération harmonique contribuent aux erreurs de cette voie,
mais leur modification conjointe ne donne pas un correcteur sélectionnable.

La prochaine question utile est le coût de reconstruction : lorsque le pool
contient les notes attendues, quelles composantes choisies les remplacent
(harmoniques, sous-harmoniques, fréquences proches ou notes étrangères), et
quelle modification contrôlée du critère peut les distinguer sans annotations ?
Ne pas relancer les grilles de pentes ou de capacités sans une preuve nouvelle.

## Reproduction

Les spectres internes sont désormais conservés dans l'archive de cette étape.
Les annotations de diagnostic sont dans l'archive acoustique précédente. Après
extraction des archives dans un dossier temporaire :

```bash
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B \
  scripts/replay_v273_candidate_audit.py \
  --ranking /chemin/ranking --capacity /chemin/capacity \
  --cases /chemin/acoustic/cases.jsonl --output /chemin/replay.json
```

Pour recalculer le contrôle de capacité depuis l'audio vérifié et les exports
figés du run `37356100423` :

```bash
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B \
  scripts/audit_v273_candidate_capacity_guard.py \
  --exports /chemin/residual_v2_results --dataset /chemin/GuitarSet \
  --decay /chemin/decay --output /chemin/nouveau-resultat
```

Dépendances utilisées : Python 3.12.14, NumPy 1.26.4, SciPy 1.17.1,
scikit-learn 1.4.2. Les sorties existantes ne sont pas écrasées.
