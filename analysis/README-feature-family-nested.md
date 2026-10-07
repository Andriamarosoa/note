# Familles de features : sélection sur FIT, bilan sur VAL

**Le gain exploratoire +20 est reproduit, mais aucune amélioration n'est obtenue
après sélection exclusivement sur FIT. La référence est conservée.** Audit du
7 octobre 2026 sur les seuls folds internes 0, 1, 2, 4.

## Défaut confirmé et correction

Le [run interne 37547761255](https://github.com/Andriamarosoa/note/actions/runs/37547761255)
entraînait ses modèles sur FIT, puis cherchait famille et seuil sur les VAL réunis.
Son résultat `base_geom_attack @ 0,59` — 53 corrections, 33 régressions,
net +20 — était donc un **résultat de sélection**, pas un bilan après sélection
sur FIT. Cela ne rend pas faux les comptes publiés ; cela limite leur portée.

Le nouveau script effectue trois rotations internes dans le FIT de chaque fold
VAL. Il choisit la famille et, pour le contrôle secondaire, le seuil sur ces
prédictions internes exclusivement. Une absence de gain FIT provoque l'abstention.
Les annotations de fréquence n'interviennent jamais dans les features.

Le [protocole](v273-feature-family-nested-protocol.md) et le code ont été publiés
dans [`e064c0c`](https://github.com/Andriamarosoa/note/commit/e064c0ce9e7f9791d4f44658ccf4db191679854a)
avant le calcul du nouveau bilan. Aucune information du fold 3 ni de player 05
n'a été chargée pour cette étude. Les expériences externes présentes sur la
branche n'interviennent pas dans la décision.

## Comparaison sur les mêmes cas

Même réseau de base, même B_low, mêmes masques et FIT/VAL du run source
`37356100423` : **1 666 lignes physiques, 122 pistes, 845 lignes VAL**.
Les résidus historiques sont conservés par apparition FIT/VAL, comme dans
le run comparé. Les huit features ajoutées reproduisent son extraction actuelle.
Il s'agit du chemin normal uniquement.

| Politique | Sélection | Actions | Corrections | Régressions | Autres K, restant faux | Net |
|---|---|---:|---:|---:|---:|---:|
| Enrichie à 0,59, contrôle descriptif reproduit | VAL réunis | 161 | 53 | 33 | 75 | +20 |
| Principale, seuil 0,50 inchangé | FIT seulement | 0 | 0 | 0 | 0 | **0** |
| Témoin à deux résidus, grille robuste | FIT seulement | 4 | 1 | 1 | 2 | **0** |
| Choix entre les quatre familles, grille robuste | FIT seulement | 5 | 1 | 1 | 3 | **0** |

« Net » signifie corrections moins régressions par rapport aux prédictions
du compteur de base sur ces mêmes lignes. Le contrôle robuste est secondaire :
il n'est pas choisi après coup à la place de la politique principale.

Les quatre familles à seuil fixe 0,50 reproduisent aussi exactement les comptes
du run précédent : deux résidus −17, géométrie −31, attaque +11,
géométrie + attaque +1. Ce sont des bilans descriptifs VAL ; ils ne remplacent
pas la sélection prévue sur FIT. Le −17 utilise les résidus historiques ; le
−15 de l'ancien test du tri stable désignait une autre intervention.

### Pourquoi la règle principale s'abstient

| Fold VAL tenu à l'écart | FIT interne : base | + géométrie | + attaque | + les deux |
|---|---:|---:|---:|---:|
| 0 | −75 | −68 | −63 | −71 |
| 1 | −39 | −49 | −44 | −55 |
| 2 | −53 | −63 | −55 | −50 |
| 4 | −67 | −57 | −68 | −72 |

Aucune famille n'a un net FIT positif à 0,50. La politique principale conserve
donc chaque prédiction de base. Pour le contrôle robuste :

| Fold VAL | Choix FIT entre familles | Résultat VAL |
|---|---|---|
| 0 | Abstention | Aucune action |
| 1 | Géométrie, seuil 0,75 | 1 action sur un autre vrai K, restant fausse |
| 2 | Base, seuil 0,63 | 1 correction, 1 régression, 2 autres K |
| 4 | Abstention | Aucune action |

## Contrôles et limites

- Probabilités historiques de référence reproduites à `3,33e-16` près ; comptes
  de chaque famille reproduits dans chaque fold, et contrôle +20 identique.
- **5 tests ciblés réussis** : séparation par fold/enregistrement, rejet du fold 3,
  choix sans labels VAL, abstention, seuil inclusif et bilan des autres K,
  comparaison du rejeu portable à `predict_proba`.
- Rejeu des **16 modèles finaux et 48 modèles internes**, sans audio ni
  réajustement : erreur de probabilité **zéro**, choix et comptes identiques.
- Le bundle global n'est pas chargé. Les horodatages viennent des exports
  vérifiés ; le fold est contrôlé avant chaque décodage audio.
- Les folds ont déjà été explorés. Cette séparation corrige l'étape de sélection
  des résidus, mais ne constitue pas une validation indépendante de toutes les
  décisions antérieures. Le réseau et le routage amont restent figés.
- Aucune conclusion sur le chemin compressé, aucun réseau réentraîné,
  aucune promotion du modèle.

Un contrôle descriptif ajouté **après gel des prédictions** montre que la part
de K2 parmi les K2/K3 diffère entre les populations FIT et VAL :

| Fold externe | Part K2 sur FIT | Part K2 sur VAL |
|---|---:|---:|
| 0 | 34,69 % (171/493) | 47,86 % (67/140) |
| 1 | 26,02 % (83/319) | 43,52 % (47/108) |
| 2 | 30,32 % (114/376) | 28,00 % (21/75) |
| 4 | 35,31 % (202/572) | 49,09 % (81/165) |

Ce constat n'est pas une cause démontrée du manque de transfert : le fold 2
présente le sens inverse. Aucun seuil ni modèle n'est modifié à partir de ce
tableau. La suite utile est de séparer ce changement de proportions d'un
changement du signal acoustique **à vrai K identique**, en réutilisant les
features archivées avant tout autre essai de correction.

## Preuves et rejeu

[Rapport](evidence/v273-feature-family-nested/report.json),
[contrôle de rejeu](evidence/v273-feature-family-nested/replay-check.json),
[empreintes](evidence/v273-feature-family-nested/checksums.json).
L'archive du même dossier contient les entrées réduites, features supplémentaires,
modèles, scores, rapport source et provenance. Elle ne contient ni audio ni fold 3.

Python 3.12.14, NumPy 1.26.4, SciPy 1.17.1, scikit-learn 1.4.2.
Après vérification des empreintes et extraction dans `replay/nested` :

```bash
export PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B -m unittest test.test_v273_feature_family_nested -v
python -B scripts/audit_v273_feature_family_nested.py replay \
  --output replay/nested --reference replay/nested/source-report.json
```

La CI dédiée vérifie les cinq tests, toutes les empreintes et ce rejeu exact.
Une nouvelle extraction n'est nécessaire que pour contrôler la provenance audio,
pas pour reproduire la sélection et son bilan.
