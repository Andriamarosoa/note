# Audit des résidus pair/triplet — version 2

Cette version corrige la méthode d'audit du correcteur K2/K3 après les runs
[37342597795](https://github.com/Andriamarosoa/note/actions/runs/37342597795)
et [37347597565](https://github.com/Andriamarosoa/note/actions/runs/37347597565).
Le correcteur initial a corrigé 108 erreurs et dégradé 125 prédictions exactes,
soit −17 Exact-K. Cette modification ne démontre aucun nouveau gain et ne
promeut aucun modèle.

## Correction principale

`global_stratified` découpe **les mêmes scores du modèle global** en registres
low/mid/high/unassigned. Le nombre de lignes, d'actions, de corrections et de
régressions de ces quatre catégories doit se sommer exactement au total global.
Les AUC ne s'additionnent pas.

`register_refit` contient, seulement avec `--include-register-refit`, les modèles
réajustés séparément par registre. Leurs résultats ne sont jamais additionnés
pour décrire le modèle global.

Le registre est estimé à partir de `median_triplet_f0`. Les limites sont les
tertiles des cas K2/K3 valides de FIT. Ce ne sont pas des annotations des notes
réelles ni des seuils grave/aigu fixes. Une valeur de registre manquante reste
dans `unassigned` et ne supprime pas une ligne dont les deux résidus sont valides.

## Protocole conservé

- Folds internes 0, 1, 2 et 4 ; aucune prédiction, sélection ou évaluation sur le fold 3.
- Réseau principal utilisé en inférence depuis les checkpoints existants.
- Deux entrées du correcteur : `best_pair_residual_ratio` et `best_triplet_residual_ratio`.
- StandardScaler + LR C=1, L2, lbfgs, class_weight=balanced, max_iter=3000.
- Ajustement sur les vrais K2/K3 de FIT dans B_low avec prédiction de base K3.
- Action 3→2 au seuil fixé de 0,5 sur **tous** les cas valides B_low/base-K3 de VAL,
  y compris les autres K. Aucun seuil choisi sur VAL.
- Les signes des scores univariés sont choisis sur FIT global, puis conservés
  dans les découpages par registre.
- Aucun ajustement du registre comme entrée de la LR, aucune promotion automatique.

Le correcteur et l'audit partagent maintenant la fonction d'ajustement : dtype
float64, matrice contiguë C et random_state=28431. Cette uniformisation évite
une divergence future des chemins d'ajustement. Elle ne prouve pas la cause de
l'écart historique d'AUC du fold 2, qui reste non expliqué faute d'exports anciens.

## Contenu des résultats par fold

| Fichier | Contenu |
|---|---|
| `report.json` / `report.md` | AUC, distributions, corrélations, comptes globaux et par registre |
| `rows.jsonl` | Un enregistrement par cas d'action FIT/VAL, y compris les exclusions |
| `replay.npz` | Identifiants, labels, enregistrements, folds, routage B_low, probabilités de base, features structurelles, résidus, scores et probabilités |
| `model.json` | Scaler, classes, coefficients, intercept, paramètres et nombre d'itérations de la LR globale |
| `manifest.json` | Empreintes SHA-256 des exports, matrices, sources, config et checkpoints, identité vérifiée du bundle, versions d'exécution |
| `replay_check.json` | Vérification du rejeu des scores figés et de la comptabilité |

Chaque ligne JSONL contient aussi l'échantillon de début, le temps en secondes,
le vrai K, la prédiction de base et la décision 3→2. Les décisions FIT sont des
diagnostics sur l'apprentissage, distincts des résultats de validation. La
probabilité exportée est celle de la LR pondérée ; elle n'est pas annoncée comme
une probabilité naturelle calibrée.

Les résidus indisponibles et les résidus non finis ont un motif d'exclusion
explicite. Leurs identifiants restent présents ; aucune valeur manquante n'est
remplacée par un résidu nul.

## Vérifier un export sans audio ni TensorFlow

Avec NumPy, SciPy et scikit-learn installés, depuis la racine du dépôt :

```bash
PYTHONPATH=.:src python -B scripts/replay_v273_residual_audit.py --input chemin/du/fold
```

Le rejeu ne réentraîne pas la LR. Il vérifie les empreintes, reconstruit les scores
à partir du scaler et des coefficients, compare les probabilités à 1e-12 près,
exige les mêmes décisions exactes et recalcule les comptes globaux et par registre.
Les identifiants et enregistrements FIT/VAL doivent être disjoints.

```bash
PYTHONPATH=.:src python -m unittest test.test_v273_residual_audit -v
```

Les tests utilisent des données synthétiques et ne mesurent pas les performances
du projet. Le workflow effectue ensuite le vrai rejeu depuis l'audio et les
checkpoints, puis le contrôle indépendant des exports pour chaque fold.

## Suite après le run

Comparer d'abord populations, empreintes, probabilités et résultats aux rapports
historiques. Les exports permettront ensuite les scatter plots, l'analyse par
enregistrement et les comparaisons de représentations sur exactement les mêmes
lignes. Ces variantes ne sont pas incluses dans ce run. La corrélation et les
inversions de moyenne restent des constats descriptifs, pas des causes démontrées.
