# Audit des informations utiles au comptage polyphonique

Audit local terminé le 7 octobre 2026. Comptage K directement supervisé ; aucune identification de note produite.
59 309 groupes, dont 7 385 polyphoniques, 190 enregistrements et 19 compositions. Folds 0, 1, 2, 4 ; fold 3 et player 05 exclus.
Les paramètres et les quatre bras ont été fixés avant lecture des scores. Les compositions du fold évalué sont exclues de l’apprentissage de chaque sonde.

## Résultats des sondes

| Modèle | Exacts poly | Exact poly | Exact global |
|---|---:|---:|---:|
| Référence freeze_local_combo | 2530/7385 | 34.2586% | 81.6976% |
| Sonde : candidats seuls | 2213/7385 | 29.9661% | 79.9069% |
| Sonde : candidats + spectre normalisé | 2425/7385 | 32.8368% | 81.3536% |
| Sonde : candidats + spectre brut | 2593/7385 | 35.1117% | 81.6874% |
| Sonde : brut + FFT 1024/2048 | 2675/7385 | 36.2221% | 82.0921% |

Une sonde est ici un classifieur de comptage HistGradientBoosting, avec 120 itérations et 15 feuilles maximum par arbre, identique dans les quatre bras. Elle ne remplace pas la référence.

## Informations établies

1. Les indices spectraux apportent +212 comptes poly exacts par rapport aux seuls résumés des candidats (+2,871 points).
2. Conserver les valeurs brutes des trois canaux apporte +168 comptes poly exacts (+2,275 points) et +198 comptes globaux. Le gain poly est positif sur chacun des quatre folds. Intervalle descriptif à 95% par bootstrap des 19 compositions : +1,316 à +3,269 points.
3. Ajouter des fenêtres FFT réelles de 1024 et 2048 échantillons, toujours dans les mêmes 4096 échantillons disponibles, apporte +82 comptes poly exacts (+1,110 point). Intervalle descriptif : -0,125 à +2,281 points ; le signal est plus faible.

LayerNormalization sur les trois canaux centre leur moyenne locale et normalise leur dispersion. Une branche conservant les niveaux et leur dispersion, en complément de la branche normalisée, est donc une piste concrète. Le présent test ne sépare pas parfaitement préservation de l’information et facilité d’apprentissage de la représentation.

## Comparaison complète avec la référence

| K vrai | Effectif | freeze_local_combo | Sonde brut + multirésolution | Solde exacts |
|---:|---:|---:|---:|---:|
| 0 | 39652 | 95.922% | 94.921% | -397 |
| 1 | 12272 | 64.285% | 68.245% | +486 |
| 2 | 3445 | 34.340% | 35.559% | +42 |
| 3 | 2289 | 39.974% | 40.192% | +5 |
| 4 | 1207 | 34.548% | 38.028% | +42 |
| 5 | 355 | 4.225% | 20.000% | +56 |
| 6 | 89 | 0.000% | 0.000% | +0 |

Bilan poly : 1188 corrections, 1043 régressions, +145 net. Sous-comptages poly : 3630 → 3303 ; surcomptages poly : 1225 → 1407.
Le gain poly de +1,963 point reste modeste. Il est positif sur trois folds ; le fold 4 recule de 0,594 point. K0 perd 397 réponses correctes et K6 reste à 0%. Ce résultat ne satisfait pas encore l’objectif d’une forte amélioration.

## L’essai de normalisation antérieur est pris en compte

Le run 36490993251 avait déjà comparé, sur le composant natif et une validation interne, la normalisation et une division fixe par 12 : +47 comptes poly exacts mais -95 K1 et -11 globaux. Il avait été rejeté. Cet audit confirme un signal dans les valeurs brutes avec un autre apprenant et quatre folds ; il ne transforme pas cet ancien échec en succès.
La modification à évaluer doit préserver la branche normalisée et ajouter les informations manquantes, avec validation appariée. Aucun score de ce futur réseau n’est annoncé.

## Comparaison descriptive avec YourMT3+

YourMT3+ corrige 2430 erreurs poly de la référence mais en perd 932 que celle-ci réussissait. Le choix parfait entre les deux atteindrait 67,163% ; ce chiffre est un oracle utilisant la vérité, pas un score obtenu par un système. Les prédictions YourMT3+ ne servent jamais à entraîner ces sondes.

## Reproduction

Environnement mesuré : Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0 ; deux threads par apprentissage.
Sources : release `v273-window-pair-36351028493` (`native-window-bundle.zip`), configuration du commit `aebfe9f8cb6dc2c0dc391f773fd331756e4c5a45`, audio GuitarSet mono pickup mix vérifié par MD5, et artifact `yourmt3-exactk-summary` du run 37605163312.
Les 190 reconstructions de contrôle du spectre d’origine concordent avec le cache float16 à la précision attendue (erreur absolue maximale 0,003942). Les nouvelles FFT n’accèdent jamais à des échantillons après début du groupe +2788.
Scripts : `scripts/audit_exactk_count_information.py` (étapes cache, audio, train) et `scripts/summarize_exactk_count_information.py`. Empreintes, matrices de confusion, métriques par fold et protocole sont dans le rapport JSON. Les décisions par ligne sont conservées dans le NPZ.

```sh
python scripts/audit_exactk_count_information.py cache --summary SUMMARY.zip --bundle BUNDLE --config CONFIG.json --output RESULTS
python scripts/audit_exactk_count_information.py audio --summary SUMMARY.zip --bundle BUNDLE --audio audio_mono-pickup_mix.zip --output RESULTS
python scripts/audit_exactk_count_information.py train --arm candidate_only --output RESULTS
python scripts/audit_exactk_count_information.py train --arm native_channel_norm --output RESULTS
python scripts/audit_exactk_count_information.py train --arm native_raw --output RESULTS
python scripts/audit_exactk_count_information.py train --arm native_raw_multires --output RESULTS
python scripts/summarize_exactk_count_information.py --input RESULTS --output analysis
```

Aucune modification de la référence, aucune évaluation du fold 3 ou du player 05, aucune promotion automatique.
