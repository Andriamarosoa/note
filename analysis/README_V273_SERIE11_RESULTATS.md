# V27.3 — Résultats de la série 11 : garde-fou risque K0–K6

Run de preuve : https://github.com/Andriamarosoa/note/actions/runs/37853663514 (CI réussi). Sources et paramètres fixés avant ajustement par `README_V273_SERIE11_GARDE_RISQUE_NATIF.md`.

Les 24 politiques annoncées, les 38 modèles (19 morceaux × 2 estimateurs, chacun entraîné sans son morceau) et l'alignement des 59 309 fragments ont été générés. Tous les résultats concernent les **données de développement déjà exposées**.

| Politique | Exact global | Exact poly | Régressions vs freeze | Corrections vs freeze | Corrections de régression vs S9 | Corrections S9 perdues |
|---|---:|---:|---:|---:|---:|---:|
| Série 9, parent figé | 82,8913 % | 40,4739 % | 2 226 | 2 934 | 0 | 0 |
| Série 11 : vote + temps, logistic, tous K, seuil 0 | **83,3870 %** | 37,1970 % | **1 126** | **2 128** | 1 100 | 806 |
| Série 11 : vote + temps + acoustique partielle, HGB, tous K, seuil 0 | 83,3550 % | 37,1293 % | 1 075 | 2 058 | 1 151 | 876 |
| Série 11 : vote + temps + acoustique partielle, HGB, K0, seuil .25 | 83,0093 % | **40,3250 %** | 2 048 | 2 826 | 178 | 108 |

L'amélioration maximale globale face au parent donne +294 bonnes réponses nettes, mais dégrade la polyphonie et **ne conserve pas les corrections originales**. Aucune des 24 variantes n'améliore simultanément les exact-K global et poly du parent. Aucune politique promue. La série 12 est pré-déclarée, afin de fournir des caractéristiques acoustiques même aux événements K0/K1 et d'introduire un veto à deux modèles sans déduire la qualité sur les mêmes résultats.

Les vrais K des morceaux évalués n'entrent pas dans la formation du garde ; toutefois le parent série 9 a été choisi après examen de cette cohorte. **Les chiffres n'ont pas valeur de validation indépendante.**
