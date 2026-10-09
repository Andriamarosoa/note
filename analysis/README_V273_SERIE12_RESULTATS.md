# V27.3 — Série 12 : résultats audio complet + double accord

Run GitHub : https://github.com/Andriamarosoa/note/actions/runs/37853979683, terminé avec succès.

Série 12 utilise les 58 caractéristiques acoustiques de chacun des 59 309 événements (les quatre archives `v273-open-features-0/1/2/4.tar` du run d'origine 37837235723), sans label de test. Le sélecteur fournit 24 politiques (logistic, HGB, consensus AND et moyenne) entraînées avec exclusion du morceau évalué. Les distributions des producteurs G/P excluaient le fold évalué. Le parent série 9 est lui-même issu de choix sur les données exposées et n'a pas valeur de validation externe.

| Politique | Exact global | Exact poly | Régressions évitées vs S9 | Corrections perdues vs S9 | Régressions restantes vs freeze |
|---|---:|---:|---:|---:|---:|
| Parent série 9 | 82,8913 % | **40,4739 %** | 0 | 0 | 2 226 |
| Sér. 12 moyenne, tous K, seuil 0 | **83,3516 %** | 37,6168 % | 1 122 | 849 | **1 104** |
| Sér. 12 moyenne, tous K, seuil 0.1 | 83,3094 % | 38,5240 % | 813 | 565 | 1 413 |
| Sér. 12 double accord, tous K, seuil 0.1 | 83,1948 % | 38,9844 % | 558 | 378 | 1 668 |
| Sér. 12 double accord, K0, seuil 0.25 | 82,9739 % | 40,2031 % | 138 | 89 | 2 088 |

Conclusion : **aucune des 24 nouvelles politiques n'améliore simultanément le global et la polyphonie du parent**. Récupérer l'audio pour K0/K1 ne suffit pas sous cette représentation 58-dimensionnelle ; le veto moyen évite 1 122 régressions mais détruit 849 corrections. Aucun modèle promu. Résultats exploratoires, non indépendants. Conserver tous les poids, probabilités et prédictions en archive permanente.

Nouvelle piste à auditer sans supposer son efficacité : les trajectoires audio sur 42×49 (déjà extraites sans vérité) peuvent contenir une différence temporelle pré/post attaque que les seules moyennes acoustiques ne permettent pas d'exploiter.
