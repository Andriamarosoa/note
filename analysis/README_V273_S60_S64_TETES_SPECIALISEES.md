# S60–S64 — nouvelles têtes spécialisées, audit et décisions

**Statut : exploration sur développement historique, NON PROMUE.** Les folds natifs 0,1,2,4 ont déjà servi à d'autres recherches ; ils ne constituent pas une épreuve indépendante. La référence non modifiée est `freeze_local_combo`, le parent exploratoire de ces boucles est S58.

## Audit des 409 régressions du parent S58

59 309 événements, 1 852 décisions modifiées : **844 corrections**, **409 régressions** et **599 mauvaises→mauvaises**.

Par transition, les plus nombreuses : `K3→2` 136 régressions / 223 corrections ; `K2→3` 83 / 121 ; `K4→3` 42 / 78 ; `K3→4` 41 / 73 ; `K1→0` 37 / 150 ; `K0→1` 29 / 84. Les mêmes transitions contiennent donc aussi des corrections légitimes : un veto statique serait dommageable.

## Nouvelles têtes effectivement entraînées

| Architecture ou ablation | Corrections | Régressions | Gain net vs freeze | Décision |
|---|---:|---:|---:|---|
| S58 parent | 844 | 409 | +435 | Référence exploratoire |
| S60 : 6 têtes Ridge sur transitions, paramètres choisis hors fold évalué | 826 | 397 | +429 | **Rejet** |
| S61 : contre-preuves des voisins du même enregistrement | 838 | 405 | +433 | **Rejet** |
| S62 : 6 têtes non linéaires (acoustique, morphologie, votes, voisins) | 840 | 401 | +439 | Exploratoire +4 |
| S63 : contre-vote de la prédiction initiale (seuil par source) | 842 | 405 | +437 | Exploratoire +2 |
| **S62+S63** | **838** | **397** | **+441** | **Exploratoire, NON PROMU** |
| S64 : troisième candidature K, tête globale | 835 | 409 | +426 | **Rejet** |
| S64 : troisième candidature K, tête par source | 839 | 409 | +430 | **Rejet** |
| S64 : troisième candidature K, tête par transition | 837 | 409 | +428 | **Rejet** |

Le système combiné S62+S63 refuse **39 actions** : 12 régressions évitées, 6 corrections perdues, 21 mauvaises→mauvaises annulées. Elles concernent exclusivement K3→K2 (25 refus, +3 net) et K2→K3 (14 refus, +3 net). Gains additionnels par fold : +2, 0, +3, +1. **10 414 erreurs restantes** ; **48 895/59 309 Exact-K correct global** ; **2 773/7 385 correct polyphonique**. Les vrais K4 ne sont pas améliorés par ces deux têtes.

Le signal des votes des événements voisins atteint une AUC de discrimination R/C ≈ 0,58 ; insuffisant pour un veto général. Une tête acoustique à 7 classes entraînée sur toute la cohorte n'apporte aucune amélioration crédible des refus K2↔3/K3↔4 (AUC ≈0,48–0,51). Le constructeur S64 identifie la bonne classe parmi les 36 politiques pour 347 des 599 cas mauvaises→mauvaises, mais les décisions sélectionnées ne généralisent pas sur les folds évalués : les gains deviennent négatifs.

**Robustesse indicative** : 19 morceaux, bootstrap des effets additionnels S62+S63 sur les mêmes morceaux : intervalle percentile 95 % +1 à +12. Ceci ne corrige pas le biais de sélection historique, donc **aucune généralisation indépendante démontrée**.

## Implémentation et intégrité

La banque expérimentale `scripts/experimental_v273_regression_heads.py` définit les six sous-populations disjointes, le codage correction/régression, un constructeur d'arbres régularisés, les compteurs de votes et la règle de refus qui retourne uniquement vers la référence figée. Elle **ne charge aucun poids et ne remplace pas le modèle principal**. Tests : `test/test_experimental_v273_regression_heads.py`. Le rejeu complet S60–S64 est archivé en scripts et résultats locaux, avec données sources : PR16, S58 historique, résumés acoustiques natifs déjà prétraités. Le code d'inférence ne doit jamais recevoir la vérité terrain.

Aucune nouvelle tête ne sera promue avant évaluation sur des enregistrements jamais utilisés, exclusion imbriquée stricte des producteurs, et preuve de gain net et de baisse des régressions par K. Prochain verrou majeur : encore 397 régressions, dont K3 reste la plus importante, et la génération de nouveaux candidats K4/K5/K6.
