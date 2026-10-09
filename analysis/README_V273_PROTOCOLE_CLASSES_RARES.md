# V27.3 — série 5 : classes rares

Protocole adaptatif fixé avant ajustement, après les séries 1–3 et avant lecture du bilan complet de série 4. Le meilleur combinateur de série 3 reconnaît 0/89 K6 ; les trois folds d'entraînement ont 54–74 K6, contre 28 889–30 630 K0. Ce déséquilibre justifie un test distinct ; il ne prouve pas que sa correction suffira.

Réutiliser exactement la représentation `owned` de série 4 (527 entrées, incluant freeze), sans verdict YourMT3+, identité ou label comme caractéristique. Un modèle HGB par fold évalué, entraîné sur les trois autres folds, mêmes paramètres que série 4. Seule la pondération change : poids de la classe k = min(8, sqrt(N_train / (7 * n_train,k))), puis normalisation du poids moyen sur l'apprentissage. Poids entièrement calculés hors du fold évalué. Quatre modèles indépendants peuvent être exécutés simultanément ; aucune adaptation des hyperparamètres selon un résultat partiel.

Conserver deux interprétations des sorties : probabilités pondérées brutes, et correction du prior divisant chaque sortie par son poids de classe d'entraînement puis renormalisant. Cette dernière est une approximation et ne garantit pas une calibration.

Pour chacune : inclinaison poly 1 ou 1.5, mélange alpha .6, .8 ou 1 avec freeze et avec le précédent meilleur figé. Soit 24 politiques, alias inclus, toutes conservées. Les sorties du parent précédent servent uniquement au décodage, jamais à l'ajustement. Elles gardent leurs limites d'indépendance déjà documentées.

Critères : métriques natives et K0–K6 ; corrections/régressions contre chaque baseline ; replay des modèles et exclusions. Aucun seuil ajusté sur le fold prédit. Objectif YourMT3+ strictement inchangé, aucune promotion, validation sur données inédites toujours requise. Les chiffres oracle ne seront pas présentés comme un système obtenu.
