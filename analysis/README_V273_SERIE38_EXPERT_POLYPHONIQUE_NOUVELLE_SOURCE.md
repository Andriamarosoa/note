# S38 — apprendre l’existence d’une note polyphonique, au lieu de filtrer les mêmes réponses

## Expérience fixée avant entraînement

Après S35 (382 corrections / 309 régressions, poly 38,1043 %) et S36b (2/0, poly 40,6229 %), le point de blocage n’est plus l’ordre des têtes : **les spécialistes actuels ne distinguent pas assez bien les fondamentales simultanées des harmoniques parasites**. Une tête plus puissante est nécessaire, pas vingt nouveaux seuils.

**H9 est exclue** des poids, prédictions, entrées et décisions. S18 demeure immuable. H8 non disponible sans transformation réelle de WAV reste masquée.

Créer un **expert multi-classe K1 à K6**, entraîné sur l’acoustique originale des autres folds seulement :
- Entrées **V335** : 335 variables audio/votes/flux déjà extraites sans vérité du morceau testé.
- Entrées **V825** : V335 + 490 descripteurs de la véritable trajectoire temporelle 42×49 (attaque, décroissance, énergie, largeur, position de pic), standardisés ou calculés sans regarder les labels. Aucun pitch-shift simulé ou entrée YourMT3+.
- Deux familles de réseau/classes : `HistGradientBoostingClassifier` régularisé (150 itérations, max_depth=5, max_leaf_nodes=24, min_samples_leaf=25, l2=20, learning_rate=.065) et `ExtraTreesClassifier` (220 arbres, max_features=0.7, min_samples_leaf=3, n_jobs=2). Random seed fixée.
- Pour chaque fold externe 0,1,2,4, entraîner sur les **autres trois folds** les classes réelles K1–K6 ; évaluer uniquement sur le fold tenu à l’écart. Ces poids ne remplacent pas S18. Conserver modèles, scalers et SHA256 des identifiants de fit et d’évaluation.
- Évaluation de l’expert sur **tous vrais K2–K6** : accuracy globale des événements poly, et audit par vrai K2,K3,K4,K5,K6 et par fold. Comparer au parent S18 K0–K6 sur la même cohorte.
- Expériences d’intervention **préfixées** : modifier S18 seulement si son ancien K est dans K1–K6 ou K2–K6 et si l’expert propose une classe distincte, avec soit le choix argmax direct, soit `P(K_proposé) - P(K_S18) > 0.10` ou `>0.25`. Ces interventions donnent 4 modèles × 2 domaines ×3 règles = 24 politiques, + 4 expert-only diagnostiques (K0 ne peut pas être prédit par ces spécialistes, ne pas les présenter comme classifieurs globaux complets). Rapporter toute correction, régression, neutre, exact poly et global, sans retenir uniquement les succès.
- Si les modèles ne battent pas le niveau S18 **dans le diagnostic poly K2–K6** avec une marge utile, ne pas essayer de compenser la faiblesse par un nouveau juge de confiance. Si au moins un modèle ajoute des centaines de corrections mais avec des pertes, les conserver en tant que preuve des **nouvelles décisions nécessaires** et chercher un arbitre indépendant sur un prochain jeu de compositions jamais consulté.

Critère d’exploration : progrès **substantiel**, pas seulement +2, et comparaison explicite au poly 40,5958 %, puis YourMT3+ 54,5430 % comme *benchmark externe*, jamais comme entrée de l’expert. Aucun résultat issu de ce corpus de développement déjà consulté ne sera qualifié de validation finale indépendante ni promu automatiquement.