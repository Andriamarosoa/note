# V27.3 — série 4 : attaques attribuées au fragment

Protocole adaptatif fixé après observation des séries 1–3 et avant tout ajustement de cette série. Aucun résultat de développement ne sera présenté comme validation inédite. Objectif inchangé : dépasser simultanément 51 328/59 309 au global et 4 028/7 385 en polyphonie, puis confirmer sur données inédites.

## Hypothèse et contrôles

La série 3 atteint 83,2808 % au global mais 34,4888 % en polyphonie. Son meilleur score global change 3 375 verdicts du précédent meilleur : 1 614 corrections, 838 régressions, gain net 776. Il prévoit 1 547,2 gains nets contre 776 réalisés. Les anciennes moyennes acoustiques et l'aplatissement des trajectoires ne contraignent pas le modèle à distinguer attaque nouvelle, résonance et attaque voisine. Cette insuffisance de représentation est une hypothèse, pas une cause acoustique démontrée.

Réutiliser exclusivement les activations harmoniques et la géométrie conservées par la série 2, sans nouvelle lecture de vérités pour fabriquer les caractéristiques. Les activations restent des proxys : une composante harmonique n'est pas une note indépendante. Aucun verdict YourMT3+ comme entrée.

Deux représentations :

1. contrôle : 58 résumés acoustiques, géométrie existante de 46 valeurs, codage one-hot du verdict freeze (111 entrées) ;
2. contrôle augmenté de statistiques des augmentations d'activation par hauteur et par temps, dans la zone attribuée au fragment et dans les zones adjacentes. Décalages temporels fixés à 1, 2, 4 et 8 trames ; zones : propriété native, propriété dilatée de deux trames, avant et après. Pour chaque zone/décalage : valeurs maximales par hauteur triées (8 premières), masse totale, concentration, nombre effectif, quantités dépassant 2 %, 5 %, 10 % et 20 % du maximum, statistiques du flux temporel et sa concentration. Ajouter pour chaque zone les mêmes statistiques sur l'activation, ainsi que les masses négatives. Pas de hauteur absolue comme nouvelle entrée.

Les modèles apprennent la vérité K0–K6 sur les trois autres folds entiers. Aucun label du fold receveur, aucun ajustement par morceau receveur, aucune identité d'enregistrement en entrée. Un modèle HGB par représentation et pondération poly 1 ou 2 : 16 ajustements (4 folds × 2 × 2). Paramètres fixes : learning_rate=.06, max_iter=200, max_leaf_nodes=31, max_depth=6, min_samples_leaf=40, l2_regularization=10, early_stopping=False, random_state=27407. Les poids de classe sont normalisés sur l'apprentissage seulement ; aucune calibration n'est revendiquée.

Décodage intégral K0–K6 : inclinaison poly 1, 1.5 ou 2 ; mélange alpha .6, .8 ou 1 avec freeze et avec le précédent meilleur déjà figé. Soit 72 politiques, doublons inclus et conservés. Les sorties du précédent meilleur ne sont jamais des entrées ni des cibles de l'ajustement : seul le décodage final les utilise. Une variante mélangée avec ce parent n'est donc pas un système final sans labels du fold receveur. Les variantes avec freeze le sont sous réserve de la provenance déjà documentée de freeze.

Conserver tous les modèles, vecteurs, cas corrigés/régressés, exclusions et alias. Rejouer les modèles sauvegardés. Comparer chaque politique à freeze, au précédent meilleur et à YourMT3+, au global, K0–K6, polyphonie et par fold. Auditer la surestimation et les transitions des régressions. Aucun seuil ne sera modifié silencieusement après résultats.
