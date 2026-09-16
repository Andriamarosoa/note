# Correction de la dérive d'apprentissage — fold externe 3

L'audit a montré que 32 des 35 nouveaux surcomptages apparaissent avec une
entrée ln nulle. Réentraîner tout le réseau et recalibrer les seuils a
modifié les décisions même en l'absence de cet indice.

Cette correction part des **poids entraînés du contrôle du fold 3**.
Tous les poids existants, l'ancre et les seuils du contrôle sont figés.
Seule la projection linéaire sans biais `native_decay_projection`, ajoutée
aux caractéristiques internes avant les couches de comptage, est entraînée.
Il n'y a pas de correcteur supplémentaire après la prédiction.

À entrée ln nulle, l'ajout vaut zéro quels que soient les poids de cette
projection. Le modèle doit donc conserver les probabilités et décisions
du contrôle, à la précision numérique près. Des tests sur les trois réseaux
vérifient ce comportement après des mises à jour réelles, l'absence de
modification des anciens poids, et le rechargement des modèles exportés.

## Sélection interne déclarée avant exécution

- Fold externe : **3 seulement** ; validation interne : **0**, conformément
  aux partitions déjà figées. Les trois autres partitions servent à l'ajustement
  interne. Aucun réseau d'un autre fold externe n'est entraîné ou évalué.
- Initialisation : projection nulle sur les poids du contrôle, séparément
  pour les phases interne et finale. Chaque phase réutilise ses propres
  poids et garde ses anciennes partitions.
- Contexte acoustique calculé une fois avec le contrôle figé, dropout
  désactivé. Son équivalence avec l'inférence native complète est vérifiée.
- Adam 0,0002, lots de 128, mélange déterministe, 8 / 8 / 11 époques
  pour V260 uniforme / pondéré / V272. Les poids de classes sont calculés
  sur toute la partition d'apprentissage correspondante.
- Seules les lignes d'apprentissage à indice ln non nul sont présentées
  à l'optimiseur ; les autres ont un gradient de projection nul. Cela
  définit un nouvel entraînement de branche, avec son propre nombre de pas
  Adam ; ce n'est pas la reproduction du réentraînement complet précédent.
- Deux objectifs prévus : étiquette réelle seule, ou moyenne de l'étiquette
  réelle et de la distribution du contrôle figé. Le second limite les
  changements excessifs en conservant aussi l'information du contrôle.
- Trois forces de projection : **0,25 ; 0,5 ; 1**, communes aux trois réseaux.
  Elles sont appliquées aux poids de projection, sans changer les seuils.
  Cela donne six candidats, plus le contrôle.

Un candidat est admissible seulement s'il **augmente le nombre de bons K
polyphoniques**, ne réduit pas le nombre global de bonnes réponses et
n'augmente ni le surcomptage global ni le surcomptage polyphonique sur la
validation interne. Les candidats admissibles sont départagés par les
bonnes réponses polyphoniques, globales, les surcomptages, puis la force
la plus faible et l'objectif conservant l'information du contrôle.
Si aucun candidat ne passe, le contrôle est conservé.

Les frontières choisies sur la validation interne peuvent être sensibles
aux très faibles différences numériques entre CPU. Les probabilités du
contrôle recalculé doivent reproduire les archives à la tolérance fixée ;
tout changement de décision interne est consigné. Les critères de sélection
doivent être satisfaits face au contrôle recalculé **et** au contrôle archivé.
La parité ln=0 est vérifiée dans le même calcul pour éviter de confondre
différence numérique d'inférence et effet de la branche.

Le choix est écrit et haché dans `selection.json` avant l'ajustement final
et avant le calcul des scores du fold externe. Le candidat choisi est ajusté
sur les quatre partitions autorisées, avec les mêmes réglages et les poids
du contrôle final figés. Les probabilités, décisions, poids et seuils de
déploiement sont sauvegardés. Si le contrôle est retenu, ses poids et seuils
exactement conservés constituent la sortie.

## Interprétation

Cette architecture supprime la dérive du contrôle lorsque ln=0. Un gain
sur les lignes ln non nulles reste à mesurer ; aucune garantie de gain
global n'est annoncée. Les nouveaux surcomptages de ces lignes sont comptés.

Le fold 3 a déjà servi à l'audit qui motive cette correction. Même avec
une sélection interne séparée, son résultat est une vérification de
développement après observation, pas un nouveau test indépendant. Les
limites des sources candidates communes et des données reconstruites
restent celles du protocole initial. Aucune promotion automatique ni
agrégation de tous les folds n'est prévue ici.
