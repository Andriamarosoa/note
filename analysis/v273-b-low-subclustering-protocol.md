# Protocole — sous-clustering de B_low + prédiction K3

Date : 6 octobre 2026.

## Question

Le sous-ensemble `B_low + base_prediction K3` mélange-t-il plusieurs régimes
structurels distincts dont certains peuvent être corrigés 3→2 sans dégrader les
vrais K3 ?

Cette expérience teste directement l'hypothèse des sous-clusters. Elle ne
rajoute aucune nouvelle feature acoustique et n'utilise pas le fold 3.

## Population

Pour chaque rotation interne avec VAL dans {0,1,2,4} :

- FIT = les trois autres folds parmi {0,1,2,4} ;
- population = lignes `B_low & base_K==3` ;
- clustering formé sur FIT uniquement ;
- transfert des clusters vers VAL par le même scaler/PCA/KMeans ;
- fold 3 strictement interdit.

Les vrais K ne sont jamais utilisés pour former les clusters ni choisir K.

## Features de clustering

Utiliser uniquement `*_structural` déjà exporté par l'audit résiduel vérifié.

Ces 37 dimensions sont disponibles à l'inférence et contiennent :

- stats structurelles ;
- nombre de candidats ;
- moyennes/max des 8 features pseudo-count/contexte V8.8 ;
- 5 résumés spectraux ;
- 7 probabilités courantes du modèle.

Aucun résidu pair/triplet, aucune annotation de fréquence et aucun vrai K ne
sert à la géométrie du clustering.

## Prétraitement

Sur FIT seulement :

1. StandardScaler ;
2. PCA à variance expliquée >= 90 %, avec au maximum 16 composantes ;
3. KMeans.

Le même scaler et la même PCA sont appliqués à VAL.

## Nombre de sous-clusters

Tester K = 2,3,4,5,6 sur FIT uniquement.

Pour chaque K :

- KMeans `n_init=100` ;
- calculer silhouette FIT ;
- rejeter K si un cluster contient moins de 5 % des lignes FIT ;
- mesurer en diagnostic la stabilité par 8 réinitialisations supplémentaires
  et l'ARI moyen vers la partition principale.

Sélectionner K par silhouette FIT maximale parmi les K admissibles.
Départage : K plus petit.

Aucun label n'intervient dans cette sélection.

## Politique de correction cluster-par-cluster

Après que le clustering est figé :

Pour chaque cluster c sur FIT :

- `gain_fit(c) = #trueK2 - #trueK3` parmi les lignes du cluster ;
- activer `3→2` dans ce cluster uniquement si `gain_fit(c) > 0`.

Les autres vrais K sont comptés mais ne déterminent pas la règle.

Appliquer ensuite cette liste de clusters activés à VAL sans modification.

Rapporter :

- corrections K2 ;
- régressions K3 ;
- actions sur autres K ;
- net Exact-K ;
- résultat par cluster.

## Oracle diagnostique

Avec la partition VAL déjà figée, calculer également :

`oracle_val_net = sum_c max(#K2_c - #K3_c, 0)`.

Cet oracle utilise les labels VAL uniquement après coup. Il ne sélectionne aucun
K, cluster, seuil ou modèle. Il indique seulement si la partition géométrique
contient en principe des sous-régimes séparables.

Deux interprétations importantes :

- oracle faible : le clustering lui-même ne sépare pas le problème ;
- oracle élevé mais politique FIT faible : structure potentielle mais clusters
  non stables/généralisables.

## Contrôles

- les partitions FIT/VAL et enregistrements doivent être disjoints ;
- toutes les lignes action B_low/base-K3 sont conservées ;
- aucune annotation acoustique ;
- aucun fold 3 ;
- aucune promotion automatique.

## Décision

La piste sous-clustering est considérée utile seulement si :

- K sélectionné > 1 naturellement ;
- plusieurs rotations montrent des clusters avec des compositions K2/K3
  nettement différentes ;
- la politique FIT cluster-par-cluster donne un net VAL positif et raisonnablement
  stable ;
- l'oracle VAL confirme qu'il existe réellement du potentiel de séparation.

Sinon, multiplier les sous-clusters ne résout pas le blocage actuel.
