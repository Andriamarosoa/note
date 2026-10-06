# Protocole — gain marginal nouveau de la troisième composante

Date : 6 octobre 2026.

Ce protocole est écrit après le rejet du reranking exclusif
`v273-exclusive-rerank-37392442200` et avant toute évaluation du nouveau
signal.

## Question

Le passage du meilleur couple interne au triplet déjà choisi apporte-t-il une
réduction d'erreur dans des bins harmoniques réellement nouveaux, non déjà
couverts par les deux composantes du couple ?

L'objectif est de conserver le reconstructeur historique et de mesurer
l'information marginale de la troisième composante, plutôt que de maximiser
l'exclusivité.

## Base figée

- mêmes 1 666 lignes audio uniques ;
- même population B_low + base_prediction K3 ;
- mêmes 845 lignes VAL au total ;
- pool-64, saillance historique 1/sqrt(h), gabarit gaussien 1/h² ;
- fréquences du couple/triplet de contrôle archivées inchangées ;
- base neuronale et routage B_low figés ;
- folds 0,1,2,4 seulement ; fold 3 interdit ;
- chemin normal uniquement.

## Définition de la troisième composante

Pour le triplet de contrôle déjà choisi :

1. Ajuster NNLS sur les trois colonnes du triplet.
2. Ajuster NNLS sur ses trois sous-couples possibles.
3. Retenir le sous-couple de coût minimal, avec ordre lexicographique pour les
   égalités.
4. La composante absente de ce meilleur sous-couple est appelée composante
   marginale.

Cette définition ne dépend d'aucune annotation.

## Bins nouveaux

Pour la composante marginale f3 et le meilleur sous-couple (f1,f2) :

`U = H(f3) \ (H(f1) union H(f2))`

où H utilise exactement MAX_HARMONICS, KERNEL_HZ et la bande d'analyse du
pipeline existant.

## Signal préenregistré

Soient :

- y2 : reconstruction NNLS du meilleur sous-couple ;
- y3 : reconstruction NNLS du triplet ;
- d = y3-y2 ;
- J2 et J3 : erreurs quadratiques correspondantes ;
- x2 = ||x||².

Définir :

`marginal_gain = max(J2-J3,0)/(x2+1e-12)`

`delta_unique_fraction = ||d[U]||²/(||d||²+1e-12)`

et l'unique feature nouvelle :

`novel_marginal_gain = marginal_gain * delta_unique_fraction`.

Aucun seuil, poids ou exposant n'est ajusté.

## Deux modèles seulement

- `control_residual2` : [best_pair_residual_ratio,
  best_triplet_residual_ratio] du bras pool64_t2 archivé.
- `plus_novel3` : les deux mêmes features + novel_marginal_gain.

Même StandardScaler + LogisticRegression(C=1, class_weight=balanced, lbfgs).
Le fit de la LR utilise seulement vrais K2/K3, comme auparavant.

## Sélection Exact-K

Pour chaque fold VAL 0/1/2/4 :

- sélection du modèle exclusivement par rotation sur les folds FIT ;
- critère : net Exact-K, puis moins de régressions, puis moins d'actions, puis
  ordre des modèles ;
- abstention si le meilleur net FIT n'est pas strictement positif ;
- seuil de décision fixe 0,5 ;
- toutes les actions sur autres K sont comptées.

Les résultats fixes des modèles sur VAL sont descriptifs et ne sélectionnent
rien.

## Diagnostics

Rapporter sans sélection supplémentaire :

- distribution du nouveau signal pour K2 vs K3 ;
- AUC univarié K2-vs-K3 par fold VAL ;
- gain Exact-K fixe de chaque modèle par fold ;
- corrections/régressions et autres-K ;
- contrôle bit-à-bit des deux résidus contre le bras pool64_t2 archivé.

## Décision

Le signal n'est retenu pour recherche ultérieure que si le modèle
`plus_novel3` est sélectionné sur FIT dans plusieurs rotations et produit un
bilan VAL sélectionné positif sans forte régression sur un fold.

Aucune promotion automatique et aucune évaluation du fold 3.
