# Protocole — reranking par support harmonique exclusif

Date : 6 octobre 2026.

Ce protocole est écrit après l'audit diagnostique
`v273-exclusive-harmonic-support-37391242629` et avant toute évaluation
Exact-K de ce nouveau bras.

## Motivation

Sur les 272 vrais K3, la fraction médiane de support d'attaque exclusif est
environ 0,731 pour les fréquences attendues contre 0,163 pour les composantes
sélectionnées non appariées. Les composantes non appariées tirent environ 0,894
de leur énergie d'attaque des mêmes zones harmoniques que les notes attendues.
Le sens est cohérent sur les folds internes 0, 1, 2 et 4.

Ces annotations ne peuvent pas être utilisées en inférence. Le bras testé ici
utilise uniquement l'exclusivité **interne à une combinaison candidate**.

## Base figée

- mêmes 1 666 lignes audio uniques du chemin normal ;
- mêmes 845 lignes VAL d'action ;
- base neuronale et routeur B_low figés ;
- population : B_low + prédiction de base K3 ;
- pool de 64 F0 produit par la saillance historique 1/sqrt(h) ;
- dictionnaire gaussien 1/h² ;
- même NNLS, même petite LR, même seuil 0,5 ;
- folds internes 0, 1, 2, 4 uniquement ;
- fold 3 interdit.

Le bras de contrôle doit reproduire bit pour bit le bras `pool64_t2` archivé.

## Reranking sans annotations

Pour k=2 et k=3 :

1. Calculer le coût NNLS normalisé de **toutes** les combinaisons du pool-64.
2. Conserver les 64 combinaisons de plus faible coût NNLS. Le nombre 64 est
   fixé avant l'évaluation et correspond à la capacité du pool ; il n'est pas
   ajusté par fold.
3. Pour chaque combinaison C et chaque composante f de C, calculer sur le
   spectre d'attaque positif `x` :
   - `H(f)` : union des bandes harmoniques existantes ;
   - support total : somme de x dans H(f) ;
   - support exclusif : somme de x dans H(f) hors de l'union des H(g) des
     autres composantes de C ;
   - `u(f)=exclusif/(total+1e-12)`.
4. Définir `u_min(C)=min_f u(f)`.
5. Définir le score de reranking :
   `R(C)=u_min(C)/(J(C)+1e-12)`,
   où J est le coût NNLS normalisé.
6. Parmi les 64 combinaisons présélectionnées, choisir le plus grand R.
   Départage : coût J plus faible, puis rang NNLS initial.

Aucun seuil d'unicité et aucun poids libre ne sont ajustés.

## Features Exact-K

Le contrôle utilise les deux résidus NNLS du meilleur couple/triplet historique.

Le bras `exclusive_rerank64` utilise les **résidus NNLS bruts** des couple et
triplet retenus par le reranking. L'unicité n'est pas ajoutée comme troisième
feature LR afin de ne pas confondre le diagnostic avec une nouvelle dimension
apprise.

## Sélection

Pour chaque fold VAL parmi 0/1/2/4 :

- la sélection contrôle vs reranking est faite uniquement par rotation interne
  sur les trois folds FIT ;
- critère identique aux expériences précédentes : net Exact-K, puis moins de
  régressions, puis moins d'actions, puis ordre des bras ;
- abstention si le meilleur net FIT n'est pas strictement positif ;
- après sélection, réajuster la petite LR sur tout FIT et évaluer VAL.

Les résultats fixes de chaque bras sur VAL restent descriptifs et ne servent
jamais à choisir le bras.

## Diagnostics annotés après évaluation

Sur les 488 cas acoustiques déjà figés seulement :

- nombre de notes attendues retrouvées par le couple/triplet contrôle ;
- même mesure après reranking ;
- nombre de triplets K3 complets et couples K2 complets.

Ces annotations ne changent ni la sélection FIT ni les scores.

## Décision

Le bras n'est intéressant que si :

- la sélection FIT l'autorise au moins dans plusieurs rotations ;
- le total VAL sélectionné est positif ;
- aucun fold n'a une régression importante ;
- corrections/régressions complètes sont rapportées.

Aucune promotion automatique, aucune évaluation fold 3 et aucune conclusion
sur le chemin compressé.
