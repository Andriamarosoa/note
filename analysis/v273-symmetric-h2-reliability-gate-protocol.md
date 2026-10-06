# Protocole — gate de fiabilité pour symmetric pitch TTA H=2

Date : 6 octobre 2026.

## Point de départ

Le run 37404522178 a produit une configuration fixe pré-définie :

- vues -2,-1,0,+1,+2 demi-tons ;
- action 3→2 si mean(P2-P3) > 0 ;
- total interne : 26 corrections / 16 régressions = +10 ;
- nets fold 0/1/2/4 : +5, +6, -4, +3.

Cette expérience ne change pas cette règle de base. Elle apprend uniquement à
filtrer les actions dont la symétrie est peu fiable.

## Cohorte

Même 488 cas K2/K3, folds 0,1,2,4, fold 3 interdit.

## Features de fiabilité

Calculées uniquement à partir des cinq vues H=2 :

1. mean_margin = mean(P2-P3)
2. median_margin
3. std_margin
4. min_margin
5. max_margin
6. positive_margin_fraction
7. k2_vote_fraction
8. k3_vote_fraction
9. pair_asymmetry_1 = |margin(+1)-margin(-1)|
10. pair_asymmetry_2 = |margin(+2)-margin(-2)|
11. mean_pair_asymmetry
12. center_margin = margin(0)
13. outer_mean_margin = mean(margin(-2), margin(+2))
14. inner_mean_margin = mean(margin(-1), margin(+1))
15. margin_range

Aucune annotation acoustique ou vraie note comme feature.

## Apprentissage imbriqué

Pour chaque fold VAL=v :

- réserver v ;
- sur les trois folds FIT, générer des actions OOF du correcteur H=2 fixe ;
- gate train rows = uniquement les actions OOF ;
- cible gate=1 si l'action 3→2 est correcte (vrai K2), 0 si elle régresse (vrai K3) ;
- StandardScaler + LogisticRegression C=1 balanced, threshold 0.5 ;
- réentraîner le correcteur fixe n'est pas nécessaire car il n'a aucun paramètre appris ;
- appliquer le gate aux actions H=2 de VAL.

Si moins de 16 actions OOF ou une seule classe, abstention du gate pour la rotation.

## Comparaison

Rapporter :

- H=2 brut ;
- H=2 + gate symétrique ;
- corrections/régressions/net par fold et total ;
- AUC de fiabilité parmi les actions VAL.

## Progression

Progression si :

- net gated > +10 ;
- au moins 3/4 folds non négatifs ;
- pire fold >= -3 ;
- conserve >= 50% des 26 corrections brutes.

Aucune promotion automatique.
