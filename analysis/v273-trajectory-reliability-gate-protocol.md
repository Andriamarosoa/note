# Protocole — gate de fiabilité du correcteur par trajectoire

Date : 6 octobre 2026.

## But

Le classifieur de forme complète de trajectoire corrige certains cas
`B_low + base K3`, mais son action 3→2 est fiable sur certains folds et
nocive sur d'autres. Cette expérience ne cherche pas une nouvelle correction
globale. Elle teste si l'on peut **router uniquement les actions fiables**.

## Sources figées

- trajectoires +0..+24 demi-tons : run 37400666855 ;
- résidus pair/triplet et probabilités de base : run 37356100423 ;
- cohorte : 488 cas K2/K3 ;
- folds internes : 0,1,2,4 ;
- fold 3 interdit.

## Niveau 1 : correcteur trajectoire

Identique au run 37402050672 :

- 17 features de forme de trajectoire ;
- StandardScaler + LogisticRegression C=1 balanced lbfgs ;
- K2=1, K3=0 ;
- action 3→2 si p(K2) >= 0.5.

## Niveau 2 : gate de fiabilité

Le gate n'est entraîné que sur les lignes où le niveau 1 **agit**.

Pour chaque rotation externe VAL=v :

1. réserver v entièrement ;
2. sur les trois folds FIT restants, construire des prédictions OOF :
   pour chaque fold h de FIT, entraîner le niveau 1 sur les deux autres folds,
   prédire h ;
3. sur les actions OOF uniquement, définir :
   - cible gate=1 si vrai K2 (action correcte) ;
   - cible gate=0 si vrai K3 (régression) ;
4. entraîner le gate sur ces actions OOF ;
5. entraîner le niveau 1 sur les trois folds FIT ;
6. prédire VAL ;
7. laisser passer une action VAL seulement si p_gate >= 0.5.

Ainsi le gate ne voit jamais une décision niveau 1 produite par un modèle
entraîné sur la même ligne.

## Deux gates préenregistrés

### A — trajectory_only

Entrées :

- les 17 features de trajectoire ;
- p_traj_K2 du niveau 1.

### B — trajectory_plus_acoustic

Entrées de A plus :

- best_pair_residual_ratio ;
- best_triplet_residual_ratio ;
- median_triplet_f0 ;
- base_probability_K2 ;
- base_probability_K3 ;
- base_margin_K3_minus_K2.

Toutes ces entrées sont disponibles en inférence dans les exports existants.
Aucune annotation de note n'est utilisée comme feature.

## Modèle du gate

Pour A et B :

- StandardScaler ;
- LogisticRegression C=1 ;
- class_weight=balanced ;
- solver=lbfgs ;
- random_state=39531 ;
- seuil fixe 0.5 ;
- aucune recherche de seuil ;
- aucune sélection de features.

Si les actions OOF n'ont pas les deux classes ou moins de 24 lignes, le gate
abstient pour cette rotation.

## Comptabilité

Rapporter pour chaque fold :

- correcteur trajectoire brut : actions, corrections, régressions, net ;
- gate A : idem ;
- gate B : idem ;
- nombre d'actions bloquées ;
- AUC de fiabilité sur VAL parmi les actions du niveau 1, si définie.

Les actions autres que K2/K3 n'existent pas dans cette cohorte figée.

## Critère de progression

Le gate B est considéré comme une vraie progression interne seulement si :

- net total > net du correcteur trajectoire brut ;
- net total > 0 ;
- au moins 3/4 folds ont net >= 0 ;
- aucune rotation ne perd plus de 5 Exact-K ;
- il conserve au moins 25 % des corrections brutes.

Gate A sert à séparer l'effet "forme de trajectoire" de l'apport acoustique.

Aucune promotion automatique et aucun fold 3.
