# Protocole — forme complète de la trajectoire K sous pitch shift

Date : 6 octobre 2026.

Cette étape réutilise uniquement les trajectoires déjà générées par le run
37400666855. Aucun nouveau pitch shift n'est recalculé et aucun fold 3 n'est lu.

## Question

La **forme complète** de K(+0..+24) distingue-t-elle mieux les vrais K2 des
vrais K3 que le seul premier passage à K=2 ?

## Population

- 488 cas figés B_low + base K3 ;
- K2=216, K3=272 ;
- folds internes 0,1,2,4 ;
- rotation FIT = trois folds, VAL = quatrième ;
- fold 3 interdit.

## Features figées

Le pas +0 vaut toujours K3 et n'est pas compté dans les fractions de temps.

À partir de K(+1..+24) :

1. fraction_k2
2. fraction_k3
3. fraction_below3
4. transitions_3_to_2
5. transitions_2_to_3
6. transitions_total
7. longest_k2_run
8. longest_k3_run_after_exit
9. total_variation = sum |K_s - K_{s-1}|
10. unique_k_count
11. state_entropy
12. final_k
13. first_k2_step (25 si absent)
14. first_non3_step (25 si absent)
15. first_below3_step (25 si absent)
16. reentered_k3 (0/1)
17. monotone_nonincreasing (0/1)

Aucune sélection de sous-ensemble et aucune transformation apprise sur VAL.

## Modèle

- StandardScaler
- LogisticRegression
- C=1
- class_weight=balanced
- solver=lbfgs
- random_state=39431
- seuil de décision fixe 0.5
- cible : vrai K2 = 1, vrai K3 = 0

Le modèle est ajusté uniquement sur FIT puis appliqué tel quel à VAL.

## Comptabilité

Comme la prédiction de base vaut K3 pour toute la cohorte :

- prédire K2 => action 3→2 ;
- vrai K2 => correction ;
- vrai K3 => régression.

Rapporter corrections, régressions et net.

## Diagnostics

- AUC VAL par fold ;
- coefficients standardisés par fold ;
- statistiques univariées des 17 features ;
- matrice de corrélation FIT ;
- calibration non revendiquée.

## Critère

Cette représentation n'est considérée utile que si :

- net total cross-fold > 0 ;
- au moins 3/4 folds ont un net >= 0 ;
- le gain ne repose pas sur un seul fold ;
- AUC moyenne > 0.55.

Sinon, la forme complète de la trajectoire n'apporte pas de séparation exploitable.

Aucune promotion automatique.
