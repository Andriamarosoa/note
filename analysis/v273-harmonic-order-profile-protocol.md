# Protocole — profil harmonique ordre par ordre

Date : 6 octobre 2026.

Ce protocole est écrit après le rejet du signal scalaire
`novel_marginal_gain` et avant le calcul des résultats de cette étape.

## Question

Les vraies notes attendues et les F0 sélectionnées non appariées présentent-elles
des profils de partiels différents que les agrégats précédents masquent ?

Cette étape est strictement diagnostique. Elle ne change aucune prédiction
Exact-K et n'entraîne aucun nouveau modèle d'inférence.

## Population figée

- mêmes 488 cas acoustiques ;
- K3_regressed=125, K3_preserved=147, K2_corrected=108, K2_missed=108 ;
- folds 0,1,2,4 uniquement ;
- fold 3 interdit ;
- chemin audio normal uniquement.

## Spectre

Même extraction que les audits acoustiques :

- PRE : [s-2048,s)
- POST1 : [s,s+2048)
- Hann ;
- FFT 8192 ;
- bande d'analyse existante ;
- spectre d'attaque `A=max(P_POST1-P_PRE,0)`.

## Mesures h1…h10

Pour chaque F0 f et ordre harmonique h=1..10 encore dans la bande :

- `E_h(f)` = somme de A dans +/- KERNEL_HZ autour de h*f ;
- `global_fraction_h = E_h / (sum(A)+eps)` ;
- `profile_fraction_h = E_h / (sum_j E_j + eps)` ;
- `pre_fraction_h` et `post1_fraction_h` par rapport aux puissances totales
  PRE et POST1 ;
- `onset_ratio_h = log((POST1_h+eps)/(PRE_h+eps))`.

Aucun regroupement des ordres n'est fait avant de voir les résultats.

## Rôles

À partir de l'appariement 55 cents déjà utilisé :

- expected ;
- selected_matched ;
- selected_unmatched.

Les annotations ne servent qu'à attribuer ces rôles.

## Comparaisons principales

Priorité aux vrais K3 :

1. expected vs selected_unmatched, ordre par ordre ;
2. K3_regressed vs K3_preserved pour les expected ;
3. mêmes comparaisons par fold 0/1/2/4.

Pour chaque ordre et métrique, rapporter :

- n ;
- médiane et moyenne par rôle ;
- différence médiane expected - selected_unmatched ;
- AUC descriptif pour distinguer expected de selected_unmatched dans chaque fold,
  avec orientation rapportée mais sans sélectionner un seuil.

## Diagnostic de chevauchement

Pour chaque selected_unmatched et chaque ordre h, indiquer si son centre h*f tombe
à +/- KERNEL_HZ d'un harmonique 1..10 d'une fréquence expected du même cas.

Rapporter par ordre :

- taux de partiels parasites qui chevauchent un partiel expected ;
- énergie correspondante.

Ce diagnostic est annotation-conditionné et ne peut pas devenir directement une
feature d'inférence.

## Critère pour la suite

Une nouvelle feature d'inférence n'est autorisée que si un ou plusieurs ordres
montrent un contraste cohérent sur plusieurs folds, avec direction stable, et
que ce contraste n'est pas simplement le reflet d'un chevauchement annoté.

Sinon, conclure que le profil h1…h10 ne fournit pas de structure exploitable
supplémentaire sous cette représentation.

## Sorties

- components.jsonl ;
- report.json ;
- report.md ;
- aucun gain Exact-K annoncé ;
- aucune promotion ;
- empreintes des sources et du script.
