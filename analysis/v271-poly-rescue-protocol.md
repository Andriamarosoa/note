# V27.1 — `poly_rescue` sélectif

## État gelé avant l'expérience

La référence de développement est V27 `low_k_fusion` :

- exact-K polyphonique : **37,5705 %** ;
- gain vs V10.4 : **+1,0637 point** ;
- gain d'exact-K polyphonique sur les cinq folds ;
- exact-K global : **82,4953 %** (**+0,5614 point** vs V10.4) ;
- F1 événementiel à 50 ms : **80,1896 %** (**−0,1946 point** vs V10.4).

La métrique principale est l'**exact-K polyphonique**. L'exact-K global, le F1 à 50 ms et TP/FP/FN restent secondaires et sont toujours publiés.

## Pourquoi `poly_rescue` n'est pas un seuil sur V26-uniform

`low_k_fusion` utilise déjà V26-`uniform` sur **toutes** les lignes où V10.4 prédit `K<=1`, y compris lorsque V26-`uniform` prédit `K>=2`. Ajouter un seuil sur ce même signal ne peut donc créer aucune nouvelle correction polyphonique par rapport à V27 ; cela ne pourrait que retirer des rescues déjà présents.

V27.1 exploite à la place le bras V26 **`weighted`**, entraîné dans le même run V26 et sauvegardé dans les mêmes artefacts, comme spécialiste polyphonique complémentaire.

## Règle testée

On part toujours de `low_k_fusion`.

Une ligne est éligible au rescue seulement si :

1. `V10.4_K <= 1` ;
2. `low_k_fusion_K <= 1` — V27 n'a donc pas déjà récupéré cette ligne en polyphonie ;
3. `V26_weighted_K >= 2`.

Pour une ligne éligible, la confiance est :

`P_v260_weighted(K = argmax)`

c'est-à-dire la probabilité attribuée par V26-weighted à la **classe K exacte qu'il propose**. Ce choix est fixé avant l'évaluation externe car l'objectif principal est l'exact-K, pas seulement la détection `poly / non-poly`.

Le rescue est appliqué si cette confiance est supérieure ou égale au seuil calibré du fold. Sinon la prédiction `low_k_fusion` est conservée.

## Garde-fou structurel

Le traitement ne peut modifier qu'une prédiction `low_k_fusion <= 1`.

Donc, pour une vérité `K>=2`, toute ligne déjà exacte avec `low_k_fusion` a nécessairement une prédiction `>=2` et est **inéligible**. V27.1 ne peut ainsi diminuer l'exact-K polyphonique de V27, ni globalement ni sur un fold individuel, indépendamment du seuil choisi.

## Calibration nested, fold par fold

Pour chaque outer fold :

- l'outer fold reste totalement exclu de la sélection du seuil ;
- le meta-validation fold est le même que dans V26 : le plus petit identifiant parmi les quatre folds restant après retrait de l'outer fold ;
- les quatre shards V10.4 nested du run source `33647694565` fournissent les prédictions d'experts strictement outer-clean ;
- la fusion V10.4 interne est rejouée sur ces shards avec l'architecture et le protocole V10.4 existants ;
- les probes V26 `uniform` et `weighted` sont rejoués sur `meta_fit -> meta_val` avec les seeds, pondérations, early stopping et architecture du run V26 ;
- le nombre d'epochs V26 re-sélectionné doit être identique au rapport V26 source, sinon le fold échoue avant toute évaluation externe.

Les probabilités externes V10.4 et V26 ne sont **pas réentraînées** pour l'évaluation : on réutilise les artefacts outer-clean déjà gelés de V11.1/V26 afin que `low_k_fusion` reproduise exactement V27.

## Sélection du seuil interne

Tous les seuils de rupture observés parmi les confidences des lignes éligibles internes sont évalués, avec en plus une option explicite « aucun rescue ».

L'ordre de sélection, fixé avant lecture des outer folds, est lexicographique :

1. maximiser le nombre de lignes polyphoniques exactes ;
2. à égalité, maximiser le nombre total de lignes exactes ;
3. à égalité, minimiser le nombre de rescues activés ;
4. à égalité, préférer le seuil le plus élevé.

Le F1 événementiel n'intervient jamais dans le choix du seuil.

## Évaluation externe

Une fois le seuil du fold figé :

- il est appliqué une seule fois à l'outer fold ;
- les labels outer ne servent qu'au rapport final ;
- sont publiés : exact-K polyphonique, exact-K global, F1 à 50 ms, TP/FP/FN, seuil, nombre de rescues et transitions `V10.4=0 -> poly` / `V10.4=1 -> poly` ;
- les cinq folds sont publiés sans sélection a posteriori.

## Décision fixée avant le run

`poly_rescue` devient la nouvelle référence uniquement si son **exact-K polyphonique agrégé est strictement supérieur à 37,5705 %**.

L'exact-K global et le F1 restent des métriques secondaires publiées et quantifiées, mais ne constituent pas un veto post-hoc. En cas d'égalité sur l'exact-K polyphonique agrégé, V27 `low_k_fusion` reste la référence la plus simple.

Aucune validation historique ni `locked12` n'est indexée ou évaluée dans cette expérience.
