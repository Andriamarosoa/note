# Protocole — sous-types RUBBER_ONLY H2

Date : 6 octobre 2026.

## Motivation

Le routing par désaccord a montré :

- RUBBER_ONLY interne : 10 corrections / 13 régressions = -3 ;
- Rubber Band H2 outer global : +2 net ;
- le gain outer vient donc potentiellement d'un sous-ensemble de RUBBER_ONLY
  qui n'est pas isolé par le simple signe du margin.

Cette expérience cherche des sous-types **interprétables** de RUBBER_ONLY.

## Sources figées

Interne :
- Librosa symmetric H2 : run 37404522178 ;
- Rubber Band H2 : run 37478200631.

Outer :
- Librosa H2 : run 37421205179 ;
- Rubber Band H2 : run 37503022853.

Aucun audio n'est recalculé.

## Population

RUBBER_ONLY uniquement :

- Librosa H2 mean margin <= 0 ;
- Rubber Band H2 mean margin > 0.

Interne : folds 0,1,2,4 ; fold 3 exclu.
Outer : fold 3 déjà exposé historiquement, uniquement pour mesure exploratoire finale.

## Features Rubber Band H2

À partir des cinq margins par vue s in {-2,-1,0,+1,+2} :

- R_mean : moyenne des 5 margins ;
- R_min : minimum ;
- R_std : écart-type ;
- positive_views : nombre de vues avec margin > 0 ;
- pair1_mean : moyenne des margins -1 et +1 ;
- pair2_mean : moyenne des margins -2 et +2 ;
- both_pairs_positive : pair1_mean>0 et pair2_mean>0 ;
- all_shifted_positive : margins -2,-1,+1,+2 tous >0 ;
- center_positive : margin step 0 >0.

Le margin step 0 est la probabilité de référence figée.

## Sous-types candidats

Les seuils quantitatifs sont appris sur FIT uniquement.

1. VOTE4
   - positive_views >= 4.

2. PAIRS
   - pair1_mean > 0 et pair2_mean > 0.

3. ALL_SHIFTED
   - les quatre vues transposées ont margin > 0.

4. STRONG
   - R_mean >= médiane FIT de R_mean sur RUBBER_ONLY.

5. LOW_VAR
   - R_std <= médiane FIT de R_std sur RUBBER_ONLY.

6. STRONG_PAIRS
   - STRONG et PAIRS.

7. STRONG_VOTE4
   - STRONG et VOTE4.

8. PAIRS_LOW_VAR
   - PAIRS et LOW_VAR.

## Validation interne

Leave-one-fold-out sur 0/1/2/4.

Pour chaque fold VAL :

1. calculer les seuils médian STRONG/LOW_VAR sur les 3 folds FIT ;
2. évaluer les 8 sous-types sur FIT ;
3. un sous-type est admissible si :
   - net FIT > 0 ;
   - corrections FIT >= 3 ;
4. choisir le sous-type admissible au meilleur net FIT ;
   départage : moins de régressions, moins d'actions, ordre de la liste ci-dessus ;
5. appliquer inchangé au fold VAL.

Rapporter corrections/régressions/net par fold.

## Sélection outer

Sur les quatre folds internes combinés :

- recalculer les seuils ;
- choisir le meilleur sous-type avec la même règle ;
- appliquer ce sous-type inchangé au RUBBER_ONLY outer.

## Comparaison finale

Mesurer deux politiques outer :

A. RUBBER_ONLY filtré seul ;
B. BOTH + LIBROSA_ONLY + RUBBER_ONLY_filtré.

La politique B permet de voir si le filtre améliore le routing global précédent.

Références :
- Rubber Band H2 outer : +2 ;
- routing précédent BOTH+LIBROSA_ONLY : 0 ;
- consensus mean_engine_margin : +1.

Aucune promotion automatique.
