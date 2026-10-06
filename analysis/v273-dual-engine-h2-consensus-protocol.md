# Protocole — consensus Librosa + Rubber Band H2

Date : 6 octobre 2026.

## Motivation

Les deux moteurs H2 produisent des comportements différents :

- Librosa H2 interne : 26 corrections / 16 régressions = +10 ;
- Librosa H2 outer : 5 / 5 = 0 ;
- Rubber Band H2 interne : 22 / 20 = +2 ;
- Rubber Band H2 outer : 7 / 5 = +2.

Cela suggère que leurs erreurs ne sont pas parfaitement corrélées.

## Sources figées

Interne :
- Librosa symmetric H2 : run 37404522178 ;
- Rubber Band H2 : run 37478200631.

Outer fold 3 :
- Librosa H2 : run 37421205179 ;
- Rubber Band H2 : run 37503022853.

Aucun audio n'est recalculé.

## Population

Même B_low + base K3.

Interne :
- 488 cas K2/K3 ;
- folds 0,1,2,4 ;
- fold 3 exclu.

Outer :
- 296 cibles B_low + base K3 sur 15,279 lignes ;
- fold 3 déjà exposé historiquement.

## Scores moteur

Pour chaque ligne :

L = mean_{s=-2..2}(P_L(K2)-P_L(K3))
R = mean_{s=-2..2}(P_R(K2)-P_R(K3))

## Règles figées

### 1. strict_consensus
Action K3->K2 seulement si :
L > 0 ET R > 0.

### 2. mean_engine_margin
Action si :
(L + R)/2 > 0.

### 3. positive_agreement_weighted
Action si :
L > 0 ET R > 0 ET min(L,R) / (max(|L|,|R|)+1e-12) >= 0.25.

Le seuil 0.25 est fixé avant lecture des résultats de ce test.

## Sélection

Sur l'interne uniquement :
- calculer les 3 règles sur les folds 0/1/2/4 ;
- choisir la règle au meilleur net total ;
- départage : moins de régressions, moins d'actions, ordre strict_consensus,
  positive_agreement_weighted, mean_engine_margin.

Puis appliquer cette règle inchangée à l'outer fold 3.

## Mesures

- actions ;
- corrections ;
- régressions ;
- net ;
- folds non négatifs ;
- Exact-K global outer ;
- Poly Exact-K outer.

Aucune promotion automatique.
