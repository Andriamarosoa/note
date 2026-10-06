# Protocole — RUBBER_ONLY avec vue -2 positive

Date : 6 octobre 2026.

## Motivation

Sur l'outer fold 3, les trois corrections RUBBER_ONLY observées avaient toutes
la vue -2 positive, alors que l'unique régression avait la vue -2 négative.

Ce constat outer est **diagnostique uniquement**. La sélection de la règle
doit être faite exclusivement sur les folds internes 0/1/2/4.

## Population

RUBBER_ONLY :
- mean Librosa H2 margin <= 0
- mean Rubber Band H2 margin > 0

## Features utilisées

À partir des cinq margins Rubber Band aux étapes [-2,-1,0,+1,+2] :

- m_m2 = margin(-2)
- m_m1 = margin(-1)
- m_0  = margin(0)
- m_p1 = margin(+1)
- m_p2 = margin(+2)

## Règles candidates figées

1. M2_POS
   - m_m2 > 0

2. M2_M1_POS
   - m_m2 > 0 et m_m1 > 0

3. M2_PAIR_POS
   - m_m2 > 0 et mean(m_m2,m_p2) > 0

4. M2_CENTER_NEG
   - m_m2 > 0 et m_0 <= 0

5. M2_P1_POS
   - m_m2 > 0 et m_p1 > 0

6. M2_OR_M1_STRONG
   - m_m2 > 0 et (m_m2 + m_m1)/2 > 0

Aucun seuil numérique appris à partir de l'outer.

## Validation interne

Leave-one-fold-out sur 0/1/2/4.

Pour chaque fold VAL :
- évaluer les 6 règles sur les 3 folds FIT ;
- règle admissible si net FIT > 0 et corrections FIT >= 3 ;
- choisir meilleur net FIT ;
- départage : moins de régressions, moins d'actions, ordre ci-dessus ;
- appliquer inchangé au fold VAL.

Rapporter :
- corrections / régressions / net par fold ;
- total CV ;
- nombre de folds non négatifs.

## Outer

Après validation :
- choisir la règle sur les 4 folds internes combinés avec le même critère ;
- appliquer cette règle inchangée au RUBBER_ONLY outer ;
- mesurer la règle seule ;
- mesurer Librosa + règle filtrée.

Références :
- Rubber Band H2 outer : +2
- Librosa outer : 0
- consensus mean-engine : +1
- routing précédent : 0

Aucune promotion automatique.
