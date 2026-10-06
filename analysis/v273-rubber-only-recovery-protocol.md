# Protocole — récupération complémentaire du pattern RUBBER_ONLY

Date : 6 octobre 2026.

## Point de départ

La règle actuelle `M2_P1_POS` :
- interne complet : 6 corrections / 3 régressions = +3 ;
- outer fold 3 : 2 corrections / 0 régression = +2.

Elle récupère deux des trois corrections RUBBER_ONLY outer, mais manque le cas
de pattern `11001`.

L'unique régression outer observée avait le pattern `01011`.

Cette expérience reste exploratoire car le fold 3 a déjà été exposé.

## Population

RUBBER_ONLY :
- mean Librosa H2 margin <= 0
- mean Rubber Band H2 margin > 0

## Base figée

BASE = M2_P1_POS :
- margin(-2) > 0
- margin(+1) > 0

## Branches complémentaires candidates

Les branches suivantes ne s'appliquent qu'aux cas non déjà pris par BASE.

1. REC_M2_M1_P2
   - margin(-2) > 0
   - margin(-1) > 0
   - margin(+2) > 0

2. REC_M2_M1_P2_P1NEG
   - REC_M2_M1_P2
   - margin(+1) <= 0

3. REC_M2_M1_P2_CENTERNEG
   - REC_M2_M1_P2
   - margin(0) <= 0

4. REC_M2_M1_P2_P1NEG_CENTERNEG
   - REC_M2_M1_P2
   - margin(+1) <= 0
   - margin(0) <= 0

5. REC_M2_M1
   - margin(-2) > 0
   - margin(-1) > 0

6. REC_M2_P2
   - margin(-2) > 0
   - margin(+2) > 0

Aucun seuil continu n'est appris sur l'outer.

## Validation interne

Leave-one-fold-out sur 0/1/2/4.

Pour chaque VAL :
- BASE reste fixe ;
- sur les 3 folds FIT, évaluer chaque branche complémentaire sur les cas
  RUBBER_ONLY non déjà pris par BASE ;
- branche admissible si net FIT > 0 et corrections FIT >= 2 ;
- choisir meilleur net FIT ;
- départage : moins de régressions, moins d'actions, ordre ci-dessus ;
- appliquer BASE + branche au fold VAL.

Rapporter :
- BASE seul ;
- branche seule ;
- BASE + branche ;
- corrections / régressions / net par fold.

## Sélection outer

Sur les 4 folds internes combinés :
- choisir la branche avec le même critère ;
- appliquer BASE + branche inchangé au fold 3.

Comparer :
- BASE seul ;
- branche seule ;
- BASE + branche ;
- Rubber Band H2 complet.

Aucune promotion automatique.
