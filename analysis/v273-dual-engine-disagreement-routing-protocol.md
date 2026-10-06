# Protocole — routing par désaccord Librosa / Rubber Band H2

Date : 6 octobre 2026.

## But

Librosa H2 et Rubber Band H2 ne font pas les mêmes erreurs. Cette expérience
teste si le **pattern de désaccord entre moteurs** permet de router les
corrections K3->K2 sans moyenner leurs scores.

## Sources figées

Interne :
- Librosa H2 : run 37404522178 ;
- Rubber Band H2 : run 37478200631.

Outer fold 3 :
- Librosa H2 : run 37421205179 ;
- Rubber Band H2 : run 37503022853.

Aucun audio n'est recalculé.

## Scores

Pour chaque ligne :

L = mean_{s=-2..2}(P_L(K2)-P_L(K3))
R = mean_{s=-2..2}(P_R(K2)-P_R(K3))

## Cellules de désaccord

- BOTH: L>0 et R>0
- RUBBER_ONLY: L<=0 et R>0
- LIBROSA_ONLY: L>0 et R<=0
- NONE: L<=0 et R<=0

Seules les trois premières cellules peuvent produire une action 3->2.

## Sélection interne robuste

Pour chaque fold VAL parmi 0/1/2/4 :

1. utiliser les trois autres folds comme FIT ;
2. calculer pour chaque cellule : corrections, régressions, net ;
3. activer une cellule seulement si :
   - net FIT > 0 ;
   - corrections FIT >= 3 ;
4. appliquer inchangé l'ensemble des cellules activées au fold VAL.

Aucun seuil de margin supplémentaire, aucune feature supplémentaire.

Rapporter également les statistiques descriptives de chaque cellule sur
l'ensemble des quatre folds internes.

## Outer fold 3

Pour la mesure outer :

- recalculer les statistiques des trois cellules sur les quatre folds internes ;
- activer une cellule si net interne total > 0 et corrections >= 4 ;
- appliquer ces cellules telles quelles aux 296 cibles B_low + base K3 du fold 3 ;
- aucun tuning sur fold 3.

## Comparaisons

Références :

- Librosa H2 interne : +10 ; outer : 0 ;
- Rubber Band H2 interne : +2 ; outer : +2 ;
- moyenne moteur interne : +9 ; outer : +1.

## Succès

Le routing est intéressant si :

- politique cross-fold interne net > +10, OU
- outer net > +2 sans perdre la stabilité interne ;
- au moins 3/4 folds internes non négatifs.

Aucune promotion automatique.
