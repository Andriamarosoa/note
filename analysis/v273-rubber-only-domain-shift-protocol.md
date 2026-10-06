# Protocole — diagnostic du décalage RUBBER_ONLY interne -> outer

Date : 6 octobre 2026.

## Constat de départ

- RUBBER_ONLY interne : 23 actions, 10 corrections, 13 régressions, net -3.
- Le meilleur sous-type robuste VOTE4 : 4 corrections / 1 régression = +3 interne,
  mais 0 action sur l'outer fold 3.
- Rubber Band H2 complet : +2 net sur l'outer fold 3.

Le but ici est **diagnostique** : comprendre quelle forme prennent les quelques
RUBBER_ONLY outer qui créent le gain, et pourquoi ils ne ressemblent pas au
sous-type VOTE4 interne.

## Sources

Interne :
- Librosa H2 run 37404522178
- Rubber Band H2 run 37478200631

Outer :
- Librosa H2 run 37421205179
- Rubber Band H2 run 37503022853

Aucun recalcul audio.

## Population

RUBBER_ONLY :
- mean Librosa H2 margin <= 0
- mean Rubber Band H2 margin > 0

## Descripteurs

Pour les cinq vues Rubber Band (-2,-1,0,+1,+2) :

- sign_pattern : suite binaire de 5 bits (margin > 0)
- positive_views
- mean_margin
- min_margin
- max_margin
- std_margin
- center_margin
- pair1_mean = mean(-1,+1)
- pair2_mean = mean(-2,+2)
- asymmetry_1 = |m(-1)-m(+1)|
- asymmetry_2 = |m(-2)-m(+2)|

Et entre moteurs :

- librosa_mean_margin
- rubber_mean_margin
- engine_gap = R_mean - L_mean

## Analyse interne

Rapporter pour chaque pattern observé :
- rows
- corrections
- regressions
- net
- folds présents

Rapporter aussi positive_views = 1..5.

## Analyse outer

Rapporter chaque ligne RUBBER_ONLY individuellement :
- true K
- corrected/regressed/neutral
- pattern
- tous les descripteurs ci-dessus

Puis comparer chaque pattern outer aux patterns déjà vus en interne.

Aucune règle n'est sélectionnée dans cette expérience.
Aucune promotion automatique.
