# Protocole — disparition progressive sous pitch shift par demi-ton

Date : 6 octobre 2026.

Cette expérience teste directement l'idée originale : augmenter la hauteur du
signal par pas de **1 demi-ton** jusqu'à ce que la cardinalité prédite change.

## Question

Dans `B_low + prédiction de base K=3`, le nombre de demi-tons nécessaires
avant que le modèle cesse de prédire K=3 distingue-t-il les vrais K2 des vrais
K3 ?

Exemple visé :

`K prédit 3 à +0, +1, +2, ... puis K prédit 2 à +s`.

Le point `s` est la frontière de disparition observée.

## Population et protocole

Pour chaque fold VAL parmi 0,1,2,4 :

- reconstruire la base robuste déjà utilisée :
  `freeze_local_combo + hidden1 [42,52,61,64]` ;
- reconstruire `B_low` sur FIT uniquement avec la procédure historique ;
- population : `B_low & base_prediction == 3` ;
- conserver tous les vrais K dans la comptabilité, mais l'analyse discriminante
  K2/K3 utilise seulement les vrais K2 et K3 ;
- fold 3 strictement interdit.

## Transformation audio

Pour chaque ligne :

- charger l'audio GuitarSet original ;
- extraire une fenêtre locale avec marge autour du `cluster_start_sample` ;
- appliquer `librosa.effects.pitch_shift`, durée conservée ;
- pas : `0, +1, +2, ..., +24` demi-tons ;
- reconstruire exactement la carte spectrale V27.3 `31 x 64 x 3` avec la
  fenêtre `COVERED` ;
- conserver inchangés `candidate_set`, `candidate_mask` et `cluster_stats`
  puisque le pitch shift ne modifie pas les temps d'attaque.

Le test isole donc la sensibilité de la branche spectrale à la transposition.
Ce n'est pas une reconstruction complète des features audio des candidats.

## Contrôle +0

Avant toute conclusion :

- la carte spectrale reconstruite à +0 est quantifiée comme le cache historique ;
- elle doit reproduire la carte cachée avec erreur max <= 1e-3 ;
- la prédiction à +0 doit être identique à la prédiction cachée pour toutes les
  lignes analysées ;
- toutes les lignes de population doivent commencer à K=3.

Si ces contrôles échouent, l'audit est invalide.

## Mesures de trajectoire

Pour chaque ligne :

- `first_non3_step` : premier s où K(s) != 3 ;
- `first_below3_step` : premier s où K(s) < 3 ;
- `first_k2_step` : premier s où K(s) == 2 ;
- valeur 25 si aucun événement jusqu'à +24 ;
- `reentered_k3` : vrai si K=3 réapparaît après une première sortie ;
- `monotone_nonincreasing` : vrai si la séquence K(s) ne remonte jamais ;
- trajectoire complète K(0..24) et probabilités K2/K3.

Aucune monotonie n'est supposée.

## Test discriminant K2/K3

La feature principale préenregistrée est `first_k2_step`.

Deux familles de règles sont autorisées, choisies uniquement sur FIT :

1. `first_k2_step <= t` -> corriger 3→2 ;
2. `first_k2_step >= t` -> corriger 3→2 ;

avec t entier de 1 à 25.

Pour chaque rotation :

- sélectionner sur FIT le seuil/direction avec le meilleur net
  `#K2 corrigés - #K3 régressés` ;
- départage : moins de régressions, moins d'actions, seuil plus petit,
  puis règle `<=` avant `>=` ;
- abstention si le meilleur net FIT <= 0 ;
- appliquer la règle inchangée sur VAL ;
- compter aussi les actions sur autres K.

## Diagnostics

Rapporter séparément :

- distribution `first_k2_step` K2 vs K3 ;
- AUC univarié descriptif ;
- taux de sortie de K3 par demi-ton ;
- taux de réentrée K3 ;
- proportion monotone ;
- matrice de transitions entre pas successifs ;
- résultats par fold et total.

## Limites

Le modèle garde les features candidats temporelles originales et ne remine pas
les propositions après transposition. Le test mesure donc une frontière de
stabilité **spectrale conditionnelle aux mêmes candidats**, ce qui correspond à
l'hypothèse testée mais pas à un rerun complet de toute la chaîne de détection.

Aucune promotion automatique. Aucun fold 3.
