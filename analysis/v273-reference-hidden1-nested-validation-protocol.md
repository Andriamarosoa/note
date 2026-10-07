# Validation nested confirmatoire de la référence hidden1 [42,52,61,64]

Date: 7 octobre 2026.

## Hypothèse figée

La seule hypothèse testée est la référence actuelle dérivée de V27.3 :

`freeze_local_combo + candidate_hidden1 [42,52,61,64]`.

Aucun autre groupe, neurone, seuil ou feature n'est recherché dans ce protocole.

## Périmètre

- folds autorisés : 0, 1, 2, 4 ;
- fold 3 interdit ;
- players autorisés : 00–04 ;
- player 05 interdit ;
- mêmes données, architecture, seed, epochs, pondération et freeze-local que le
  protocole V27.3 historique.

## Validation nested

Pour chaque fold externe F parmi 0/1/2/4 :

1. F est totalement interdit à la sélection.
2. Sur les trois folds restants, effectuer trois sous-validations :
   - un fold inner-val ;
   - entraînement uniform sur les deux autres folds seulement ;
   - fine-tune freeze_local_combo sur ces mêmes deux folds ;
   - mesurer exactement l'effet du groupe fixe [42,52,61,64] sur inner-val.
3. Le groupe est accepté pour F seulement si, sur les trois inner-val :
   - low_net >= 0 partout ;
   - poly_net >= 0 partout ;
   - global_net > 0 dans au moins 2/3 inner-val ;
   - somme global_net > 0.
4. Si accepté, appliquer le groupe fixe au modèle externe entraîné sur les trois
   folds autres que F et évaluer F une seule fois.
5. Si rejeté, la politique nested conserve freeze_local_combo sur F ; aucune
   alternative n'est cherchée.

Les checkpoints externes existants du run 37271262794 peuvent être réutilisés,
car chacun a été entraîné sans son fold externe respectif. Les sous-modèles inner
doivent être réentraînés dans ce run sur deux folds uniquement.

## Critère confirmatoire pré-déclaré

La référence [42,52,61,64] est dite **confirmée par nested validation** seulement si :

1. elle est acceptée par les sous-validations dans les 4 rotations externes ;
2. sur les 4 folds externes :
   - low_net >= 0 partout ;
   - poly_net >= 0 partout ;
   - global_net > 0 dans au moins 3/4 folds ;
   - somme global_net > 0.

Le résultat fixe du groupe est également rapporté fold par fold, mais aucun
résultat ne sert à modifier le groupe ou le protocole après coup.

Aucun fold 3, aucun player 05, aucune promotion automatique.
