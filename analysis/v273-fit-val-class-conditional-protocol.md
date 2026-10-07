# Audit FIT ↔ VAL conditionné par vrai K — players 00–04

Préenregistré le 7 octobre 2026.

## Périmètre

- GuitarSet players 00, 01, 02, 03, 04 uniquement.
- Folds internes 0, 1, 2, 4 uniquement.
- Fold 3 interdit.
- Player 05 interdit.
- Aucune nouvelle extraction audio : utiliser exclusivement les entrées archivées
  de `analysis/evidence/v273-feature-family-nested/evidence.zip`.
- Aucun nouveau modèle, aucun seuil, aucune correction Exact-K.

## Question

Le changement de proportions K2/K3 entre FIT et VAL ne suffit pas à expliquer
le manque de transfert. Mesurer donc le **changement du signal à vrai K identique**.

Pour chaque fold externe, chaque feature et chaque classe vraie K2/K3 :

1. comparer distribution FIT vs VAL ;
2. rapporter médiane FIT/VAL ;
3. mesurer un AUC de domaine orientation-invariante
   `max(AUC(FIT vs VAL), 1-AUC)` ;
4. mesurer un déplacement robuste de médiane normalisé par l'IQR FIT.

En parallèle, pour chaque feature :

1. apprendre uniquement sur FIT le sens K2 vs K3 de la feature ;
2. calculer l'AUC K2-vs-K3 orientée sur FIT ;
3. appliquer **le même sens gelé** à VAL ;
4. compter les inversions de sens sur VAL et la perte/gain d'AUC.

## Sorties

Classer descriptivement les features selon :

- nombre de folds où le sens K2/K3 s'inverse sur VAL ;
- AUC moyen de domaine à K fixé ;
- variation moyenne d'AUC K2/K3 FIT→VAL.

Aucune feature n'est sélectionnée pour une correction dans ce run. Tout résultat
est diagnostic et doit précéder un éventuel nouveau protocole.
