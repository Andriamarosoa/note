# Protocole — mesure globale outer du Symmetric Pitch TTA H2

Date : 6 octobre 2026.

## Objectif

Mesurer le score Exact-K global du modèle robuste V27.3 après application du
trick H2 qui a obtenu +10 net sur les folds internes 0/1/2/4.

Cette mesure utilise le fold outer 3 déjà exposé historiquement. Elle est donc
**exploratoire** et ne constitue pas un nouveau test aveugle.

## Base globale

Référence outer figée :

- modèle : freeze_local_combo + candidate_hidden1 [42,52,61,64] ;
- outer fold : 3 ;
- prédictions figées : release v273-outer-b-low-eval-37316567815 ;
- score de référence attendu : exact global 83.533%, poly exact 39.817%.

## Population d'application

H2 est appliqué uniquement si, sans regarder le vrai label :

- la ligne appartient au B_low déjà calculé par le pipeline outer ;
- la prédiction de base robuste vaut K3.

Aucun vrai K n'est utilisé pour sélectionner les lignes.

## H2

Pour chaque ligne cible, calculer cinq vues audio :

- -2, -1, 0, +1, +2 demi-tons.

Même méthode de pitch shift et même spectral map COVERED 31x64x3 que les runs
37400666855 et 37404522178. Les entrées temporelles/candidates restent figées.

Pour chaque vue :

margin = P(K2) - P(K3)

Règle fixe :

- si mean(margin sur les 5 vues) > 0 : K3 -> K2 ;
- sinon conserver K3.

Aucun seuil appris, aucun gate, aucune sélection sur fold 3.

## Contrôles

- la vue step=0 doit reproduire les probabilités outer figées avec tolérance
  absolue <= 1e-5 ;
- les indices globaux et vrais K du fichier outer doivent correspondre au bundle ;
- aucune autre prédiction que les K3 B_low ciblées ne peut changer.

## Rapport

Sur les 15,279 lignes outer :

- exact global avant/après ;
- poly exact avant/après ;
- corrections, régressions, net ;
- nombre de lignes B_low K3 ciblées ;
- nombre d'actions H2 ;
- delta par vrai K ;
- taux de réussite parmi les actions.

Aucune promotion automatique.
