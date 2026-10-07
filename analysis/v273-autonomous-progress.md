# Point de reprise — Exact-K

## Reprise du 7 octobre : sélection FIT des familles de features

HEAD vérifié : `a39dfbb8ff357d2c27b3909c28d2f3fe456512cb`. La branche a reçu
d'autres travaux ; aucune modification concurrente n'est écrasée. Aucun run
en cours ou en attente lors de cette vérification.

Le run interne `37547761255` est réussi. Son correcteur enrichi annonce
53 corrections / 33 régressions (+20), mais `robust_threshold` choisit le seuil
sur les VAL réunis : c'est un bilan de sélection exploratoire. Aucun résultat
du fold 3 ni de player 05 n'est chargé pour cette reprise.

[Protocole de sélection imbriquée](v273-feature-family-nested-protocol.md),
écrit avant le nouveau calcul : mêmes exports, base, routage et populations ;
rotations internes dans FIT, seuil principal 0,50, abstention. La grille robuste
existante est un contrôle secondaire sélectionné sur FIT exclusivement.
Quatre familles figées ; aucun nouvel entraînement neuronal.

Les cinq nouveaux tests passent. Les quatre exports sources ont été récupérés
et vérifiés. L'extraction des 1 666 lignes utilise uniquement leurs horodatages,
sans charger le bundle global. À terminer : reproduction des quatre familles,
évaluation imbriquée, rejeu des modèles figés, publication des résultats et CI.

L'audit local « onset-support » commencé dans la reprise précédente n'a pas
atteint un commit publié ; ses fichiers temporaires ne sont plus disponibles.
Ses chiffres ne doivent pas être présentés comme de nouvelles preuves archivées.
Entre-temps, l'étude `v273-attack-novelty` a été ajoutée à la branche : ne pas
dupliquer ce diagnostic. Le résultat ci-dessous reste le dernier checkpoint
complet de cette série d'audits.

## Dernière étape terminée

Audit de stabilité temporelle, le 6 octobre 2026. Point de départ :
`d9f75cb23e14204615bf5a374629ba1f06c720f9`. Le commit contenant ce fichier
est le checkpoint de cette étape.

[README](README-temporal-stability-audit.md),
[protocole initial](v273-temporal-stability-protocol.md),
[protocole de la feature directe](v273-temporal-persistence-feature-protocol.md),
[preuves](evidence/v273-temporal-stability/).

- Deux fenêtres post successives de 2 048 échantillons, avec le même pré.
  Surcoût futur 46,44 ms, total 92,88 ms. Pool-64 produit par la première fenêtre,
  saillance et gabarit gaussien `1/h²` figés ; aucune annotation en inférence.
- Même 1 666 lignes, 122 enregistrements, 845 lignes VAL, 488 cas diagnostiques.
  Folds 0, 1, 2, 4 exclusivement. Première fenêtre : spectres et fréquences
  identiques à l'archive ; écart numérique résiduel maximal `6,28e-16`.
- Sur K3, 313/405 composantes attendues du premier triplet persistent, contre
  165/411 non appariées. Cette association ne discrimine pas correctement K2/K3 :
  moyenne de fréquences persistantes 1,884 pour K2, 1,757 pour K3 ; AUC brute
  « davantage de persistance => K3 » 0,4618. Aucun seuil sélectionné.
- Critère conjoint : 135 corrections, 167 régressions, 183 actions autres K,
  net VAL **−32**. Nets FIT **−81, −74, −90, −95** ; abstention partout.
  Triplets complets 33/272 contre 31 ; couples complets 51/216 contre 47.
- Feature directe de persistance ajoutée aux deux résidus initiaux :
  124 corrections, 145 régressions, 191 actions autres K, net VAL **−21**.
  Nets FIT **−70, −62, −78, −78** ; abstention partout.
- Contrôle reproduit exactement 137 corrections, 135 régressions, net descriptif
  +2 ; il reste rejeté sur FIT. Aucune variante promue.
- Toutes les notes attendues des 488 cas gardent >=10 % de couverture Hann dans
  la seconde fenêtre ; des notes étrangères y sont actives dans 126 K3 et 128 K2.
  Nouvelles attaques étrangères dans cette fenêtre : 15 K3 et 16 K2. Ces comptes
  de contexte ne prouvent pas une cause de l'échec.
- 34 tests ciblés au total. Rejeu des 1 666 extractions, 488 traces, 16 modèles
  finaux et 48 internes ; aucune relecture audio ni aucun réajustement, erreur de
  probabilité zéro. Les spectres de la seconde fenêtre sont archivés.

## Décision et prochaine étape

Conserver la base et rejeter les deux variantes temporelles. La persistance de
fréquences sélectionnées n'est pas une correction Exact-K utile dans ces tests.
Ne pas répéter les essais de pente, capacité, forme Hann, pondération log,
critère temporel moyen ou feature de persistance sans information nouvelle.

Avant une autre correction, auditer la différence entre une nouvelle attaque
et une composante déjà présente : support des fondamentales annotées dans les
puissances pré, post-1, post-2, puis après la soustraction. Relire l'audit acoustique
initial, réutiliser ses mesures pré/post-1 et ne compléter que les mesures
manquantes. Préenregistrer ce contrôle avant calcul ; pas de nouvelle grille de
seuils ou de durées. Ne proposer un bras d'inférence que si un défaut précis et
un contraste contrôlé sont mesurés. Annotations exclusivement diagnostiques.

## Commits, runs et reprise fiable

- Audit temporel : `699b7ef6a19bbc2c0043ba2596f49f8fe2acb7e7`.
  CI [`37375379888`](https://github.com/Andriamarosoa/note/actions/runs/37375379888)
  vérifié réussi le 6 octobre 2026. Aucun run dupliqué.
- Critère de reconstruction et gardes statiques :
  `ff48f26757203b86161bcb1272b1ff73c96fec31`, checkpoint documentaire `d9f75cb`.
  CI [`37371872093`](https://github.com/Andriamarosoa/note/actions/runs/37371872093)
  vérifié le 6 octobre : run `failure`, job `111970757347` `cancelled`, aucune
  étape exécutée, logs absents (404), aucun artefact. Cause non déterminée ;
  aucune assertion scientifique n'a été exécutée. Ne pas le présenter comme réussi.
- Classement/capacité : `45265a0ad4b3faa950121106bd8d1fcf3e08499c`,
  CI `37367957693` réussi.
- Gabarits/pentes : `aad8dfb83c96cf976fc7a46c3347853b514b58f2`,
  run `37366044960` signalé `failure`, job annulé avant toute étape.
- Audit acoustique/tri stable : `0bc2c59226a3d87a44e085febd1e4d36b885d4fe`,
  CI `37362400113` réussi.
- Exports figés : run `37356100423`, commit `cd5ed339`.
- Pour le nouveau rejeu, extraire les archives vérifiées `v273-temporal-stability`,
  `v273-reconstruction-criterion` et `v273-residual-acoustics` ; suivre le README.
  Ne pas supposer que les fichiers temporaires sont encore présents.
- Vérifier le HEAD et les nouveaux runs avant toute reprise. Ne pas dupliquer
  un run actif et préserver toute modification concurrente.

## Contraintes persistantes

Folds 0, 1, 2, 4 uniquement ; fold 3 exclu. Base, B_low et populations figés.
Pas de nouveau réseau complet, pas de promotion à partir de VAL descriptif.
Les résultats normaux ne démontrent aucune cause ou amélioration du chemin
compressé. Publier uniquement sur la branche de recherche, sans fusion automatique.
La poursuite reste autorisée : aucune correction sélectionnable ni blocage
terminal de données ou d'accès n'est établi ; les contrôles CI sont suivis
séparément des résultats locaux.
