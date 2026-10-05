# Point de reprise — Exact-K

## Dernière étape terminée

Audit du critère de reconstruction et deux corrections contrôlées, le 6 octobre
2026. Base de cette étape : commit
`83dda55a7578443652a7de66e4aba8ee74656357`. Le commit contenant ce fichier est
le checkpoint de reprise. Résultats :
[README](README-reconstruction-criterion-audit.md),
[protocole](v273-reconstruction-criterion-protocol.md),
[preuves](evidence/v273-reconstruction-criterion/).

- Même cohorte de 488 cas, folds 0, 1, 2 et 4. Pool-64 complet dans 267/272 K3
  et 213/216 K2. Annotations exclusivement diagnostiques.
- Sur les 267 K3 couverts, gabarit `1/h²` : le choix libre retrouve 0, 1, 2 ou
  3 notes dans 33, 98, 105 et 31 cas. L'oracle complet a un coût normalisé
  supérieur de 0,03469 en médiane. Dans les groupes K3 dégradé/préservé :
  0,06143 contre 0,01748.
- Contrôle synthétique : 267/267 K3 complets pour les deux pentes, résidu maximal
  `7,11e-15`. Rejeu des sorties réelles : écart zéro. Le solveur n'est pas la
  source de l'écart observé sur les spectres réels.
- Relations non exclusives des 400 composantes K3 non appariées avec `1/h²` :
  157 harmoniques attendus, 132 voisines à 55–150 cents, 23 sous-harmoniques,
  25 fondamentales étrangères, 89 harmoniques étrangers, 14 sous-harmoniques
  étrangers et 83 non classées. Ces proximités ne prouvent pas une origine.
- Pondération fixe log-fréquence `w(f)=65/f` : 146 corrections, 151 régressions,
  213 actions autres K, net −5. Triplets K3 complets 30/272 contre 31 au contrôle.
  Nets FIT −67, −85, −82, −79 ; rejet et abstention dans les quatre folds.
- Réponse exacte Hann, pool-64 et `1/h²` : 125 corrections, 141 régressions,
  186 actions autres K, net −16. Triplets K3 complets 29/272. Nets FIT
  −90, −54, −69, −91 ; rejet et abstention dans les quatre folds.
- Contrôle gaussien pool-64 / `1/h²` reproduit le +2 descriptif précédent :
  137 corrections, 135 régressions. Il reste rejeté par FIT et non promu.
- 29 tests locaux réussis. Rejeu des modèles : erreur de probabilité zéro.
  Les 1 666 spectres sont archivés pour les prochains audits sans redécodage.

## Décision

Conserver la base. Ne promouvoir ni la pondération log-fréquence, ni la réponse
Hann, ni le contrôle à +2 descriptif. La disponibilité des fréquences, le choix
de la pente, la mesure fréquentielle et la forme des pics ont maintenant des
contrôles séparés. Aucun n'établit une correction Exact-K sélectionnable.

## Prochaine étape autorisée

Tester une information qui ne réutilise pas uniquement le même spectre statique :
stabilité temporelle des composantes sur deux fenêtres postérieures du chemin
normal. Préenregistrer les fenêtres et le critère avant l'évaluation. Utiliser
les 1 666 spectres archivés comme contrôle de la première fenêtre, mais redécoder
l'audio vérifié uniquement si une seconde fenêtre absente de l'archive est
nécessaire. Comparer d'abord la stabilité des choix sans annotations, puis
évaluer toute feature/correction avec la même rotation FIT, tous vrais K inclus
et abstention. Les annotations peuvent seulement expliquer après coup.

Cette proposition est une hypothèse à contrôler, pas une cause établie. Ne pas
relancer les grilles de pente, capacité, poids log-fréquence ou forme Hann sans
information nouvelle. Aucun nouvel entraînement neuronal complet.

## Commits, runs et reprise fiable

- Critère de reconstruction et gardes statiques :
  `ff48f26757203b86161bcb1272b1ff73c96fec31`. CI
  [`37371872093`](https://github.com/Andriamarosoa/note/actions/runs/37371872093)
  déclenché le 6 octobre 2026 et en attente au moment de ce checkpoint.
- Point de départ de cette étape : `83dda55a7578443652a7de66e4aba8ee74656357`.
- Classement/capacité : `45265a0ad4b3faa950121106bd8d1fcf3e08499c`.
  CI [`37367957693`](https://github.com/Andriamarosoa/note/actions/runs/37367957693)
  terminé avec succès le 5 octobre 2026 UTC.
- Gabarits/pentes : `aad8dfb83c96cf976fc7a46c3347853b514b58f2`.
  Son run `37366044960` a été annulé avant toute étape, sans log de calcul.
- Audit acoustique/tri stable : `0bc2c59226a3d87a44e085febd1e4d36b885d4fe`,
  CI `37362400113` réussi.
- Exports figés : run `37356100423`, commit `cd5ed339`.
- Vérifier le HEAD et les nouveaux runs de `codex/v273-failure-clustering` avant
  toute reprise. Ne pas dupliquer un run actif. Extraire les archives vérifiées ;
  ne pas supposer que les dossiers temporaires existent encore.

## Contraintes persistantes

Folds 0, 1, 2, 4 uniquement ; fold 3 exclu. Base, B_low et populations figés.
Pas de promotion sur un résultat VAL descriptif, un gain supposé ou une
régression. Les résultats du chemin résiduel normal ne démontrent rien pour le
chemin compressé. Publier uniquement sur la branche de recherche, préserver les
travaux concurrents et ne pas fusionner automatiquement. L'automatisation reste
utile pour l'audit temporel ; aucune correction sélectionnable ni blocage
terminal d'accès ou de données n'est établi.
