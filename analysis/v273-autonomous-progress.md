# Point de reprise — Exact-K

## Périmètre joueurs confirmé

- **Players autorisés : 00–04 uniquement. Player 05 reste exclu.**
- Le garde-fou est déjà codé et testé dans `causal_note.guitarset`.
- Les runs historiques player 05 sont conservés pour traçabilité mais ne
  participent plus à aucune sélection, validation ou métrique de référence.
- Politique détaillée : [v273-player-scope-policy.md](v273-player-scope-policy.md).

## Dernière étape : sélection des familles sur FIT exclusivement

Audit du **7 octobre 2026**, depuis `a39dfbb8ff357d2c27b3909c28d2f3fe456512cb`.
[README](README-feature-family-nested.md),
[protocole](v273-feature-family-nested-protocol.md),
[preuves](evidence/v273-feature-family-nested/).

- Le run interne `37547761255` est réussi. Son +20 (53 corrections, 33 régressions)
  résulte d'un seuil choisi sur les VAL réunis. Les comptes sont reproduits ;
  ils restent un bilan de sélection exploratoire.
- Défaut corrigé dans le nouvel audit : choix de famille/seuil sur trois rotations
  internes de FIT ; VAL ne participe jamais au choix. Le protocole et le code
  ont été publiés avant évaluation dans `e064c0ce9e7f9791d4f44658ccf4db191679854a`.
- Mêmes exports du run `37356100423`, base, B_low, masques, FIT/VAL : 1 666 lignes,
  122 pistes, 845 cas VAL. Les résidus historiques sont conservés par apparition.
  Huit features supplémentaires réextraites sans annotation ; aucun bundle global
  chargé, horodatages issus des exports, folds vérifiés avant décodage.
- Politique principale à seuil 0,50 : tous les nets FIT des quatre familles
  sont négatifs. **Abstention dans les quatre folds**, zéro action, net VAL 0.
- Grille robuste secondaire choisie sur FIT : 1 correction, 1 régression,
  3 actions sur d'autres K ; **net VAL 0**. Choix : abstention fold 0 ; géométrie
  @0,75 fold 1 ; base @0,63 fold 2 ; abstention fold 4.
- Témoin robuste à deux résidus : 1 correction, 1 régression, 2 autres K ; net 0.
- Contrôles fixes à 0,50 reproduits dans chaque fold : base −17, géométrie −31,
  attaque +11, géométrie+attaque +1. Ce sont des descriptions VAL, pas des choix
  validés sur FIT. Le −15 historique correspondait à la correction du tri stable.
- Référence reproduite à `3,33e-16` près. **5 tests réussis**, 16 modèles finaux
  et 48 internes rejoués sans audio ni ajustement : erreur de probabilité zéro,
  choix et comptes identiques.
- Aucun résultat du fold 3 ni de player 05 chargé pour cette étude. Les runs
  externes présents sur la branche ne sont pas utilisés pour sélectionner.
- Décision : **conserver la référence**. La correction du protocole est vérifiée,
  mais aucun gain Exact-K sélectionnable n'est démontré.

## Prochaine étape

Ne pas refaire les grilles, gabarits, fenêtres ou features déjà rejetés sans
information nouvelle. Avant une autre variante, séparer le changement de
proportions K2/K3 FIT/VAL d'un changement du signal acoustique à vrai K identique.
Réutiliser `inputs.npz` et `extra-features.npz` archivés : pas de nouvel audio
nécessaire. Préenregistrer les comparaisons, conserver les quatre folds, traiter
les différences comme diagnostics et non causes uniques.

Observation descriptive après gel des prédictions : part K2 parmi K2/K3 sur
FIT puis VAL : fold 0 34,69 % / 47,86 % ; fold 1 26,02 % / 43,52 % ; fold 2
30,32 % / 28,00 % ; fold 4 35,31 % / 49,09 %. Le fold 2 est un contre-exemple
à une explication universelle par ce seul changement de proportions. Le routage
et le réseau amont restent figés ; aucune nouvelle variante n'est sélectionnée
à partir de cette observation.

## Commits, CI et reprise

- Protocole/code imbriqué : `e064c0ce9e7f9791d4f44658ccf4db191679854a`.
- Preuves et CI dédiée : `4ebe66504842fed2c8bad74c7f15a88e29e1f695`.
  Run `37570350189` : tests réussis, rejeu scientifique passé, puis échec de
  l'égalité globale du rapport. Logs inspectés, aucun artefact.
  Cause reproduite localement avec BLAS Haswell : seul `max_probability_error`
  passe de 0 à `3,33e-16`, toutes les décisions et empreintes identiques.
  La comparaison finale applique désormais la tolérance déjà prévue `<1e-12`
  pour ce champ et conserve l'égalité stricte du reste.
- Correction CI : `3d22697f5efe55ad14770ece9708396b3d27f2b4`.
  Run [37570591433](https://github.com/Andriamarosoa/note/actions/runs/37570591433)
  vérifié **completed / success** le 7 octobre 2026, job `112628068518`.
  Cinq tests réussis, 64 modèles rejoués, écart CI `3,33e-16` ; choix et comptes
  strictement identiques. Ce checkpoint documentaire clôt l'étape.
- Lire d'abord le HEAD et les runs ; ne pas dupliquer un run actif. La branche a
  reçu des travaux concurrents, préservés intégralement.
- Rejeu courant : vérifier `analysis/evidence/v273-feature-family-nested/checksums.json`,
  extraire `evidence.zip`, puis exécuter la CI dédiée ou les commandes du README.
  L'archive contient les 8 fichiers nécessaires, dont le rapport source.
- Audit temporel : `699b7ef6a19bbc2c0043ba2596f49f8fe2acb7e7`, CI
  [37375379888](https://github.com/Andriamarosoa/note/actions/runs/37375379888) réussi.
  Critère conjoint net −32, feature de persistance −21, tous deux rejetés sur FIT.
- Gabarits statiques : `ff48f26757203b861bcb1272b1ff73c96fec31`, CI `37371872093`
  annulée avant toute étape ; ne pas annoncer ce run comme réussi. Ses rejeux ont
  ensuite été inclus dans la CI temporelle réussie.
- Classement/capacité : `45265a0ad4b3faa950121106bd8d1fcf3e08499c`, CI `37367957693`
  réussie. Audit initial/tri stable : `0bc2c59226a3d87a44e085febd1e4d36b885d4fe`,
  CI `37362400113` réussie.
- L'ancien travail local `onset-support` n'a pas atteint un commit publié et ses
  fichiers temporaires ont disparu. Ne pas annoncer ses chiffres comme preuves
  archivées. L'étude `v273-attack-novelty` ajoutée entre-temps à la branche évite
  de dupliquer cette question.

## Contraintes persistantes

Folds 0, 1, 2, 4 seulement ; fold 3 exclu de toute nouvelle analyse. Annotations
uniquement diagnostiques, aucun nouveau réseau complet, pas de promotion depuis
VAL descriptif. Le chemin normal ne prouve rien sur le chemin compressé.
Publier uniquement sur la branche de recherche ; aucune fusion automatique.
La poursuite reste autorisée : aucun gain Exact-K vérifié ni blocage terminal
n'est établi. Ne pas confondre la correction du protocole avec un gain du modèle.
