# Point de reprise — Exact-K

## Périmètre joueurs confirmé

- **Players autorisés : 00–04 uniquement. Player 05 reste exclu.**
- Le garde-fou est déjà codé et testé dans `causal_note.guitarset`.
- Les runs historiques player 05 sont conservés pour traçabilité mais ne
  participent plus à aucune sélection, validation ou métrique de référence.
- Politique détaillée : [v273-player-scope-policy.md](v273-player-scope-policy.md).

## Verdict confirmatoire de la référence hidden1 historique

Run [37572714248](https://github.com/Andriamarosoa/note/actions/runs/37572714248)
**completed / success** le 7 octobre 2026 ; release vérifiée
[v273-reference-hidden1-nested-37572714248](https://github.com/Andriamarosoa/note/releases/tag/v273-reference-hidden1-nested-37572714248).
Le run respecte le protocole préenregistré : folds 0/1/2/4 uniquement,
players 00–04 uniquement, groupe fixe `[42,52,61,64]`, aucune recherche de
groupe alternatif et aucune promotion automatique.

Résultat mesuré : le groupe fixe améliore descriptivement les quatre folds
externes (`+2`, `+8`, `+14`, `+17`), soit **+41 exacts** au total, dont
**+27 low** et **+14 poly**. Il satisfait donc l'ancienne règle externe.
Cependant, il échoue à la règle nested pré-déclarée dans **4 rotations sur 4** :

| fold externe | accepté en inner | min global | min low | min poly | somme global |
|---:|:---:|---:|---:|---:|---:|
| 0 | non | -14 | -11 | -3 | +2 |
| 1 | non | -4 | -2 | -6 | +10 |
| 2 | non | -12 | -11 | -7 | -10 |
| 4 | non | -4 | +1 | -5 | +27 |

La politique nested s'abstient donc sur les quatre folds et produit un net
sélectionné de **0**. Décision : **`candidate_hidden1 [42,52,61,64]` n'est pas
confirmé comme référence robuste**. La référence conservée est le dernier niveau
antérieur validé, **`freeze_local_combo`**. Le `+41` reste un résultat descriptif
du groupe fixe et ne justifie aucune promotion.

Exact-K agrégé de la référence conservée sur les mêmes folds externes :

| vrai K | exacts / cas | exact |
|---:|---:|---:|
| 0 (silence) | 38 035 / 39 652 | 95,9220 % |
| 1 | 7 889 / 12 272 | 64,2846 % |
| 2 | 1 183 / 3 445 | 34,3396 % |
| 3 | 915 / 2 289 | 39,9738 % |
| 4 | 417 / 1 207 | 34,5485 % |
| 5 | 15 / 355 | 4,2254 % |
| 6 | 0 / 89 | 0,0000 % |

Exact global : **48 454 / 59 309 = 81,6976 %** ; exact polyphonique K2–K6 :
**2 530 / 7 385 = 34,2586 %**.

Le run de confirmation pairwise dédupliqué
[37577749778](https://github.com/Andriamarosoa/note/actions/runs/37577749778),
lancé avant observation du premier verdict, est lui aussi **completed / success**.
Ses six entraînements et son job de synthèse sont réussis ; la
[release](https://github.com/Andriamarosoa/note/releases/tag/v273-reference-hidden1-pairwise-37577749778)
contient les rapports et checkpoints. Il reproduit la décision confirmatoire :
**0/4 rotation acceptée, net de politique nested 0, verdict FAIL**. Les nombres
externes figés sont identiques (`+41` global, `+27` low, `+14` poly).

Audit de reproductibilité : quatre paires d'entraînement sur six reproduisent
bit à bit le checkpoint uniform du premier run. Les paires `(0,2)` et `(1,2)`
ont des poids et scores inner différents malgré des ordres d'époques identiques.
Il ne faut donc pas présenter les valeurs inner comme bit-identiques entre
runners. Cette variation ne change toutefois aucune décision : les quatre
rotations sont rejetées dans chacun des deux runs indépendants. Détails et
empreintes : [README de confirmation](README-v273-reference-hidden1-confirmation.md).

La branche conditionnelle de re-sélection nested n'est pas ouverte, puisque le
groupe fixe a échoué. Aucun retry n'est nécessaire pour trancher la référence.

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


## Audit FIT↔VAL à vrai K identique — players 00–04 uniquement

Run [37571738846](https://github.com/Andriamarosoa/note/actions/runs/37571738846)
**completed / success**. Player 05 et fold 3 explicitement exclus ; aucun audio
rechargé, aucune sélection de modèle ni de seuil.

L'audit sépare le changement de proportions K2/K3 d'un changement du signal
à classe vraie fixée. Résultat principal :

- `geom_harmonic_relation_min_error` : **4/4 inversions de sens**, AUC classe
  moyenne FIT 0,513 → VAL 0,455 ;
- `weak_unique_post1_norm` : 2 inversions, 0,539 → 0,472 ;
- `geom_span_cents` : 2 inversions, 0,535 → 0,500 ;
- `amp3_over_amp2` : 2 inversions, 0,539 → 0,499 ;
- `best_pair_residual_ratio` : **0 inversion**, 0,575 → 0,558 ;
- `nov_onset_contrast_norm_median` : **0 inversion**, 0,540 → 0,601 ;
- `best_triplet_residual_ratio` : 1 inversion, 0,559 → 0,562 ;
- `joint_unique_x_post_median` : 1 inversion, 0,518 → 0,565.

Le déplacement de domaine à K fixé reste modéré en moyenne (AUC domaine
environ 0,51–0,56), mais plusieurs features changent néanmoins de relation
K2/K3 selon le fold. Le simple changement de proportions K2/K3 n'explique donc
pas seul l'absence de gain nested.

Décision : **aucune feature promue depuis cet audit**. La référence nested reste
inchangée, net sélectionné 0. Toute nouvelle variante doit sélectionner sa
stabilité exclusivement sur FIT avant d'être évaluée sur VAL.


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
