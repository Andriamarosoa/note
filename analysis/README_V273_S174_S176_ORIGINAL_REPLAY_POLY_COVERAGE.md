# S174–S176 — reprise authentique et limite du GOAT actuel

Branche de recherche : `codex/v273-scope-rescue`. Référence officielle inchangée `freeze_local_combo`. Aucun S80 promu.

## S174 — rejouer les modèles restaurés sans annotations

[CI S174 réussi](https://github.com/Andriamarosoa/note/actions/runs/38206823027).

Sur les **7 493 événements historiques éligibles**, le rejeu via les modèles originaux récupérés en S172 confirme **0 divergence K** contre les vecteurs archivés pour :
- S1 : les **133 affectations outer-fold** des 7 familles de politique et les 49 décisions historiques ;
- S2 : les trois politiques originales pilotées par les choix historiques figés ;
- S3 : les **110** états HistGradientBoosting récupérés et 42 sorties directes ;
- S4 : les **110** états historiques de garde et 56 sorties directes.

La vérification ne lit pas `true_K`, ni MIDI, ne réentraîne rien. Elle reproduit les événements *historiques*, PAS les audios GOAT. Les choix S2 historiques ont été optimisés avec supervision dans la recherche initiale, donc leurs index figés ne constituent pas une règle nouvelle valide de déploiement.

## S175 — véritable réseau original cohérent retrouvé

[CI S175 réussi](https://github.com/Andriamarosoa/note/actions/runs/38206970870).

L'archive source `analysis/evidence/v273-audit-memory/source-archives/model-replay-bundles.zip` contient les **4 modèles Keras originaux `repair_coherent`** (folds 0/1/2/4), soit 12 jeux de poids HDF5 dans chaque état. Tous les poids et rapports ont été vérifiés contre les SHA-256 de l'inventaire source.

Le vrai modèle exige `subset_features` de taille **5×64×21**, un masque de sous-ensembles, une classe baseline, un contexte et des métadonnées keep-head. Aucun de ces tenseurs historiques complets n'a été produit sur GOAT ; le restaurer ne valide donc pas une 12e famille exécutable.

## S176 — vérité des entrées acoustiques et couverture S1–S4

[CI S176 réussi](https://github.com/Andriamarosoa/note/actions/runs/38207176026).

Les **58 noms et leur ordre** correspondent strictement entre les caractéristiques acoustiques historique S1 et le vrai extracteur S97 sur GOAT. Ceci est la première pièce du contrat d'entrée S1 confirmée sur nouvel audio.

Le S18 **adapté** prédit sur les **57 événements** GOAT :
| K | Événements |
|---|---:|
| K0 | 45 |
| K1 | 10 |
| K2 | 2 |
| K3–K6 | 0 |

**Seulement 2 événements** sont au niveau K2/3/4, domaine des boucles historiques S1–S4. Cette mesure ne lit PAS la vérité des annotations, donc ce tableau est une distribution de prédictions et non de K réel.

Les 255 propositions de combinaisons historiques, leurs 9 audits par groupe, `z` de correction-baseline, les 7 logits de risque, et les tenseurs `repair_coherent` 5×64×21 **manquent sur les nouveaux WAV**. Ne pas fabriquer des groupes/risques ni faire passer l'adaptation pour une reproduction source.

## Décision technique

Les familles S1/S2/S3/S4/`historical_repair_coherent` ont désormais leurs **sources / états historiques** reconstitués ou identifiés, mais pas encore de producteur complet sur WAV inédit. Le score de S45 demeure **11/16 producteurs sur nouvel audio** (S35 adapté, 209 Ridge), sans preuve indépendante de correction. Ne pas dépenser de nouvelles boucles répétitives S1–S4 sur les 57 mêmes événements presque tous K0/K1.

**Prochain jalon utile** : constituer d'abord un jeu *audio seul* d'événements polyphoniques diversifiés réellement inédits, sans consulter les annotations, à partir d'une source indépendante comme [IDMT-SMT-GUITAR V2](https://zenodo.org/records/7544110). Recréer les entrées originales (255 groupes, 9 audits, probabilités de facteur, tenseurs cohérents), puis seulement passer de 11/16 à 16/16 et envisager S46 → S69 → S80. Figer l'architecture, les poids et les prédictions AVANT toute ouverture des labels indépendants. Aucun des 57 GOAT déjà exposés ne prouve une amélioration Exact-K indépendante.

La porte de promotion reste fermée : au-delà des contraintes de lineage, un nouvel Exact-K global **et** polyphonique doit dépasser la référence `freeze_local_combo` avec corrections/régressions rapportées de façon indépendante.
