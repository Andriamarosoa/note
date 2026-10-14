# S502–S504 — Sélection neuronale de têtes pour la détection des **notes**, des attaques et leur Exact-K

**14 octobre 2026. Statut : architecture construite et audits diagnostiques. NE PAS PROMOUVOIR.** Cette série s'appuie sur l'architecture déjà présente dans `scripts/learn_v273_neural_head_selector.py`, en particulier `ClassConditionalAuditSelector`, `head_mask` et `head_k_mask` (poids d'attention calculés par contexte, pour chaque K0..K6 séparément).

## Pourquoi adapter l'architecture Exact-K aux notes ?

Les travaux S496–S501 ont montré que le graphe et K s'améliorent conjointement sur IDMT mais qu'une sélection rigide par K peut **supprimer sept accords polyphoniques auparavant corrects** sur MAESTRO. Une tête de confiance n'est pas utile sur tous les instruments, K et situations temporelles. Des **sélections conditionnelles apprises** par contexte sont donc préférables à un ordre de filtres statiques ou au seul `K = nombre de pics`.

Architecture cible :

```text
Signal audio → événements musicaux canoniques (S498)
            → notes candidates, onsets, offsets, K_new, K_active,
               candidats natifs S80/V88 comme preuves, harmoniques
            → BANQUE DE TÊTES [N événements, 13 têtes]
              H0..H12: propositions de hauteurs et actions musicales
            → Sélection conditionnelle par K et par action :
              head_mask [N, 13] (disponibilité)
              head_k_mask [N, 13, 7] (compatibilité K0–K6)
              audit OOF par source, vrai K, action, fixes/régressions
              attention neuronale avec KEEP/veto/abstention
            → Consensus musical / propositions de modifications
            → Exact-K + ensembles des notes + onset/offset cohérents
            → Veto appris / maintien historique si risque excessif
```

Le graphe canonique empêche de créer plusieurs sorties pour un même groupe d'attaque acoustique, mais **ne supprime jamais une réattaque réelle de la même hauteur** simplement parce qu'une note précédente résonne encore.

## Les 13 têtes opérationnellement adaptables, pas encore entraînées

| Tête | Proposition et mission | Règle d'activation observable |
|---|---|---|
| H0 | KEEP : note-set BP `K_new` original | Toujours fallback |
| H1 | Note-set encore actif `K_active` | Pitches actifs observés |
| H2 | Prior de comptage S80/V88 sur `K_new` | Prior natif présent |
| H3 | Prior sur `K_active` | Prior natif présent |
| H4 | Top hauteurs vraies par confiance sous K nouveau | Pitches disponibles |
| H5 | Top hauteurs sous K nouveau avec pénalité harmonique douce | Pitches disponibles |
| H6 | Top hauteurs sous K actif par confiance | Pitches disponibles |
| H7 | Top hauteurs actives avec pénalité harmonique douce | Pitches disponibles |
| H8 | MERGE : détecter doublons entre candidats natifs d'une attaque musicale | Deux preuves natives ou plus |
| H9 | REATTACK : préserver une nouvelle attaque de hauteur déjà soutenue | Réattaque acoustique observée |
| H10 | KEEP : protéger les accords polyphoniques suffisamment confiants | K_new≥2 + évidence de confiance |
| H11 | Réconcilier désaccords K_new/K_active/K source | Désaccord observable (pas vrai K) |
| H12 | ABSTAIN : risque de source ou de fiabilité insuffisante | Prior manquant ou confiance faible |

**Important :** toutes les têtes ne s'ouvrent pas à tous les K. Leur `head_k_mask` interdit des classes non compatibles avec leurs propositions ; les têtes de protection ou d'abstention ne sont pas forcées de prédire une classe. La sélection finale doit être **apprise**, pas un score fixe défini à la main.

Actions prévues (au niveau événement et note) : `KEEP`, `ADD_BIRTH`, `REMOVE_FALSE_NOTE`, `CHANGE_PITCH`, `MERGE_DUPLICATE`, `REATTACK`, `TRANSFER_EVENT`, `REVISE_K`, `ABSTAIN`. Le S502 actuel produit les contrats et propositions ; **il n'exécute pas ces actions sur S80**. Les actions de fusion et réattaque restent des indications structurelles à raccorder à un décodeur de graphes.

## Implémentation et premières preuves

### S502 : vraie banque issue des prédictions source originales

[GitHub Actions S502 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38594382211). Scripts : `scripts/s502_note_birth_head_selection_adapters.py`.

| Têtes structurellement disponibles | IDMT guitare, 60 événements canoniques | MAESTRO piano, 234 événements |
|---|---:|---:|
| H8 — indices de doublons natifs | 42 | 214 |
| H9 — réattaques | 1 | 13 |
| H10 — protection polyphonique | 13 | 10 |
| H11 — désaccords de K | 28 | 216 |
| H12 — risque/abstention | 33 | 229 |

Les têtes produisent leurs ensembles de notes observés, masques `[événement,tête,K]`, type, action, compte proposé et contexte acoustique sur les vrais fichiers S498/S500 **sans lire XML ou MIDI**. Les spécialistes H1–H12 sont désactivés pour tout chemin de production tant que les audits OOF d'entraînement ne sont pas fournis.

### S503 : audit individuel par vrai K ET par nom des notes

[GitHub Actions S503 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38594677574). Script : `scripts/s503_individual_note_head_regression_by_K_audit.py`.

**Audit rétrospectif sur 45 vraies attaques IDMT et 322 vraies attaques MAESTRO** : corrections, régressions, transformations erronée→erronée, métrique d'identité exacte des notes et métrique Exact-K pour chaque tête, chaque classe réelle K0–K6 et chaque K source. Le vrai K est utilisé **seulement pour l'audit a posteriori**, jamais pour ouvrir une sélection en inférence.

Sur IDMT (référence brute note-set BP actif), la sélection H6/H7 améliore les ensembles complets **27/45→29/45** : 3 corrections, 1 régression. Sur piano, H4/H5 modifie 25 attaques, **7 corrections/7 régressions** des ensembles complets, avec régressions poly déjà établies dans S501. La tête H1 qui remplace les nouvelles notes piano par des notes **actives** cause **115 régressions** des ensembles complets contre seulement 3 corrections — une preuve directe qu'une sélection valable dans un domaine/cible peut être destructrice ailleurs.

**Ne pas interpréter H8/H9 comme un correcteur formé** : ils fournissent pour l'instant des signaux/contextes d'action et des notes sources ; leur véritable déduplication/REATTACK devra être exécutée et auditée séparément avant un score de corrections crédible. Les métriques S503 sont des comparaisons de proposition de pitch-set, pas un succès de ces actions.

Ces mesures **ne sont pas des audits OOF d'entraînement** et ne peuvent pas être recyclées comme des features d'entraînement sur ces mêmes attaques déjà vues.

### S504 : raccord au réseau neuronal historique

`scripts/s504_note_heads_to_existing_neural_Kselector_contract.py` convertit les propositions en tenseurs compatibles avec l'interface de `ClassConditionalAuditSelector` :
- `head_logits [N,13,8]` : K0–K6 et KEEP ;
- `head_k_mask [N,13,7]`, `head_mask [N,13]`, `head_type [N,13,4]` ;
- `context [N,16]`, `baseline [N,7]` ;
- vecteur d'audit attendu `[N,13,21]` **absent par conception** tant qu'aucune validation OOF propre ne l'a produit.

**Ce n'est pas un nouveau réseau entraîné.** L'inférence et les modifications de S80 sont interdites tant que les poids, métriques de risques, audit OOF et vraie généralisation inter-instruments ne sont pas validés. L'adaptateur prouve la possibilité d'un raccord architectural, pas la performance future.

## Pour aboutir à une sélection réellement neuronale

1. Construire un corpus d'entraînement distinct des 5 WAV IDMT et 3 MAESTRO déjà utilisés, contenant véritables onsets/offsets, hauteurs et **négatifs d'attaques**, avec partitions strictes par morceau/instrument/joueur. Une absence de correspondance XML ne constitue pas un négatif.
2. Exécuter sur chaque fold d'entraînement les 13 producteurs de propositions **hors-fold** et auditer chaque action par K, instrument et configuration acoustique ; stocker fixes/régressions, et ne jamais utiliser le fold d'évaluation pour fabriquer ses propres poids/audits.
3. Entraîner `ClassConditionalAuditSelector` ou une extension équivalente avec une tête de confiance pour conserver/ajouter/supprimer/fusionner une note et des contraintes temporelles. La sélection doit considérer **les interactions non additives entre les têtes**, pas seulement des votes indépendants.
4. Faire un repassage effectif des nouvelles notes sélectionnées dans le graphe, une fois par événement musical, avec traçabilité de l'ordre et veto de duplication.
5. Tester sur de nouveaux sons, métriques séparées : précision/rappel des attaques, faux positifs/négatifs, Exact-K, vrais ensembles de notes, onset/offset, stabilité Live et **toutes les régressions**. Bloquer promotion si le contrat n'est pas satisfait.

**Aucune promotion de S80 ni changement de `freeze_local_combo`, de Live, des offsets ou des prédictions officielles.**
