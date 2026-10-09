# S43 — AUDIT RÉTROSPECTIF DES CORRECTIONS : verdict et priorités de têtes

**9 octobre 2026, branche `codex/v273-open-k0-k6`.** Audit exécuté : [S43 run 37887837855](https://github.com/Andriamarosoa/note/actions/runs/37887837855). Vérification indépendante des **910 prédictions natives DISTINCTES**, de leurs SHA256 et de chaque correction/régression par vrai K et fold : [run 37888008660](https://github.com/Andriamarosoa/note/actions/runs/37888008660), succès. Archive immuable [v273-series43-research-37888008660](https://github.com/Andriamarosoa/note/releases/tag/v273-series43-research-37888008660).

## Couverture et références

- **1 441 variantes de décision natives**, **910 vecteurs distincts** conservés, y compris les variantes nettement régressives ; **203 variantes anciennes** des registres de mémoire et des boucles ont été recoupées bit à bit sur le corpus natif. Les trois familles d'archives historiques s'ajoutent à **30 séries S9–S41** contenant des sorties comparables.
- Séries S10, S23, S33, S39 : diagnostics, pas de nouvelles prédictions natives comparables ; signalées distinctement, non inventées. Des recherches non archivées et d'autres workflows historiques hors de ce périmètre ne sont pas certifiés complets.
- Les **références** `freeze_local_combo` et `S18` ne sont **pas** de nouvelles têtes : elles figurent dans les tables exhaustives, **pas** dans le classement de propositions nouvelles. `freeze` : 48 454/59 309 global, 2 530/7 385 poly ; `S18` : 49 178/59 309 global, 2 998/7 385 poly. Construction S18 vs freeze : 2 934 corrections / 2 210 régressions, gain net +724.
- Deux fichiers de classements ***séparés*** : `rank_all_vs_freeze.csv` et `rank_actionable_vs_S18.csv`. Jamais additionner des corrections mesurées face à des références différentes. Tous les reportings utilisent des prédictions réelles et le même `global_index`.
- **H9/YourMT3+ exclue**. Aucune entrée YourMT3+, aucun oracle de vérité introduit dans une future tête. Le vrai K intervient uniquement dans les audits rétrospectifs.

## Toutes les principales boucles positives classées selon leurs corrections BRUTES vs S18

La liste détaillée exhaustive, y compris les corrections faibles et les variants négatifs, se trouve dans `best_actionable_per_loop.csv` et `rank_actionable_vs_S18.csv` de l'archive.

| Priorité brute | Source et meilleure variante brute | Corrections | Régressions | Corr. poly K2–K6 | Régr. poly |
|---:|---|---:|---:|---:|---:|
| 1 | Boucles historiques S1 : `acoustic43_cost1.3` | **2 234** | 2 799 | 892 | 1 201 |
| 2 | Mémoire historique : `repair_coherent` | 2 208 | 2 901 | 866 | 1 303 |
| 3 | Audits post-hoc historiques : `source_count90` | 2 207 | 2 774 | 865 | 1 176 |
| 4 | S25 : `Bbefore_A0_raw` | 2 036 | 2 092 | 511 | 1 630 |
| 5 | S24 : `Bfirst_B0_raw` | 1 957 | 1 991 | 515 | 1 545 |
| 6 | S40 : CNN temporel `source_K1to6_argmax` | **1 741** | 1 746 | **1 058** | 1 269 |
| 7 | S19 : A–B–A passe 2 | 1 575 | 1 936 | 798 | 1 199 |
| 8 | S11 : `votes_time_audio_HGB` | 1 143 | 884 | 353 | 609 |
| 9 | S38 : `morph825_ExtraTrees` | 1 139 | 914 | 404 | 905 |
| 10 | S12 : `full_audio_logistic` | 1 114 | 900 | 367 | 577 |
| 11 | S20 : A–B–A passe 1 | 985 | 1 283 | 624 | 987 |
| 12 | S27 : `early_stable_005_raw` | 947 | 1 196 | 568 | 862 |
| 13 | S41 : fusion neuronale | 814 | 688 | 396 | 606 |
| 14 | S21 : `without_B_features_HGB` | 729 | 858 | 409 | 673 |
| 15 | S26 : consensus A, `p3` | 648 | 824 | 392 | 613 |
| 16 | S35 : routeur 18 têtes `lambda1/seuil0` | 570 | 501 | 253 | 499 |
| 17 | S22 : consensus `ABA2_ABA4` | 474 | 508 | 199 | 401 |
| 18 | S36a/S36b : ancien meilleur S35 brut | 382 chacun | 309 chacun | 125 | 309 |
| 19 | S37 : chemins multiples `full_path` | 368 | 296 | 140 | 294 |

*Un grand volume brut n'est pas un score final. Par exemple la meilleure S40 en correction brute régresse autant qu'elle corrige ; la variante S40 au meilleur score GLOBAL n'est pas la même. Ces chiffres sont exploratoires sur les morceaux de développement déjà étudiés.*

## Nouvelle statistique : corrections IRREMPLAÇABLES par les autres archives

Le nombre brut de corrections est trompeur, puisque plusieurs boucles corrigent les mêmes événements. Pour chacune des **33 familles de sources**, l'audit a pris **l'union de toutes ses variantes non-référence**, puis compté les événements corrects uniquement grâce à cette famille, parmi toutes les 33 :

| Source | Tous les cas qu'au moins une variante corrige | Cas corrects exclusifs de cette famille | Exclusifs poly K2–K6 | Exclusifs par vrai K [K0,K1,K2,K3,K4,K5,K6] |
|---|---:|---:|---:|---|
| **S40 CNN temporel** | 1 741 | **223** | **162** | [0,61,78,54,23,5,2] |
| **S19 récurrent A–B–A** | 1 755 | **128** | **71** | [5,52,35,13,19,4,0] |
| **S38 morphologie et arbres** | 1 882 | **111** | **55** | [0,56,15,18,11,10,1] |
| **Boucles régressions historiques S1–S4** | **3 180** | **42** | **42** | [0,0,16,15,11,0,0] |
| S25 B-first appris | 2 705 | 41 | 3 | [16,22,1,1,1,0,0] |
| S24 B-first poids figés | 2 644 | 14 | 6 | [3,5,4,1,1,0,0] |
| S35 routeur multi-têtes | 616 | 6 | 4 | [1,1,2,2,0,0,0] |
| S41 fusion déjà expérimentée | 814 | 0 | 0 | [0,0,0,0,0,0,0] |

**Attention** : 0 exclusif ne veut pas dire qu'une tête ne possède aucune correction ni qu'elle doit être effacée : cela veut dire que d'autres variantes préservées savent *au moins une fois* corriger les mêmes événements.

Le plafond d'union de toutes les sorties archivées est **5 309 anciennes erreurs S18 possédant au moins une bonne proposition**, dont **2 658 vrais événements poly**. Ce plafond est calculé avec la vérité : **ce n'est PAS une performance d'inférence**. L'union de 128 candidats distincts classés atteint 2 991 erreurs de S18 (oracle, non déployable). Toutes les 910 sorties sont préservées.

## Nouvelles têtes/sélections à étudier en priorité

**Ordre expérimental par valeur de corrections exclusives poly**, et non simplement par corrections brutes :

1. **T43-TEMPORAL / S40** : sélectionner conditionnellement les propositions du CNN, avec audit de la présence d'attaque/fondamentale indépendante ; **162 accords poly que nulle autre famille actuellement archivée ne corrige**. Priorité K2 (78), K3 (54), K4 (23) ; ne jamais activer par la vérité K.
2. **T43-RECURRENT / S19** : réinterpréter les trajectoires des passes A–B–A comme une tête pouvant proposer seulement ses corrections sous contrôle acoustique ; **71 exclusifs poly**.
3. **T43-MORPHO / S38** : tête morphologique/expert arbres ; **55 exclusifs poly**, avec K5 (10) et K6 (1), donc ne pas la limiter arbitrairement à K2/K3.
4. **T43-OLD-ACOUSTIC / S1–S4** : reconstituer le signal acoustique des correcteurs historiques et leur caractérisation de redondance/compétition ; **42 exclusifs poly**, répartis K2:16 / K3:15 / K4:11 ; les archives ne sont pas prêtes à être branchées telles quelles sans refaire une provenance de fit OOF.
5. **T43-BFIRST / S25/S24** : spécialistes de l'ordre initial des têtes, pour K0/K1 et certains K2 ; 41/14 exclusifs globaux mais seulement 3/6 poly.
6. Conserver toutes les autres sélections dans le catalogue S43, **y compris celles sans correction exclusive** : leur audit peut servir de caractéristique de fiabilité, de veto ou de tête secondaire dans une nouvelle architecture.

**Statut de réalisation réel :** `candidate_head_registry.json` contient **107 propositions archivées non activées** ; elles ne sont **pas** de nouvelles architectures entraînées. Chaque source est associée à ses prédictions, SHA256, corrections/régressions, folds, K et transitions K courant→K proposé. Les masques conditionnels futurs ne pourront utiliser que des signaux observables à l'inférence, apprendront uniquement sur les autres folds, et devront préserver S18 jusqu'à une validation sur musique réellement inédite.

## Fichiers de preuve

L'archive permanente, horodatée, avec `SHA256SUMS`, contient :
- `all_distinct_native_predictions.npz` — **910 vecteurs de 59 309 décisions natives** et leurs identités SHA256 ;
- `rank_all_vs_S18.csv`, `rank_all_vs_freeze.csv`, `rank_actionable_vs_S18.csv` — tous les résultats classés, avec parents/references non supprimés ;
- `best_per_loop.csv`, `best_actionable_per_loop.csv` — meilleur nombre de corrections pour chaque boucle ;
- `source_exclusive_corrections.csv`, `head_correction_overlap.csv`, `transition_audit_top128.json` — complémentarité, chevauchement et transitions K ;
- `candidate_head_registry.json` — 107 sélections candidates, aucune promue ;
- `source_manifest.json`, `uncovered_series.json`, `excluded_or_incomparable_sources.csv`, `historical_metric_registry.csv` — provenance, défauts de couverture, alias et archives anciennes ;
- `S43-verified/report.json` — vérification indépendante, SHA256 de chacun des 910 vecteurs et comptes par fold et K.

**Aucune correction historique n'est volontairement éliminée**, et le système n'est pas modifié en production. La suite doit entraîner et auditer séparément les cinq priorités ci-dessus, en privilégiant les corrections exclusives et les anciennes bonnes réponses préservées, sans jamais faire dépendre une décision du vrai K.