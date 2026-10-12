# S212–S216 — Raccordement des 16 producteurs S45 d'origine et audit indépendant S46

**Date** : 12 octobre 2026. **Branche d'exploration** : `codex/v273-scope-rescue`. **Officiel inchangé** : `freeze_local_combo`; **S80 n'est PAS promu**.

## Ce qui est réellement exécuté

| Phase | Run CI réussi | Preuve |
|---|---|---|
| S212 | [38241942489](https://github.com/Andriamarosoa/note/actions/runs/38241942489) | S11, S12, S38, S40, S41 ; vrais poids et entrées S5/S9/S18 ; 19 branches, 57 événements |
| S213 | [38242384204](https://github.com/Andriamarosoa/note/actions/runs/38242384204) | Rejeu de ces cinq producteurs dans les 304 Ridge S45 et 193 politiques S46 ; 1 528 cellules de décision modifiées par rapport à S211 |
| S214 | [38242790294](https://github.com/Andriamarosoa/note/actions/runs/38242790294) | S19, S20, S24, S25, S27 ; 19 vraies branches récurrentes S20/S19/S25, sans S135 ni PR16 utilisés comme substituts |
| S215 | [38242983478](https://github.com/Andriamarosoa/note/actions/runs/38242983478) | **16/16 producteurs d'origine exécutés sur les nouveaux vecteurs acoustiques**, 304 checkpoints Ridge originaux, 193 sorties S46 sur 19 branches ; 4 cellules S46 changées par rapport à S213 |
| S216 | [38243390558](https://github.com/Andriamarosoa/note/actions/runs/38243390558) | Audit indépendant sans annotations des 19 sorties S215, stabilité intra-fold et entre folds, décomposition amont/portes |

Autres dépendances : [S5→S18](README_V273_RECOVERED_S5_S18.md), [les cinq S1–S4/cohérentes reconstruites](README_V273_RECOVERED_FIVE_FAMILIES.md), [S29 et S35 vrais, runs S209](https://github.com/Andriamarosoa/note/actions/runs/38238398102) et [S210](https://github.com/Andriamarosoa/note/actions/runs/38238673938).

La cohorte GOAT contient **57 événements** ; les familles de sélection/réparation peuvent ne changer que les événements admissibles. Ces événements ont déjà été exposés lors d'expériences antérieures : leur score ne serait **pas** une validation indépendante.

## S216 : mesures définitives de stabilité pour les sorties S215

| Fold S18 | Branches historique originales | Cellules S46 différentes dans le même fold, comparées à la première branche | Événements concernés |
|---|---:|---:|---:|
| 0 | 5 | 4 | 1 |
| 1 | 4 | 96 | 1 |
| 2 | 5 | 6 | 1 |
| 4 | 5 | 0 | 0 |

Pour les premières branches représentatives de chaque fold : **1 058 / 11 001 cellules (événement, politique)** diffèrent entre les quatre folds, contre 904 dans le précédent diagnostic adapté S207. Les six événements dont le parent S18 change entre folds sont les seuls à porter ces 1 058 divergences. **1 058 / 1 058** correspondent à une situation où aucune des quatre décisions S46 ne modifie son propre parent S18, sur ces cellules ; 0 cellules divergentes ont leur parent S18 unanime.

Les décisions intra-fold divergent pourtant sur **106 cellules cumulées**, pour 3 événements distincts par fold concernés (1 chacun dans les folds 0, 1, 2 ; aucune en fold 4). Les producteurs réels conduisent donc à une instabilité non observée dans S207. Ni « changement », ni « désaccord » ne signifie correction ni régression Exact-K : **aucune étiquette MIDI n'est ouverte** par S212–S216.

## Limites de preuve

1. Le constructeur original 527D de S5 est reconstruit avec exactitude *numérique historique*, mais son code ancien et l'identité octet-à-octet de toute sa matrice ne sont pas retrouvés.
2. Les modèles S45 sont des vrais checkpoints et les 16 formules tournent, mais les modèles historiques étant entraînés/exclus par morceau, il n'existe pas encore de règle d'affectation ou d'agrégation **préenregistrée et évaluée sur de nouveaux morceaux**. Les 19 branches sont intentionnellement conservées.
3. Les sorties S46 complètes **ne sont pas** les sorties finales S80. Les portes S69 par fold, les sources S70/S71/S72, le garde S79/K1 et la queue S80 doivent être raccordés sur du nouvel audio sans fuite de labels.
4. Il faut figer l'identité du déploiement et ses prédictions sur un corpus **song-disjoint réellement inédit**, avant toute annotation, puis comparer au même événement `freeze_local_combo` en Exact-K global, Exact-K poly K2+, K0–K6, corrections et régressions appariées.

**Décision officielle** : aucune promotion ; garder `freeze_local_combo` comme référence. Aucun nouveau score d'Exact-K de S80 n'est revendiqué.

Scripts : [S212](../scripts/s212_rebase_five_original_s45.py), [S213](../scripts/s213_rebased_s45_s46_bridge.py), [S214](../scripts/s214_rebase_five_recurrent_S45.py), [S215](../scripts/s215_full_s45_16_original_sources.py), [S216](../scripts/s216_audit_full_source_s46_fold_stability.py). [Statut de promotion structuré](v273_s80_promotion_status.json).
