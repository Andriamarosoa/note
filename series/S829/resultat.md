---
serie: S829
statut: termine
type: experience
reference: S708
branche: codex/note-s829-s809-exactk-native256
run_id: 39624926356
commit_sha: 655d14bba15c028b257f6c01a701e086f715b2c3
promotion: false
---

# S829 — évaluation Exact-K SHADOW natif256

## Exécution
**VALIDATION SIGNÉE TERMINÉE** : [GitHub Actions #39624926356](https://github.com/Andriamarosoa/note/actions/runs/39624926356) ; commit de code `655d14bba15c028b257f6c01a701e086f715b2c3`. Smoke indépendant : sept tests PASS ; entraînement et évaluation `signed_fit80` PASS. Artefact : `S829_FULL_FIT80_S809_EXACTK_ONLY_PCM256_SHA` (ID GitHub 12357195711), `report.json`, `complete=true`, 80 pistes, 5 folds, 306 536 trames. Manifeste source SHA256 `c7b0edbf07e891ed5478af1599830f2c0785391b813cd1c77f2c48ad86685a11`.

La chaîne ne charge aucun checkpoint S708/S810/S828. Son propre DSP et deux HGB distincts sont entraînés à partir du PCM, et leur sélecteur est choisi sur inner-CAL. Aucune fusion MIDI ni modification des cinq autres axes. **L'inférence en ligne avec modèle entraîné n'a que des tests synthétiques** : le replay musical FIT80 a calculé les caractéristiques par hop puis évalué les prédictions des modèles hors ligne ; il ne prouve pas la latence d'exécution LIVE ni l'absence de différence de performance matérielle.

## Métriques
Métrique principale : **Exact-K K0–K6 sur toutes les trames**, et non Active Exact-K conditionné par activité. Résultats FIT80 sur les **mêmes** 80 pistes et 306 536 frontières PCM256 :

| Source | Correct | Trames | Exact-K | Sous-comptage | Surcomptage |
|---|---:|---:|---:|---:|---:|
| H0 historique reconstruit à 256 | 189 124 | 306 536 | 61,6972 % | 80 654 | 36 758 |
| **S829 deux têtes indépendantes** | **176 334** | **306 536** | **57,5247 %** | **83 523** | **46 679** |
| Proxy acoustique sans entraînement | 20 941 | 306 536 | 6,8315 % | 3 501 | 282 094 |

**Régression mesurée S829 vs H0 : −4,1724 points** sur même protocole. Les 84,24 % de S809 (1 024 / SHADOW) et 73,83 % de S828 (Active Exact-K / 256) ne sont **pas** des comparaisons scientifiques directes.

| Composition OUTER | Correct | Trames | Exact-K S829 | Sous | Sur |
|---|---:|---:|---:|---:|---:|
| BN1 | 35 401 | 57 744 | 61,31 % | 12 392 | 9 951 |
| Funk2 | 40 420 | 66 144 | 61,11 % | 16 695 | 9 029 |
| Jazz1 | 28 945 | 50 360 | 57,48 % | 15 668 | 5 747 |
| Rock3 | 36 267 | 66 144 | 54,83 % | 19 024 | 10 853 |
| SS1 | 35 301 | 66 144 | 53,37 % | 19 744 | 11 099 |

Le sélecteur a choisi `fusion, shift=0` sur les **cinq folds** : les modes alternatifs n'ont pas été retenus par la calibration.

| Vrai K | Trames | Correct | Accuracy K | Sous | Sur |
|---|---:|---:|---:|---:|---:|
| K0 | 57 125 | 45 874 | 80,30 % | 0 | 11 251 |
| K1 | 112 014 | 93 382 | 83,37 % | 9 160 | 9 472 |
| K2 | 46 311 | 10 196 | 22,02 % | 24 957 | 11 158 |
| K3 | 34 182 | 8 988 | 26,29 % | 14 612 | 10 582 |
| K4 | 35 190 | 15 345 | 43,61 % | 16 132 | 3 713 |
| K5 | 16 323 | 2 098 | 12,85 % | 13 722 | 503 |
| K6 | 5 391 | 451 | 8,37 % | 4 940 | 0 |

### Déclaration des six métriques obligatoires

| Axe officiel NOTE | Résultat S829 du MIDI final |
|---|---|
| Onset F1 | NON MESURÉ : MIDI final inchangé |
| Exact-K | 57,5247 % : uniquement tête SHADOW K0–K6 ; **pas** score MIDI final |
| Active Exact-K | NON MESURÉ sur nouveau MIDI, S828 conservé comme référence séparée |
| Offset | NON MESURÉ : MIDI final inchangé |
| Full Note F1 | NON MESURÉ : aucun assemblage des têtes indépendantes |
| Frame Pitch F1 | NON MESURÉ : MIDI final inchangé |

## Comparaison à S708
La référence officielle S708 (CAL32) et sa branche restent inchangées. H0 sur FIT80 est un **témoin apparié**, pas la même mesure que S708 CAL32. L'axe expérimental S829 conserve le code indépendant mais **régresse** en performance face à H0. Les cinq autres caractéristiques Onset, Active Exact-K, Offset, Frame Pitch et Full Note n'ont pas été remplacées, et leurs scores MIDI unifiés ne sont **pas mesurés sur S829**. Aucune prétention d'amélioration des six axes.

## Audits
| Rôle | Statut | Preuve |
|---|---|---|
| Données | PASS | WAV et manifeste signés, 80 pistes, cinq compositions, 306 536 trames ; run 39624926356 |
| Acoustique | UNKNOWN | Mutation du futur et découpage PCM synthétiques PASS, mais budget matériel et perturbation sur 80 WAV non testés |
| Architecture | UNKNOWN | Imports interdits absents, deux têtes indépendantes, états privés, fichiers protégés ; contre-audit multi-axes manquant |
| Apprentissage | PASS | 5 splits outer composition-disjoint, 3 TRAIN/1 CAL/1 OUTER, sélecteur choisi sans OUTER |
| Métriques | PASS | Exact-K toutes trames et K0–K6, compte correct + sous + sur = trames |
| LIVE | UNKNOWN | Aucun p95/p99 CPU ou iRig, aucun résultat du modèle en LIVE matériel |

Trois avis NVIDIA authentiques et six agents Work personnalisés : **non exécutés**, 9/9 NON revendiqué. Aucun certificat de non-régression multi-axes.

## Verdict
**TERMINÉ — NON PROMU (performance insuffisante).** Score S829 **57,52 %** contre **61,70 % H0** sur même grille FIT80. Le système K autonome est fonctionnel et validé sur ce corpus, mais le problème de sous-comptage massif K2–K6 interdit de le qualifier de meilleure référence. Les modèles entraînés ne sont pas livrés comme checkpoints de déploiement dans cet artefact. Aucun changement de S828 référence Active Exact-K ni du MIDI officiel.

## Prochaine action
Priorité : analyser les confusions acoustiques pour K2/K3/K4/K5/K6, les notes superposées et les fondamentales faibles. Concevoir de **nouveaux experts Exact-K indépendants** au lieu de réutiliser l'encodeur d'autres axes. Conserver PCM256 causal, zéro lookahead/hold, entraîner sur TRAIN, choisir sur inner-CAL, reproduire les cinq folds, garder H0 comme témoin. Exiger ensuite une validation HELD externe et la mesure réelle p95/p99 CPU avant toute promotion.
