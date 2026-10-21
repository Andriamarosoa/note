---
serie: S834
statut: termine
type: experience
reference: S708
branche: codex/note-s834-s809-six-axes-native256
run_id: 39658093988
commit_sha: 4d5c445c6f44d1c956f781f85764fe12bfba4983
promotion: false
---

# S834 — Résultats scientifiques S809 authentique porté à PCM256

## Exécution
Succès complet [GitHub Actions #39658093988](https://github.com/Andriamarosoa/note/actions/runs/39658093988), commit code `4d5c445c6f44d1c956f781f85764fe12bfba4983`. 41 tests S834/S809/S763/S712 PASS en CI. Artefact `S834_REAL_S809_SIX_AXES_PCM256_FIT80` id `12376305671`. **80 vraies pistes**, 5 compositions ×16, **306 536** fenêtres PCM256, 306 536 nouvelles inférences S708 à [49,288], 153 600/153 600 contrôles d'absence d'influence des modifications d'un axe sur les cinq autres.

## Métriques
### Les six résultats SHADOW isolés S834 — métriques locales, pas une transcription MIDI unifiée

| Axe | Compteur S834 | Score SHADOW FIT80 natif256 |
|---|---|---:|
| Exact-K | 232 295 / 306 536 | **75,780659 %** |
| Active Exact-K | 162 600 / 249 411 trames actives | **65,193596 %** |
| Onset F1 | TP 8 482, FP 4 016, FN 2 785 | **71,382285 %** |
| Offset F1 | TP 2 825, FP 649, FN 8 309 | **38,677437 %** |
| Frame Pitch F1 | TP 467 540, FP 71 501, FN 93 528 | **84,998850 %** |
| Full Note F1 | TP 8 224, FP 3 640, FN 3 043 | **71,108037 %** |

### Comparaison avec H0 MIDI historique sur les mêmes 80 pistes
| Axe, métrique | H0 historique signé FIT80 | S834 SHADOW FIT80 | Écart numérique |
|---|---:|---:|---:|
| Exact-K toutes trames | 61,697158 % | **75,780659 %** | +14,0835 points |
| Active Exact-K trames actives | 57,019538 % | **65,193596 %** | +8,1741 points |
| Onset F1 | **73,252019 %** | 71,382285 % | **−1,8697 point** |
| Frame Pitch F1 | 78,727759 % | **84,998850 %** | +6,2711 points |
| Full Note F1 | 50,652809 % | **71,108037 %** | +20,4552 points |
| Offset | accuracy 69,8862 % | F1 événements 38,6774 % | NON COMPARABLE |

Onset régresse sur H0 ; les autres scores sont des **évaluations indépendantes des axes SHADOW** sur FIT80. Ils ne permettent pas à eux seuls de prétendre à la non-régression du MIDI final. S829/S832 Exact-K privé sur les mêmes 306 536 trames 57,524728 % ; gain S834 **+18,255931 points**. Record S809 84,24 % sur SHADOW1024 non directement comparable.

### Non-interférence et coût CPU
| Vérification | Résultat |
|---|---|
| Nouvelles prédictions S708 fraîches PCM256 | 306 536/306 536 |
| Mutations intentionnelles d'un sélecteur de chacun des six axes | 6 par fenêtre × 64 premiers hops/piste |
| Comparaisons de décision et sélection des cinq autres axes | **153 600/153 600 identiques** |
| Modifications des six pipelines historiques et MIDI source | Aucune |
| p95 CPU Linux traitement bloc, intervalle des 80 pistes | **6,61 à 7,75 ms** |
| p99 CPU Linux traitement bloc, intervalle des 80 pistes | **6,81 à 9,50 ms** |
| Période disponible par hop à 44,1 kHz | 5,805 ms |
| LIVE physique iRig, E2/accords, p95/p99 bout-à-bout | **NON MESURÉ** |

Les p95 CPU sont plus longs que la période de 256 échantillons, **sur le runner CI Linux**. Il ne s'agit pas d'une mesure de latence de l'interface audio ou du pipeline Windows iRig, mais c'est un blocage avant promotion en LIVE.

## Comparaison à S708
La référence officielle CAL32 reste protégée : Onset F1 74,2639 %, Exact-K 62,0455 %, Active Exact-K 56,2762 %, Offset accuracy 77,3292 %, Full Note F1 57,1319 %, Frame Pitch F1 76,4626 %. S828 privé 73,831146 % Active Exact-K FIT80 natif256 reste une expérimentation distincte non modifiée. Aucune des six sorties MIDI officielles n'a été remplacée ; S834 est évalué en SHADOW.

## Audits
| Rôle | Statut | Preuve |
|---|---|---|
| Données | PASS | SHA S800 80 WAV, S708 gelé identifié, checksum H0 |
| Acoustique | PASS | Chaque frontière native 256 produit une inférence S708 sans futur, poids original entraînés à 1024 |
| Architecture | PASS | 41 tests, 153 600 contre-exemples zéro interférence, tous les fichiers officiels intacts |
| Apprentissage | PASS | S708 source gelé non réentraîné ; aucune assertion d'entraînement 256 ou de vrai OOF S708 |
| Métriques | PASS | TP/FP/FN chacun des six axes, comparaison FIT80 appariée H0/S829 ; régression Onset déclarée |
| LIVE | FAIL | Tous les p95 CPU des 80 pistes >5,805 ms sur CI Linux, p99 iRig et déploiement physique UNKNOWN |

## Verdict
**S834 TERMINÉ, NON PROMU.** Importante progression d'Exact-K à 75,78 %, Frame Pitch F1 à 85,00 % et Full Note F1 à 71,11 % sur des sorties SHADOW natives 256 ; 153 600 relations de non-interférence prouvées. Les contraintes utilisateur ne sont **pas entièrement satisfaites** : Onset F1 inférieur au H0 et traitement CPU CI plus lent que 5,805 ms en p95. Offset F1 difficile et métrique différente du témoin Offset accuracy. Aucune mesure p99 Windows/iRig.

## Prochaine action
S835 devra résoudre les régressions Onset et Offset sur métriques comparables et réduire la surcharge CPU (notamment traitement spectral recalculé), garder chaque architecture indépendante, mesurer de nouveau les six axes sur les mêmes 80 pistes avec une décision toutes les 256 valeurs. Utiliser exclusivement TRAIN/inner-CAL pour apprendre/sélectionner les règles et évaluer OUTER seulement ensuite, sans revendiquer de nouvelle généralisation externe car FIT80 a déjà été exploré. Jamais promouvoir une architecture plus lente que le budget LIVE ou qui dégrade une caractéristique officiellement utilisée.
