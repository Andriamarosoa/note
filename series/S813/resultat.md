---
serie: S813
statut: en_cours
type: experience
reference: S708
branche: codex/note-s813-independent-teachers85-midi256
run_id: pending
commit_sha: pending
promotion: false
---
# S813 — Premier enseignant indépendant MIDI256

## Exécution

Entraînement réellement exécuté dans cette session sur CPU, à partir du PCM GuitarSet original, puis vérification locale des décisions sauvegardées. La CI prévue effectue un recalcul indépendant des scores; ce n'est pas une seconde exécution d'entraînement.

Base S812 :b71221886292254cb71ca8343d805384be3d46b0. Source S800 :[run39352880035](https://github.com/Andriamarosoa/note/actions/runs/39352880035),80 JSON/NPZ aux SHA vérifiés. Archive audio SHA256 :237cdc58353d25c3c9683f4565a0f1cf2db30a9051abca545a919f8f1296dc28. Nouvelle extraction :80 morceaux,306 536 frames,222 caractéristiques recalculées toutes les 256 samples. L'extraction et les cinq modèles sont identifiés dans evidence/bundle.json, extraction.json et report.json.

Partitions :cinq plis par composition,48 entraînement/16 calibration/16 évaluation par pli. FIT historiquement exploré; CAL/HELD/TEST non ouverts. La comparaison H0 porte sur les mêmes annotations et horodatages mais H0 a appris historiquement sur FIT80, contrairement à la composition exclue du candidat.

## Métriques

| Enseignant nouveau | Mesure privée à 256 | Seuil |
|---|---:|---:|
| Onset F1 | non mesuré | >85 % |
| Exact-K | non mesuré | >85 % |
| Active Exact-K | non mesuré | >85 % |
| Offset | non mesuré, futurs ancrages Onset fixes | >85 % |
| Full Note F1 | non mesuré, vrais événements requis | >85 % |
| Frame Pitch F1 | **77,7132 %** | **FAIL** |

Frame Pitch :TP442 144, FP134 675, FN118 924; précision76,6521 %, rappel78,8040 %. H0 :TP414 084, FP76 787, FN146 984; F1 78,7278 %. Écart observé −1,0146 point. Le rappel augmente mais 57 888 faux positifs supplémentaires diminuent le F1; aucun gain global revendiqué.

| Composition exclue | Enseignant | Occupation H0 |
|---|---:|---:|
| BN1 | 78.1581 % | 79.3192 % |
| Funk2 | 75.4142 % | 75.8614 % |
| Jazz1 | 77.8566 % | 78.4012 % |
| Rock3 | 78.2735 % | 78.6919 % |
| SS1 | 78.2984 % | 80.4924 % |

Accords :77,4435 % contre 77,5447 % H0. Solos :78,4571 % contre 81,9817 %. Graves MIDI40–47 :72,5191 % contre 71,6553 %; médium48–68 :79,9933 % contre 79,8790 %; aigu69–88 :62,9243 % contre 78,5167 %. La forte faiblesse aiguë et les faux positifs sont des diagnostics pour la prochaine expérience, pas un motif pour recalibrer rétroactivement ces plis. Détails K0–K6 conservés dans report.json.

## Résolution et timing

Features, décisions et score :256/44100 =5,804988662 ms. Tests de mutation du futur et partitions PCM64/256/333/4096 passent sans anticipation ajoutée. Le CPU des deux têtes seules,64 mesures par modèle, donne p95 de 1,4961 à 4,2641 ms, avec un p99 jusqu'à 8,6282 ms. Extraction DSP, transfert et matériel exclus :aucune preuve de latence totale5,80 ms ni de tenue LIVE. Windows/iRig non mesuré.

## Comparaison à S708

Les 12 compteurs historiques sont reproduits exactement. Scores du flux H0 conservé :Onset73,2520 %, Exact-K61,6972 %, Active Exact-K57,0195 %, Offset69,8862 %, Full Note50,6528 %, Frame Pitch78,7278 %. Ce ne sont pas les scores des six nouveaux enseignants. Aucun MIDI unifié n'a été créé ou évalué.

## Audits

| Rôle indépendant | Verdict et portée | Preuve |
|---|---|---|
| A — architecture/LIVE | PASS causalité logicielle; UNKNOWN matériel | test_s813_native256 :mutation futur et partitions; CPU partiel dans report.json |
| B — DSP | PASS extraction réelle256; FAIL qualité aiguë |80 PCM vérifiés;222 caractéristiques; aucun H0 dans les entrées |
| C — apprentissage | PASS modèles privés du seul Frame Pitch; FAIL objectif85 |5 checkpoints;48/16/16; deux modèles distincts par enseignant |
| D — données/métriques | PASS recalcul; UNKNOWN généralisation finale | annotations SHA; TP/FP/FN exacts et 12compteurs H0; FIT recherche |
| E — code/CI | PASS 41 tests locaux; CI en attente | test_s813_*, test_s809_axis_isolation, test_s812_historical_axis_h0 |
| F — contre-revue | PASS après corrections et relecture indépendante | refus folds vides et identité de piste modifiée; cohorte commune obligatoire |

Six sous-agents réellement exécutés, sans prétendre utiliser les six configurations Work personnelles. Trois avis NVIDIA architecture/données/contradictoire :UNKNOWN, non exécutés. Aucun9/9 PASS. La vérification CI certifie les décisions sauvegardées et les contrôles exécutés, pas un réentraînement ni une performance LIVE.

## Verdict

**FAIL objectif des six enseignants >85 %.** Premier essai Frame Pitch complet, mais aucun enseignant certifié au seuil demandé. Unification=false, live_ready=false, promotion=false. Références S708/S707 préservées.

## Prochaine action

Une nouvelle expérience doit traiter les faux positifs et le registre aigu avec des sélections/têtes propres à Frame Pitch, puis construire les cinq autres enseignants. Figer les choix sur apprentissage/calibration, conserver une véritable validation hors recherche et les contraintes MIDI256. Tester la distillation seulement après six validations >85 % sur une cohorte commune; réévaluer ensuite les six métriques du flux MIDI unifié.
