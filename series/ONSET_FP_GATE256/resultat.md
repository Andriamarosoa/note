# Résultat scientifique vérifié — H_FP_EXPERT_GATE_02 PCM256

**VERDICT : REFUTED_OR_INCOMPLETE — pas de promotion.** Hypothèse enregistrée **avant expérimentation** : [issue #143](https://github.com/Andriamarosoa/note/issues/143), règle générale de réfutation [#140](https://github.com/Andriamarosoa/note/issues/140).

Le [workflow GitHub #39900140856](https://github.com/Andriamarosoa/note/actions/runs/39900140856) a terminé :
- Smoke : **SUCCESS** ; invariant des quatre anciennes têtes, 69/70/74 neurones, PCM256 et pas d'information future.
- Entraînement : 5 véritables petites têtes apprenantes MLP(16,8), input 11D, séparées entre CAL-TRAIN (8 pistes) et CAL-VAL (8 pistes), OUTER jamais consultée ; 4 activées, 1 désactivée avec fallback contrôle69.
- Replay causal : **80 WAV GuitarSet, 306536 frontières PCM256**, sans modification de S613, S701, MLP69 ErrorPrediction, FNRecovery, contrôle69/70/74. GT seulement après fin de l'audio. Vrai `replayed/report.json` complet ; échec de job **SCIENTIFIQUE attendu**, faute d'atteindre simultanément les deux critères.
- [Artefact #12479667666](https://github.com/Andriamarosoa/note/actions/runs/39900140856/artifacts/12479667666), contenant `trained/report.json`, 5 checkpoints/metadata SHA256 et `replayed/report.json`.

## Résultats du VRAI replay, pas le proxy CAL

| Branche expérimentale | TP | FP | FN | F1 réel |
|---|---:|---:|---:|---:|
| Quatre têtes gelées | 8160 | 4605 | 3107 | 67.9094541% |
| **Meilleur contrôle FP69 gelé** | **8127** | 4519 | **3140** | **67.9713963%** |
| Gate expert neuronal 11D (nouveau) | 7978 | **4284** | 3289 | 67.8141867% |

L'arbitre diminue FP de **235** contre 69D mais sacrifie **149 TP** ; FN +149, F1 **−0.1572096 point**. Rejet, conformément à la prédiction avant test exigeant F1 >67.9713963% ET FP <4519.

### Régressions par composition

| Composition | TP contrôle 69D | FP contrôle 69D | TP gate | FP gate | Δ TP | Δ FP |
|---|---:|---:|---:|---:|---:|---:|
| BN1 | 1197 | 646 | 1175 | 613 | −22 | −33 |
| Funk2 | 1757 | 663 | 1752 | 649 | −5 | −14 |
| Jazz1 | 1198 | 484 | 1198 | 484 | 0 | 0 |
| **Rock3** | 2437 | 1366 | 2349 | 1252 | **−88** | −114 |
| SS1 | 1538 | 1360 | 1504 | 1286 | −34 | −74 |

**Rock3** représente 88/149 = 59,1% des TP sacrifiés. La piste `00_Rock3-148-C_comp` perd 39 TP pour 28 FP supprimés ; `02_Rock3-148-C_comp` perd 24 TP et supprime 40 FP. **22 pistes présentent une régression de F1**. Les 8 pistes CAL d'entraînement de gate et 8 pistes CAL de sélection de seuil étaient disjointes, pourtant le mécanisme ne généralise pas entre textures/compositions. La problématique FP n'est pas « tous les parents harmoniques = veto », comme l'ont réfuté les expériences #142 et #143.

### Statut et prochaine preuve requise

Ne PAS ajuster opportunistement le seuil gate après avoir observé les résultats OUTER. **H_FP_EXPERT_GATE_02 insuffisante** ; auditer les cas Rock3 supprimés, leurs spectres réellement causaux et la variation des compositions. Deux nouvelles boucles indépendantes préenregistrées [#144](https://github.com/Andriamarosoa/note/issues/144) testent la couverture des pistes TRAIN du modèle 69D (`track_uniform` / `track_class`) sans modifier les anciennes têtes ni les autres axes.

**Pas de poids entraînés 1024 ; pas de futur audio/lookahead/hold ; MIDI de production inchangé et cinq autres caractéristiques intouchées.** Le corpus FIT80 a un historique de recouvrement training sur S613/S701 ; résultats non généralisables sans test complet indépendant. La cadence 5.805ms est une frontière audio, pas une mesure de latence matérielle p95/p99.
