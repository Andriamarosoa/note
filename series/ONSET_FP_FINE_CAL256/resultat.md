# Résultats scientifiques — H_FP_CAL_GRID_04 (issue #145), 23 octobre 2026

## Verdict confirmé : REFUTED_OR_INCOMPLETE — aucune promotion

**Workflow [#39905578488](https://github.com/Andriamarosoa/note/actions/runs/39905578488)** : tests `smoke` **SUCCESS**, deux expériences `track_uniform` et `track_class` terminées sur les **80 vrais WAV GuitarSet et 306536 frontières audio PCM256** chacune, sans lookahead, ni modification des autres cinq architectures ou du MIDI officiel. Les deux jobs sont marqués `failure` car leur **assertion scientifique** se déclenche après le replay ; ce n'est PAS une erreur de téléchargement, de syntaxe, de données ou une interruption du calcul.

Les réseaux ont été formés lors du run d'origine [#39900882849](https://github.com/Andriamarosoa/note/actions/runs/39900882849). Cette nouvelle expérimentation a repris **exactement les 10 poids signés SHA256** et modifié uniquement la précision de la grille de sélection de `calibrated_cut` sur CAL. Aucun poids neuronal réentraîné, ni comparaison OUTER utilisée pour sélectionner les seuils.

### Scores réellement évalués sur audio, non des proxies CAL

| Modèle natif 256 | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Référence S613 + S701 + ErrorPrediction69 + FNRecovery (4 têtes) | 8160 | 4605 | 3107 | 67.9094541 % |
| Meilleur contrôle FP69 antérieur (**référence à battre**) | 8127 | **4519** | 3140 | **67.9713963 %** |
| **`track_uniform` + grille fine** | **8060** | **4455** | **3207** | **67.7823564 %** |
| **`track_class` + grille fine** | **7834** | **4312** | **3433** | **66.9200871 %** |

`track_uniform` contre contrôle FP69 : **64 FP éliminés, mais 67 TP perdus**, donc F1 **−0.18904 point**. `track_class` : **207 FP éliminés, mais 293 TP perdus**, donc F1 **−1.05131 point**. Les deux échouent au critère **F1 strictement >67.9713963% ET FP strictement <4519**. Aucun seuil n'a été réoptimisé après inspection du résultat OUTER.

### Calibrations CAL / preuves que le blocage précédent a été corrigé

| Composition OUTER | CAL | Cut `track_uniform` | Cut `track_class` |
|---|---|---:|---:|
| BN1 | Funk2 | 0.005 | 0.002 |
| Funk2 | Jazz1 | 0.0007 | 0.001 |
| Jazz1 | Rock3 | 0.0005 | 0.0005 |
| Rock3 | SS1 | 0.002 | 0.0015 |
| SS1 | BN1 | 0 | 0 |

**Quatre plis sur cinq ACTIVENT réellement le nouveau discriminateur** dans chacun des deux bras. Le protocole grossier d'origine avait conservé zéro sur chacun des dix plis, donc l'expérience ancienne était peu informative : correction technique/expérimentale désormais validée. Cette fois, **le vrai compromis TP/FP généralise mal entre CAL et OUTER**, ce qui réfute l'explication selon laquelle la seule grille trop grossière empêchait un modèle autrement meilleur.

Les réseaux 70D et 74D originaux sont gelés et reproduits sans divergence : **70D TP8137 FP4549 FN3130**, **74D TP7948 FP4290 FN3319**. Le système quatre-têtes reproduit TP8160 FP4605 FN3107. Aucun 1024-trained checkpoint, aucune latence volontaire, aucun futur audio, aucune modification des cinq autres caractéristiques ou production MIDI.

### Artefacts complets conservés, même si le gate échoue

- [Track Uniform — artefact #12481946180](https://github.com/Andriamarosoa/note/actions/runs/39905578488/artifacts/12481946180)
- [Track Class — artefact #12481455992](https://github.com/Andriamarosoa/note/actions/runs/39905578488/artifacts/12481455992)

Chacun inclut `trained/report.json`, les modèles/metadata SHA256, `replayed/report.json` et les valeurs par composition/piste. Les catégories d'erreurs et les régressions individuelles doivent guider les prochaines hypothèses ; **ne pas manipuler à nouveau ces seuils sur OUTER**.

La meilleure référence scientifique mesurée dans cette série demeure **67.9713963% F1** (contrôle FP69 non promu au MIDI). Ce FIT80 n'est PAS un corpus externe totalement neuf : la recherche a déjà vu ses OUTER et les poids S613/S701 ont un overlap historique. Toute promotion définitive nécessite un vrai test externe indépendant, des non-régressions sur tous les autres axes et mesure de latence end-to-end p95/p99.

**Règle de réfutation respectée : [#140](https://github.com/Andriamarosoa/note/issues/140) ; résultat enregistré : [#145](https://github.com/Andriamarosoa/note/issues/145).**
