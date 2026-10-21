---
serie: S820
statut: termine
type: experience
reference: S708
branche: codex/note-s820-ss1-truek-spectral-forensics
run_id: 39559635752
commit_sha: 677dc6f486d0a347cde8d6d155b3aa07301ae822
promotion: false
---
# S820 — Résultat de l'audit sur les erreurs par vrai K SS1

## Exécution

Résultats S820 authentiques à obtenir via GitHub Actions. Toute sortie GitHub scientifique doit distinguer un PASS technique d'une preuve physique de confusion harmonique.

## Métriques

| Axe | Référence officielle S708 CAL32 | Candidat S820 |
|---|---:|---|
| Onset F1 | 74,2639 % | non modifié |
| Exact-K | 62,0455 % | non modifié |
| Active Exact-K | 56,2762 % | étude forensique sans nouvel enseignant |
| Offset accuracy | 77,3292 % | non modifié |
| Full Note F1 | 57,1319 % | non modifié |
| Frame Pitch F1 | 76,4626 % | non modifié |

Scores FIT80 authentiques connus de S819 : H0 142213/249411=57,01953803 % ; S815 141946/249411=56,91248582 % ; politique descriptive S819 max_high@0.70 141263/249411=56,63864 %. Ces chiffres ne représentent pas un nouveau résultat S820.

## Comparaison à S708

Aucune modification des modèles ou sorties S708/S707 et LIVE. Cette analyse est rétrospective et n'entraîne aucun modèle. Seules les données FIT80 déjà explorées seront consultées.

## Audits

| Domaine | Statut | Limite |
|---|---|---|
| Données | PASS | 80 sources SHA originales, 5 plis, 306536 trames MIDI256 |
| Acoustique | UNKNOWN | Proxy DSP fondamental/H2 calculé, mais attribution à des fondamentales et harmoniques physiquement non établie |
| Architecture | PASS | Aucun poids ni architecture de production modifiés |
| Apprentissage | PASS | Aucun entraînement ni sélection de politique |
| Métriques | PASS | 80 matrices vraie K/avant K/après K, S815/S819 reproduits |
| LIVE | UNKNOWN | Aucun test iRig de latence physique ; code intact |

## Verdict

Aucune promotion, aucune unification, aucune affirmation que le timbre ou l'harmonique est la cause avant résultats vérifiés. Les seuils issus des données externes doivent être fixés dans un protocole distinct pour toute nouvelle architecture.

## Prochaine action

Reproduire les matrices K0–K6 authentiques et mesurer les distributions de preuves spectrales, puis documenter les limites physiques avant toute expérimentation de correction.

## Mesures S820 authentiques

[Run S820 #39559635752](https://github.com/Andriamarosoa/note/actions/runs/39559635752) **SUCCESS** ; [artefact #12326639460](https://github.com/Andriamarosoa/note/actions/runs/39559635752/artifacts/12326639460). 80 pistes originales PCM SHA, cinq plis composition exclus, 306536 frames MIDI256, 249411 frames actives, 0 réentraînement. Baseline S815 **141946/249411** = 56,91248582 %, contre-factuel max_high@0.70 **141263/249411** = 56,63864 %, 4867 changements, delta −683.

### SS1 — variation de bonnes décisions par cardinalité GT

| Vrai K | Frames de vérité SS1 | Frames dont la décision change | Delta correct |
|---|---:|---:|---:|
| K1 | 20503 | 15 | −1 |
| K2 | 10780 | 151 | −8 |
| K3 | 8192 | 295 | −163 |
| **K4** | **12170** | **916** | **−555** |
| K5 | 4839 | 384 | +152 |
| K6 | 865 | 22 | +17 |

K0 8795 frames diagnostiques inchangées (hors métrique Active-K). SS1 total : **1783** changements, **355** anciennes erreurs devenues correctes, **913** anciennes décisions correctes détruites, net **−558**. Sur vrai K4, la transition précise ancien K4 correct → K5 se produit **597** fois, et ancien K4 correct → K6 **68** fois. **Le problème décisionnel prédominant est K4 compté K5/K6**, et non une simple absence de détection des K5/K6.

### Audit acoustique descriptif

Les bins `sharp` fondamentaux-like 126/134/142 et deuxième-harmonique-like 127/135/143 donnent un écart moyen `fund−h2` de **0,280527** sur les 355 corrections contre **0,278595** sur les 913 erreurs introduites. Les agrégats sont proches (**écart 0,001932**) et ne permettent pas de conclure que les harmoniques sont la cause. L'analyse par hauteur, note effectivement ajoutée et trajectoire acoustique est encore absente.

### Verdict

**DIAGNOSTIC TERMINÉ, CANDIDAT REJETÉ.** Aucun gain FIT80, aucune mesure sur jeu inédit ni matérielle iRig, aucune altération S708/S707/LIVE ou autres axes. Aucune politique ne peut être sélectionnée a posteriori à partir des scores outer. La cible individuelle >85 % reste non atteinte.

## Prochaine action

Inspecter les indices de fondamentales par hauteur et les candidates harmoniques/octaves pour les 665 frames vrai K4→K5/K6, isoler les accords de SS1 et généraliser par contrôles réservés sans choisir les seuils sur FIT80.
