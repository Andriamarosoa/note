---
serie: S832
statut: termine
type: experience
reference: S708
branche: codex/note-s832-guarded-local256
run_id: 39644879595
commit_sha: 97c9348fa641da2c3daa56ed921f157924f2c1e1
promotion: false
---

# S832 — Résultats scientifiques exacts sur FIT80

## Exécution
GitHub Actions [#39644879595](https://github.com/Andriamarosoa/note/actions/runs/39644879595), commit scientifique `97c9348fa641da2c3daa56ed921f157924f2c1e1`, `smoke` SUCCESS et `signed_fit80` SUCCESS. Artefact `S832_FIT80_GUARDED_LOCAL_PCM256` id `12367129426` (rapport signé 80 pistes, 306 536 décisions).

## Métriques
| Modèle / protocole apparié | Exact-K K0–K6 | Correct / 306 536 | Sous | Sur |
|---|---:|---:|---:|---:|
| H0 historique | 61,6972 % | 189 124 | 80 654 | 36 758 |
| S829 référence SHADOW PCM256 | 57,5247 % | 176 334 | 83 523 | 46 679 |
| S831 candidat précédent | 56,9943 % | 174 708 | 82 480 | 49 348 |
| **S832 veto local** | **57,5247 %** | **176 334** | **83 523** | **46 679** |

**Bilan S832 : +0,0000 point / zéro régression vs S829.** Les cinq sélecteurs ont choisi zéro règle et retour intégral aux prédictions S829.

| Composition OUTER | Exact-K correct S832 | Règles acceptées | CAL sous-validation — meilleur gain observé |
|---|---:|---:|---:|
| BN1 | 35 401 / 57 744 | 0 | −53 |
| Funk2 | 40 420 / 66 144 | 0 | +44 |
| Jazz1 | 28 945 / 50 360 | 0 | −127 |
| Rock3 | 36 267 / 66 144 | 0 | +939 |
| SS1 | 35 301 / 66 144 | 0 | −5 |

Les gains de sous-validation CAL ne sont **pas** des améliorations OUTER. Les règles positives étaient refusées par le veto des régressions individuelles par piste ; ce résultat ne prouve pas que ces mêmes règles auraient généralisé.

### Couverture de l'audit
| Caractéristique | Résultat |
|---|---|
| Exact-K | 57,5247 % SHADOW FIT80 toutes trames 256 |
| Active Exact-K | NON MESURÉ, S828 officiel développement inchangé |
| Onset F1 | NON MESURÉ, architecture officielle inchangée |
| Offset | NON MESURÉ, architecture officielle inchangée |
| Frame Pitch F1 | NON MESURÉ, architecture officielle inchangée |
| Full Note F1 | NON MESURÉ, architecture officielle inchangée |

## Comparaison à S708
S708 officielle CAL32 non modifiée ; S809 SHADOW1024 84,24 % et S828 Active-K256 73,83 % non comparables directement au présent Exact-K FIT80/256. Les cinq autres axes et le MIDI public restent inchangés.

## Audits
| Rôle | Statut | Preuve |
|---|---|---|
| Données | PASS | 80 WAV FIT80 S800, SHA original, cinq compositions |
| Acoustique | PASS | Tests synthétiques causalité PCM256 ; matériel non testé |
| Architecture | PASS | Absence de dépendances aux axes voisins et fichiers protégés |
| Apprentissage | PASS | 5 modèles privés par fold, règles sur 12 CAL et veto 4 CAL |
| Métriques | PASS | 306 536 trames, S829 strictement reproduit, 0 règle |
| LIVE | UNKNOWN | Aucun p95/p99 global mesuré sur iRig, ni checkpoint déployable |

## Verdict
**TERMINÉ, NON PROMU.** S832 réalise le fallback de non-régression mais n'améliore pas Exact-K. Aucune information inter-axes ni anticipation audio ajoutée.

## Prochaine action
Nouvelle expérience isolée S833 : apprentissage/calibration plus robuste des propositions polyphoniques sur les seules pistes TRAIN/inner-CAL, en préservant un contrôle de généralisation par piste. Évaluer S833 séparément et ne modifier aucun score de référence tant que tous les folds ne sont pas vérifiés.
