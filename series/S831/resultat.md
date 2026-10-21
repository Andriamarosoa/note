---
serie: S831
statut: termine
type: experience
reference: S708
branche: codex/note-s831-audit-corrections256
run_id: 39640517671
commit_sha: 38708a5c4498580677da49bcbcb0858a0a427b4d
promotion: false
---

# S831 — Bilan scientifique audit Exact-K PCM256

## Exécution
**SUCCESS** [GitHub Actions #39640517671](https://github.com/Andriamarosoa/note/actions/runs/39640517671) commit scientifique `38708a5c4498580677da49bcbcb0858a0a427b4d`. 10 tests unitaires et contrôles d'indépendance PASS, cinq folds réentraînés sur 80 pistes GuitarSet FIT, 306 536 frontières PCM256. Artefact `S831_FIT80_AUDIT_CORRECTED_EXACTK_PCM256` (GitHub artifact id `12364854124`, `report.json` avec tous les 90 candidats CAL par fold et six modes OUTER par piste).

## Métriques
| Mesure toutes trames K0–K6 | Correctes | Exact-K | Sous | Sur |
|---|---:|---:|---:|---:|
| H0 historique apparié | 189 124 | **61,6972 %** | 80 654 | 36 758 |
| S829 indépendant réentraîné | 176 334 | **57,5247 %** | 83 523 | 46 679 |
| **S831 sélectionné** | **174 708** | **56,9943 %** | **82 480** | **49 348** |

**S831 régression −0,5304 point (−1 626 décisions correctes) vs S829**. Aucune promotion. Les autres axes restent indépendants.

| Composition OUTER | Mode CAL | S829 correct | S831 correct | Net |
|---|---|---:|---:|---:|
| BN1 | s829 | 35 401 | 35 401 | 0 |
| Funk2 | s829 | 40 420 | 40 420 | 0 |
| Jazz1 | blend 0.40 | 28 945 | 27 521 | −1 424 |
| Rock3 | rescue 0.40 | 36 267 | 36 065 | −202 |
| SS1 | s829 | 35 301 | 35 301 | 0 |

### Proposition par classe K — preuve des corrections utiles et des régressions
| Vrai K | S829 correct | Expert blend correct (score sur même OUTER, **non sélectionné globalement**) | Gain/perte |
|---|---:|---:|---:|
| K0 | 45 874 | 40 058 | −5 816 |
| K1 | 93 382 | 75 883 | −17 499 |
| K2 | 10 196 | 18 712 | +8 516 |
| K3 | 8 988 | 11 429 | +2 441 |
| K4 | 15 345 | 11 030 | −4 315 |
| K5 | 2 098 | 6 608 | +4 510 |
| K6 | 451 | 735 | +284 |

En tout, le candidat `blend`, évalué à la valeur de seuil CAL choisie pour chaque fold, **corrige 39 215 cas S829 auparavant faux, mais régresse 51 094 cas auparavant justes** : net −11 879 décisions. `rescue` corrige 20 523 et régresse 23 919 : net −3 396. Ce résultat est une preuve de **complémentarité**, pas de performance globale suffisante. Un sélecteur pouvant prévoir les circonstances où un expert est fiable pourrait être exploré, mais la validation OUTER ne doit jamais servir à ajuster rétroactivement une politique déclarée validée.

### Modes candidats (sélection des seuils sur inner-CAL)
| Mode | Correct / 306 536 | Corrige S829 | Dégrade S829 | Net |
|---|---:|---:|---:|---:|
| s829 | 176 334 | 0 | 0 | 0 |
| sustain | 155 368 | 37 887 | 58 853 | −20 966 |
| joint | 146 104 | 37 894 | 68 124 | −30 230 |
| router | 165 479 | 29 175 | 40 030 | −10 855 |
| blend | 164 455 | 39 215 | 51 094 | −11 879 |
| rescue | 172 938 | 20 523 | 23 919 | −3 396 |

Pour chaque mode ci-dessus le seuil utilisé dans le rapport est **celui que le fold a sélectionné globalement**, et ne prétend pas être un optimum spécifique au mode. La compétition exhaustive de seuils reste accessible dans `candidate_CAL_stats` de l'artefact.

### Couverture du rapport S831 sur les six axes
| Caractéristique | Résultat |
|---|---|
| Exact-K | 56,9943 % SHADOW FIT80/256, **pas** score MIDI officiel |
| Active Exact-K | NON MESURÉ, référence S828 inchangée |
| Onset F1 | NON MESURÉ, axe officiel inchangé |
| Offset | NON MESURÉ, axe officiel inchangé |
| Frame Pitch F1 | NON MESURÉ, axe officiel inchangé |
| Full Note F1 | NON MESURÉ, axe officiel inchangé |

## Comparaison à S708
S708 officiel CAL32 n'est ni réévalué ni modifié ; 84,24 % S809 SHADOW/1024 et 73,83 % S828 Active-K/256 sont des métriques non appariées à Exact-K FIT80/256. Les cinq autres architectures et sorties MIDI publiques restent inchangées.

## Audits
| Rôle | Statut | Preuve |
|---|---|---|
| Données | PASS | 80 WAV GuitarSet originaux, checksums S800, cinq compositions |
| Acoustique | PASS | FFT causale 96 hops, réattaque et contexte E2/E3 synthétiques ; matériel inconnu |
| Architecture | PASS | CI anti-imports autres axes, aucun fichier MIDI officiel changé |
| Apprentissage | PASS | 2 têtes K + 3 musicales privés par fold, 3 TRAIN/1 CAL/1 OUTER disjoints |
| Métriques | PASS | Toutes les propositions CAL et OUTER auditées par K et corrections/régressions |
| LIVE | UNKNOWN | Pas de temps complet PCM→MIDI mesuré sur iRig, pas de checkpoints S831 de production |

## Verdict
**S831 TERMINÉ, NON PROMU : régression 0,53 point vs S829.** Les preuves polyphoniques contiennent des corrections utiles mais les sélecteurs globaux introduisent trop de dégradations de K0/K1/K4. Évaluer uniquement sur les folds suivants un nouveau sélecteur local appris sur inner-CAL, avec validation par composition et veto strict. Les mesures de temps matériel et d'interférence entre axes restent bloquantes avant déploiement.

## Prochaine action
Concevoir S832 sur branche séparée. Conserver la reproduction exacte de S829, journaliser les décisions candidates et les caractéristiques privées, apprendre un veto local en utilisant TRAIN/inner-CAL exclusivement ; mesurer les corrections et régressions sur OUTER disjoint. Ne jamais prétendre à un gain avant 80 pistes 306 536 trames, ni partager des poids entre caractéristiques avant audit de non-régression.
