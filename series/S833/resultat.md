---
serie: S833
statut: en_cours
type: experience
reference: S708
branche: codex/note-s833-crossfit-risk256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S833 — Résultats du risque Exact-K avec quatre cross-validations CAL

## Exécution
Nouvelle branche d'expérimentation S833 sans modification des cinq autres caractéristiques. CI technique et replay FIT80 à exécuter. Score S833 encore NON MESURÉ.

## Métriques
| Caractéristique | Résultat |
|---|---|
| Exact-K S833 FIT80 K0–K6 PCM256 | NON MESURÉ |
| Exact-K S829 témoin apparié | 57,5247 % = 176 334/306 536 |
| Exact-K S832 témoin | 57,5247 % = 176 334/306 536 |
| Onset F1 officiel | NON MESURÉ / inchangé |
| Active Exact-K officiel | NON MESURÉ / inchangé |
| Offset officiel | NON MESURÉ / inchangé |
| Full Note F1 officiel | NON MESURÉ / inchangé |
| Frame Pitch F1 officiel | NON MESURÉ / inchangé |

## Comparaison à S708
S708 CAL32 reste officielle, S828 Active-K256 reste référence de développement ; aucune de leurs mesures ne se substitue à Exact-K FIT80/256. Les cinq autres axes et le MIDI restent inchangés. Les scores S809 SHADOW/1024 ne sont pas directement comparables.

## Audits
| Rôle | Statut | Preuve |
|---|---|---|
| Données | UNKNOWN | Replay S800 original FIT80 à valider |
| Acoustique | UNKNOWN | Validation PCM256 causale et SHAs |
| Architecture | UNKNOWN | Import/fichiers protégés à contrôler |
| Apprentissage | UNKNOWN | Quatre groupes CAL sans pistes communes |
| Métriques | UNKNOWN | Résultats appariés 306 536 trames attendus |
| LIVE | UNKNOWN | Aucune mesure iRig p99 physique |

## Verdict
EN COURS, NON PROMU. Même après succès FIT80, une confirmation sur des données HELD inédites et un runtime réel reste requise.

## Prochaine action
Exécuter le smoke S833 et les cinq folds FIT80, analyser K0–K6 par piste, mesurer les nouvelles corrections et régressions, conserver S829/S828 intacts en l'absence de gain prouvé.
