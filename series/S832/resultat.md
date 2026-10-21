---
serie: S832
statut: en_cours
type: experience
reference: S708
branche: codex/note-s832-guarded-local256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S832 — Résultat du veto local indépendant Exact-K

## Exécution
En attente de la validation S832. Branche isolée après S831 ; tests unitaires, workflow scientifique et replay S800 FIT80 à exécuter sur Github. Ne pas confondre une validation technique et un gain musical.

## Métriques
| Caractéristique | Résultat |
|---|---|
| Exact-K FIT80 PCM256 S832 | NON MESURÉ |
| Exact-K FIT80 PCM256 S829 témoin | 57,5247 % sur 306 536 trames |
| Exact-K FIT80 S831 précédent | 56,9943 % sur 306 536 trames |
| Onset F1 officiel | NON MESURÉ / inchangé |
| Active Exact-K officiel | NON MESURÉ / inchangé |
| Offset officiel | NON MESURÉ / inchangé |
| Full Note F1 officiel | NON MESURÉ / inchangé |
| Frame Pitch F1 officiel | NON MESURÉ / inchangé |

## Comparaison à S708
S708 CAL32 officielle inchangée, S828 Active-K256 conservée. 84,24 % S809 SHADOW1024 non comparable avec Exact-K FIT80 256. Toute la sélection S832 est SHADOW, pas un changement MIDI public.

## Audits
| Rôle | Statut | Preuve |
|---|---|---|
| Données | UNKNOWN | Provenance S800 et 80 WAV à vérifier dans CI |
| Acoustique | UNKNOWN | Tests stricts 256 et aucune anticipation future |
| Architecture | UNKNOWN | CI contrôle anti-import et aucun changement des cinq autres axes |
| Apprentissage | UNKNOWN | Sous-splits CAL par piste avec veto et modèles privés |
| Métriques | UNKNOWN | Doit publier les gains et régressions K0–K6 par composition |
| LIVE | UNKNOWN | p95/p99 matériel iRig et checkpoints non testés |

## Verdict
EN COURS, NON PROMU. Aucun score nouveau sans run scientifique signé.

## Prochaine action
Exécuter le smoke S832 puis les cinq folds sur les 80 pistes avec le veto CAL. Auditer toute proposition acceptée, refuser une dégradation significative et ne pas autoriser la communication entre axes pour l'instant.
