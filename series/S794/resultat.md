---
serie: S794
statut: en_cours
type: experience
reference: S708
branche: codex/note-s794-activek-zero-delay-forensics
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S794 — Résultats Active Exact-K causal MIDI 256

## Exécution

GuitarSet FIT80/CAL32, S708/S707 signés ; expérience et audit en cours. Aucun nouvel événement MIDI ni décodeur de production modifiés.

## Métriques

| Métrique officielle S708 CAL32 | Référence | S794 |
| --- | ---: | --- |
| Onset F1 | 74,2639 % | source inchangée, audit en cours |
| Exact-K | 62,0455 % | source inchangée, audit en cours |
| Active Exact-K | 56,2762 % | source inchangée, audit en cours |
| Offset accuracy | 77,3292 % | source inchangée, audit en cours |
| Full Note F1 | 57,1319 % | source inchangée, audit en cours |
| Frame Pitch F1 | 76,4626 % | source inchangée, audit en cours |

Les mesures diagnostiques visent les sous-comptages, surcomptages, K1–K6 et registres, et ne sont pas des gains ou de nouveaux scores.

## Comparaison à S708

S794 ne modifie aucun événement source. Le rapport scientifique doit démontrer une parité stricte avec les compteurs de S708 et identifier les causes dominantes des erreurs Active Exact-K avant toute proposition d'entraînement.

## Audits

| Rôle | Statut | Preuve |
| --- | --- | --- |
| Données | UNKNOWN | En attente |
| Acoustique | UNKNOWN | En attente |
| Architecture | UNKNOWN | En attente |
| Apprentissage | UNKNOWN | En attente |
| Métriques | UNKNOWN | En attente |
| LIVE | UNKNOWN | En attente |

## Verdict

Analyse en cours. Aucune architecture multi-têtes promue, aucun changement du MIDI final ni de la latence.

## Prochaine action

Auditer les erreurs K1–K6 et leurs causes au niveau des événements réels, créer le nombre nécessaire de spécialistes, les valider sur FIT par composition, puis sur CAL avec toutes les métriques et la barrière S793.
