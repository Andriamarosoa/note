---
serie: S831
statut: en_cours
type: experience
reference: S708
branche: codex/note-s831-audit-corrections256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S831 — Résultats de corrections du code après audit

## Exécution
Développement séparé sur `codex/note-s831-audit-corrections256` à partir de S830, sans modifier le dépôt de référence. Exécution des tests et replay FIT80 à vérifier avant tout verdict.

## Métriques
| Caractéristique | Score expérimental S831 |
|---|---|
| Exact-K sur 80 FIT, 256 échantillons | NON MESURÉ |
| Onset F1 MIDI final | NON MESURÉ / code inchangé |
| Active Exact-K MIDI final | NON MESURÉ / code inchangé |
| Offset MIDI final | NON MESURÉ / code inchangé |
| Full Note F1 MIDI final | NON MESURÉ / code inchangé |
| Frame Pitch F1 MIDI final | NON MESURÉ / code inchangé |

S829 témoin : 57,5247 % Exact-K toutes trames, H0 historique : 61,6972 % sur 306 536 trames FIT80. Les six métriques du MIDI unifié ne sont pas déduites de l'axe Exact-K isolé.

## Comparaison à S708
Aucune promotion et aucune modification de S708 officielle CAL32, S828 Active Exact-K256, Onset, Offset, Frame Pitch, Full Note ni du MIDI public.

## Audits
| Rôle | Statut | Preuve |
|---|---|---|
| Données | UNKNOWN | FIT80 à rejouer avec SHA audio et mêmes 80 pistes |
| Acoustique | UNKNOWN | Tests longue FFT E2/E3 et vraie réattaque à lancer |
| Architecture | UNKNOWN | Imports autres axes et fichiers modifiés à contrôler en CI |
| Apprentissage | UNKNOWN | Vrais modèles réentraînés sur TRAIN disjoint |
| Métriques | UNKNOWN | Métriques de tous les modes/calibrations à collecter |
| LIVE | UNKNOWN | Latence totale et p99 sur iRig non mesurées |

## Verdict
NON PROMU, test scientifique S831 non conclu.

## Prochaine action
Faire passer les tests unitaires et le replay FIT80 signé, vérifier les erreurs par K et la sélection CAL. Ne permettre aucun partage d'état entre architectures ; différer l'influence entre les caractéristiques.
