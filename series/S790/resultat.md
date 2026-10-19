---
serie: S790
statut: en_cours
type: experience
reference: S708
branche: codex/note-s790-independent-framepitch-causal-expert
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S790 — Résultats

## Exécution

- GitHub Actions : en attente.
- SHA / exécutions expérimentales : en attente.
- Corpus : GuitarSet FIT80 (entraînement + sélection composition-held-out), CAL32 (diagnostic gelé).
- TEST/HELD : non ouverts.

## Métriques

| Métrique officielle | Référence S708 sur CAL32 | Nouveau MIDI final |
| --- | ---: | --- |
| Onset F1 | 74,2639 % | non mesuré |
| Exact-K | 62,0455 % | non mesuré |
| Active Exact-K | 56,2762 % | non mesuré |
| Offset | 77,3292 % | non mesuré |
| Full Note F1 | 57,1319 % | non mesuré |
| Frame Pitch F1 | 76,4626 % | non mesuré |

### Frame Pitch tête isolée SHADOW (horloge 1024)

Base S708 : non mesuré par la série. Nouvelle tête S790 : non mesuré. Les deux seront comparées sur le même protocole, **sans assimiler ces chiffres aux six métriques du MIDI final**.

## Comparaison à S708

Référence S708 inchangée. Aucun nouveau score encore vérifié. Les six métriques officielles devront être conservées sur un même flux MIDI strictement causal avant toute promotion.

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

**En cours — aucune promotion.** Résultats non mesurés tant que six audits et expérimentation n'ont pas abouti.

## Prochaine action

Lancer S790, télécharger l'artefact vérifié, recalculer erreurs graves et accords, documenter les gains/régressions et décider de la prochaine correction.
