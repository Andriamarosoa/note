---
serie: S791
statut: en_cours
type: experience
reference: S708
branche: codex/note-s791-framepitch-complementary-experts
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S791 — Résultats

## Exécution

- Entraînement et validation FIT : en attente.
- Diagnostic CAL : en attente.
- GitHub Actions run ID / commit : en attente.
- Corpus : GuitarSet FIT80/CAL32, HELD et TEST fermés.

## Métriques

| Mesure | S708 officiel CAL32 | S791 officiel |
| --- | ---: | --- |
| Onset F1 | 74,2639 % | non mesuré |
| Exact-K | 62,0455 % | non mesuré |
| Active Exact-K | 56,2762 % | non mesuré |
| Offset | 77,3292 % | non mesuré |
| Full Note F1 | 57,1319 % | non mesuré |
| Frame Pitch F1 | 76,4626 % | non mesuré |

Métriques SHADOW 1024 : non mesuré. Les scores officiels demeurent inchangés.

## Comparaison à S708

S791 recherche des têtes complémentaires mais n'a pas encore vérifié un nouveau score expérimental. Toutes les améliorations devront être contrôlées sur un flux MIDI complet pour être promues.

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

Non terminé. Aucun modèle promu. Il est interdit d'annoncer plus de 80 % avant des résultats reproductibles.

## Prochaine action

Terminer l'implémentation du routeur par fragment, tester l'apport marginal de chaque spécialiste et lancer l'évaluation FIT80/CAL32 avec six audits.
