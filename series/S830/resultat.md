---
serie: S830
statut: en_cours
type: experience
reference: S708
branche: codex/note-s830-private-musical-evidence256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S830 — reconstruction des signaux musicaux autonomes pour Exact-K

## Exécution
Code de la nouvelle expérience S830 présent sur branche distincte, tests et CI à exécuter. Aucun run scientifique FIT80 S830 complet certifié au moment de la création du rapport.

## Métriques
Métrique principale : Exact-K K0–K6 **toutes trames** à 256 échantillons, 80 pistes FIT80. Objectif dépasser le score S829 apparié sans perdre l'indépendance.

| Mesure | S829 source | S830 candidat |
|---|---:|---:|
| Exact-K toutes frames, FIT80 | 57,5247 % | NON MESURÉ |
| H0 original FIT80, témoin | 61,6972 % | NON MESURÉ / à reproduire |
| Onset F1 (MIDI final) | Inchangé / non mesuré | NON MESURÉ |
| Active Exact-K (MIDI final) | Inchangé / non mesuré | NON MESURÉ |
| Offset (MIDI final) | Inchangé / non mesuré | NON MESURÉ |
| Full Note F1 (MIDI final) | Inchangé / non mesuré | NON MESURÉ |
| Frame Pitch F1 (MIDI final) | Inchangé / non mesuré | NON MESURÉ |

Les têtes appelées « Onset », « Sustain », « Release » sont des **experts privés de comptage Exact-K**, elles ne constituent pas les métriques Onset F1 ou Offset officielles du système.

## Comparaison à S708
S708 reste référence officielle CAL32, S828 référence de développement Active Exact-K 256, S829 référence locale de comparaison Exact-K indépendante. Aucune sortie MIDI officielle n'est modifiée. Les scores S809 SHADOW/1024 ne sont pas utilisés comme validateur direct S830/256.

## Audits
| Rôle | Statut | Preuve manquante |
|---|---|---|
| Données | UNKNOWN | Exécuter FIT80 signé 80/80 et vérifier les dénominateurs |
| Acoustique | UNKNOWN | Tests causalité et nouvelles 256 features |
| Architecture | UNKNOWN | Test anti-import et protection de cinq autres axes |
| Apprentissage | UNKNOWN | Vrais folds et trois têtes entraînées + deux K |
| Métriques | UNKNOWN | S829 recalculé sur le même OUTER, erreurs par K |
| LIVE | UNKNOWN | Parité synthétique online et runtime matériel p99 |

## Verdict
**NON PROMU — NON MESURÉ.** Ne pas comparer des valeurs disparates ni modifier le MIDI public.

## Prochaine action
Faire passer les tests CI, exécuter le replay exact à 80 pistes, vérifier les décisions K2–K6, rejeter tout lookahead et toute dégradation des cinq autres axes. Après résultats, seulement ensuite étudier les interactions entre caractéristiques.
