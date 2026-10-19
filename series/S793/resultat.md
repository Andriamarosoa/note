---
serie: S793
statut: termine
type: infrastructure
reference: S708
branche: codex/note-s793-zero-added-latency-hard-gate
run_id: 39332676717
commit_sha: f768e0b9ca9f2100bb33f5db406e7297371e7310
promotion: false
---

# S793 — Résultats de la règle de promotion sans délai ajouté

## Exécution

Treize tests anti-régression exécutés localement avec succès. Rejeu sur le rapport S792 GitHub #39328341071 : retenue zéro bloc. Validation CI [GitHub Actions #39332676717](https://github.com/Andriamarosoa/note/actions/runs/39332676717) : **SUCCESS** (13 tests de veto et 4 scénarios d'application du garde-fou au validateur de promotion, plus les tests de reporting existants). Aucun test physique Windows/iRig effectué.

## Métriques

Les six métriques officielles S708 GuitarSet CAL32 restent inchangées : Onset F1 74,2639 %, Exact-K 62,0455 %, Active Exact-K 56,2762 %, Offset accuracy 77,3292 %, Full Note F1 57,1319 %, Frame Pitch F1 76,4626 %. Aucun nouveau score.

La règle impose zéro échantillon supplémentaire d'attente sur NoteOn, NoteOff, buffering, lookahead et ordonnancement. Les chemins matériels requièrent p50/p95/maximum du temps CPU, capture vers NoteOn, capture vers NoteOff et délai global, sur le même périphérique. Sans mesures, promotion bloquée.

## Comparaison à S708

S708 reste intacte. S792 a été rejetée à cause des pertes Offset/Full Note et du délai ajouté aux NoteOff. Le contrôle des identifiants d'événement exige que les sorties originales ne soient jamais émises plus tard.

## Audits

| Rôle | Statut | Preuve |
| --- | --- | --- |
| Données | UNKNOWN | Contrôle de provenance logique, aucune nouvelle source évaluée |
| Acoustique | UNKNOWN | Aucun nouvel essai acoustique |
| Architecture | UNKNOWN | Politique 0 retard testée, pas audit expérimental d'une nouvelle architecture |
| Apprentissage | UNKNOWN | Aucun modèle nouvellement entraîné |
| Métriques | UNKNOWN | Scores S708 inchangés, pas de nouvelle évaluation métrique |
| LIVE | UNKNOWN | Aucune mesure réelle de microphone ; promotion bloquée sans preuve |

## Verdict

**Contrôle logiciel terminé et validé en CI.** Règle : rejeter une architecture dès qu'elle augmente la latence d'une attaque, d'une extinction, d'un traitement ou de la sortie, indépendamment des gains F1. La latence physique incompressible existe ; le critère est l'augmentation par rapport à la référence. Le CI ne peut pas mesurer à distance un périphérique LIVE.

## Prochaine action

Intégrer S793 à la branche retenue du projet pour que les futures branches héritent de la règle, puis exiger la mesure matérielle couplée pour chaque promotion S794+. Tant que S793 n'est pas fusionnée, elle ne protège automatiquement que les branches qui l'incluent.
