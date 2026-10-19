---
serie: S793
statut: en_cours
type: infrastructure
reference: S708
branche: codex/note-s793-zero-added-latency-hard-gate
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S793 — Résultats de la règle de promotion sans délai ajouté

## Exécution

Treize tests anti-régression exécutés localement avec succès. Rejeu sur le rapport S792 GitHub #39328341071 : retenue zéro bloc. Validation CI dédiée : en attente. Aucun test physique Windows/iRig effectué.

## Métriques

Les six métriques officielles S708 GuitarSet CAL32 restent inchangées : Onset F1 74,2639 %, Exact-K 62,0455 %, Active Exact-K 56,2762 %, Offset accuracy 77,3292 %, Full Note F1 57,1319 %, Frame Pitch F1 76,4626 %. Aucun nouveau score.

La règle impose zéro échantillon supplémentaire d'attente sur NoteOn, NoteOff, buffering, lookahead et ordonnancement. Les chemins matériels requièrent p50/p95/maximum du temps CPU, capture vers NoteOn, capture vers NoteOff et délai global, sur le même périphérique. Sans mesures, promotion bloquée.

## Comparaison à S708

S708 reste intacte. S792 a été rejetée à cause des pertes Offset/Full Note et du délai ajouté aux NoteOff. Le contrôle des identifiants d'événement exige que les sorties originales ne soient jamais émises plus tard.

## Audits

| Rôle | Statut | Preuve |
| --- | --- | --- |
| Données | UNKNOWN | CI en attente |
| Acoustique | UNKNOWN | CI en attente |
| Architecture | UNKNOWN | CI en attente |
| Apprentissage | UNKNOWN | CI en attente |
| Métriques | UNKNOWN | CI en attente |
| LIVE | UNKNOWN | CI en attente |

## Verdict

Règle : rejeter une architecture dès qu'elle augmente la latence d'une attaque, d'une extinction, d'un traitement ou de la sortie, indépendamment des gains F1. La latence physique incompressible existe ; le critère est l'augmentation par rapport à la référence. Le CI ne peut pas mesurer à distance un périphérique LIVE.

## Prochaine action

Valider le CI et faire de cette barrière une condition obligatoire de toutes les promotions suivantes, incluant les mesures physiques appariées.
