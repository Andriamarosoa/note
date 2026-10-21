---
serie: S826
statut: en_cours
type: experience
reference: S708
branche: codex/note-s826-exact-s708-native256-rebuild
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S826 — Réexécution S708 entière sur grille d'extraction, inférence et événements MIDI256

## Exécution

Moteur S708 complet strict256 à construire et tester sur checkpoint d'origine SHA vérifié; **le premier checkpoint est entraîné à une cadence 1024**. Résultats du modèle adaptés à 256 NON MESURÉS.

## Métriques

| Évaluation | S708 référence CAL32 | S826 strict256 |
|---|---:|---|
| Onset F1 | 74,2639 % | NON MESURÉ |
| Exact-K | 62,0455 % | NON MESURÉ |
| Active Exact-K | 56,2762 % | NON MESURÉ |
| Offset accuracy | 77,3292 % | NON MESURÉ |
| Full Note F1 | 57,1319 % | NON MESURÉ |
| Frame Pitch F1 | 76,4626 % | NON MESURÉ |

## Comparaison à S708

S708 ancien : extraction des 126D natifs chaque 256 mais 288D et tête temporelle+GuidedSelector tous les 1024. S826 : même architecture tous les 256, sans dépendance à un nouveau modèle S825. **La parité des poids ne garantit pas parité des scores quand l'horloge change : réentraîner la tête S708 avec des séquences à 256.** S823 et ses 76,8775 % ne sont pas un score de la même chaîne S708 strict256.

## Audits

| Domaine | Statut | Objectif |
|---|---|---|
| Données | UNKNOWN | Chronologie stricte 256, mono 44,1kHz |
| Acoustique | UNKNOWN | Identité 192D historiques + 96D sharp courants, aucune trame future |
| Architecture | UNKNOWN | Même TemporalHeads/GuidedSelector à chaque 256; ancien S712 default inchangé |
| Apprentissage | UNKNOWN | Checkpoint S708 original SHA; futur apprentissage 256 non effectué |
| Métriques | UNKNOWN | Six scores S704 exacts, même flux MIDI et split gelé |
| LIVE | UNKNOWN | Sélection opt-in uniquement, pas branché sur production ni temps micro certifié |

## Verdict

Pas de promotion avant preuve complète et réentraînement 256; ne pas présenter le contrôle 1024 comme une reproduction 256.

## Prochaine action

Évaluer la parité à 1024 et la cadence à 256 sur un vrai checkpoint; puis réentraîner la **même** TemporalHeads S708 avec `decision_ends=np.arange(256,N+1,256)` et recalibrer GuidedSelector sans hold ajouté, comparer les six scores, K0–K6, la latence réelle et les non-régressions.
