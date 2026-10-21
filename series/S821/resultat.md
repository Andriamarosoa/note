---
serie: S821
statut: en_cours
type: experience
reference: S708
branche: codex/note-s821-nvidia-ss1-k4-guard
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S821 — Résultats de la revue et du correctif Active-K

## Exécution

Consultations NVIDIA indépendantes et test de garde-fou acoustique à exécuter, sans résultat musical inventé.

## Métriques

| Axe | Référence officielle S708 CAL32 | S821 SHADOW |
|---|---:|---|
| Onset F1 | 74,2639 % | inchangé |
| Exact-K | 62,0455 % | inchangé |
| Active Exact-K | 56,2762 % | non mesuré |
| Offset accuracy | 77,3292 % | inchangé |
| Full Note F1 | 57,1319 % | inchangé |
| Frame Pitch F1 | 76,4626 % | inchangé |

Comparaison équitable FIT80 : H0 57,019538 %, S815 56,912486 %, S817 56,912486 %, S820 max_high@.70 56,63864064 % (diagnostic, non candidat). FIT80 déjà exploré, non preuve de généralisation.

## Comparaison à S708

Aucune modification S708, aucun MIDI LIVE branché, aucun nouveau score des six axes. Seule recherche Active-K SHADOW, décisions 256 échantillons causales.

## Audits

| Domaine | Statut | Motif |
|---|---|---|
| Données | UNKNOWN | SHA et partitions à vérifier dans CI |
| Acoustique | UNKNOWN | Preuves f0/harmoniques individuelles à auditer |
| Architecture | UNKNOWN | Poids et sélecteur privés à vérifier |
| Apprentissage | UNKNOWN | Train/cal/test composition-disjoints à vérifier |
| Métriques | UNKNOWN | Reproduction des compteurs avant tout score |
| LIVE | UNKNOWN | Absence de futur à vérifier, latence physique non mesurée |

## Verdict

En cours, aucun candidat S821 promu ni unifié. L'avis d'un LLM ne constitue jamais une mesure d'amélioration.

## Prochaine action

Revoir indépendamment les trois réponses NVIDIA, vérifier leurs contre-arguments, construire le garde-fou conditionnel K4 et tester le nouveau modèle sur cinq compositions.
