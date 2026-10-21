---
serie: S834
statut: en_cours
type: experience
reference: S708
branche: codex/note-s834-s809-six-axes-native256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S834 — Résultats S809 porté à PCM256

## Exécution
Portage des six axes à 256 écrit dans une branche indépendante ; tests locaux/CI et replay FIT80 réel à exécuter. Ne pas annoncer de réussite avant les artifacts scientifiques.

## Métriques
| Caractéristique | Nouvelle mesure S834 PCM256 |
|---|---|
| Exact-K | NON MESURÉ |
| Active Exact-K | NON MESURÉ |
| Onset F1 | NON MESURÉ |
| Offset | NON MESURÉ |
| Frame Pitch F1 | NON MESURÉ |
| Full Note F1 | NON MESURÉ |

Le score 84,24 % S809 **SHADOW1024** n'est pas un score démontré à 256. H0 S708 sur FIT80 est le témoin historique signé, indépendant du choix expérimental S834. Aucune modification des sorties MIDI S708.

## Comparaison à S708
Les six scores officiels S708 CAL32 restent inchangés : Onset 74,2639 %, Exact-K 62,0455 %, Active Exact-K 56,2762 %, Offset accuracy 77,3292 %, Full Note F1 57,1319 %, Frame Pitch F1 76,4626 %. S834 travaille sur SHADOW/256, pas une seule transcription MIDI unifiée. S828 Active Exact-K/256 reste non promu et inchangé.

## Audits
| Rôle | Statut | Preuve |
|---|---|---|
| Données | UNKNOWN | Validation signé S800 FIT80 des 80 WAV à réaliser |
| Acoustique | UNKNOWN | S708 actualisé à chaque 256 et non-anticipation à valider |
| Architecture | UNKNOWN | Tests 30 comparaisons entre les six architectures attendus |
| Apprentissage | UNKNOWN | Origine checkpoint S708 gelé à auditer, aucune nouvelle tête apprise |
| Métriques | UNKNOWN | Scores par axe non promus attendus sur même cohort FIT80 |
| LIVE | UNKNOWN | CPU et iRig p95/p99 non mesurés |

## Verdict
EN COURS, NON PROMU.

## Prochaine action
Exécuter les tests d'indépendance et le replay FIT80, examiner chaque axe par K/note/onset/offset, bloquer toute régression lors d'une future sélection de branche. Ne pas modifier le moteur MIDI officiel.
