---
serie: S796
statut: en_cours
type: experience
reference: S708
branche: codex/note-s796-retrigger-forensics-no-extra-latency
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S796 — Bilan mesuré de S795 et correctif logiciel non promu

## Exécution

S795 : [run 39340511467](https://github.com/Andriamarosoa/note/actions/runs/39340511467) et artefact `S795_ACTIVEK_MULTIHEAD_REAL_MIDI256_FIT80_CAL32`. FIT80 + CAL32 parcourus, mais CAL refusé avant publication, car k_correct=78367 au lieu de 78366 dans le rapport signé. S796 exécute des tests et audits légers sans réentraîner les 80+32 fichiers.

## Métriques

| Métrique FIT80 GuitarSet | Référence S708/FIT | Meilleur candidat Active-K S795/FIT |
|---|---:|---:|
| Onset F1 | 73,2520 % | 68,7671 % |
| Exact-K | 61,6972 % | 62,7075 % |
| Active Exact-K | 57,0195 % | 58,2925 % |
| Offset accuracy | 69,8862 % | 61,5546 % |
| Full Note F1 | 50,6528 % | 41,8761 % |
| Frame Pitch F1 | 78,7278 % | 79,6088 % |

Candidat `mask31_cut0.85` : +1,27 point Active Exact-K contre −8,78 points Full Note F1, −8,33 points Offset et −4,48 points Onset. Un seul candidat sur 25 admissible = identité native (0 tête retenue). Le score S796 après correctif n'est pas mesuré. Pas de chiffres CAL publiables.

## Comparaison à S708

Le système S708/S707 de production reste strictement inchangé. Aucun gain S796 déclaré et aucune confirmation de non-régression des six axes, donc aucune promotion. Les scores du tableau concernent FIT80 sur l'ancien S795 non promu.

## Audits

| Rôle | Statut | Preuve |
|---|---|---|
| Données | UNKNOWN | Exécution S796 en attente |
| Acoustique | UNKNOWN | Exécution S796 en attente |
| Architecture | UNKNOWN | Exécution S796 en attente |
| Apprentissage | UNKNOWN | Exécution S796 en attente |
| Métriques | UNKNOWN | Exécution S796 en attente |
| LIVE | UNKNOWN | Exécution S796 en attente |

## Verdict

Échec scientifique S795 confirmé ; correctif S796 seulement candidat d'ingénierie. La causalité du guard est testable, mais le benchmark réel du microphone et la latence additionnelle CPU ne sont pas mesurés. Aucune promotion vers LIVE, ni accès HELD/TEST.

## Prochaine action

Valider le test de réarmement et les six audits indépendants, puis concevoir une sélection de têtes adaptée à chaque fragment (et pas un masque global) qui pénalise directement les faux NoteOn et l'erreur NoteOff. Exiger tous les scores midi256 et absence de latence ajoutée avant d'envisager une promotion.
