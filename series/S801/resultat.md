---
serie: S801
statut: en_cours
type: experience
reference: S708
branche: codex/note-s801-fit-fragment-metric-ablations
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S801 — résultats réels des huit ablations MIDI256 sur FIT80

## Exécution

S801 dépend directement des archives OOF S800 run [#39352880035](https://github.com/Andriamarosoa/note/actions/runs/39352880035), dont la reconstruction 80/80 et les contrôles signés sont PASS, avec réaudit indépendant PASS au run [#39354212035](https://github.com/Andriamarosoa/note/actions/runs/39354212035). L'évaluation de huit politiques S801 et ses audits sont en attente d'exécution. Il n'y a aucun accès CAL32.

## Métriques

| Métrique | S708 CAL32 officiel | S800 FIT80 contrôle | S801 candidat FIT80 |
|---|---:|---:|---|
| Onset F1 | 74,2639 % | 73,2520 % | non mesuré |
| Exact-K | 62,0455 % | 61,6972 % | non mesuré |
| Active Exact-K | 56,2762 % | 57,0195 % | non mesuré |
| Offset accuracy | 77,3292 % | 69,8862 % | non mesuré |
| Full Note F1 | 57,1319 % | 50,6528 % | non mesuré |
| Frame Pitch F1 | 76,4626 % | 78,7278 % | non mesuré |

## Comparaison à S708

Aucun des scores officiels CAL32 n'a été remis en cause ni modifié. S801 mesure d'abord FIT80 avec les cartes de probabilités OOF authentiques, sans utiliser CAL pour sélectionner le meilleur masque. Aucun hold ni confirmation supplémentaire de NoteOff et aucune modification du modèle LIVE. Les gains éventuels devront être réévalués sans fuite sur CAL et sans latence physique ajoutée.

## Audits

| Rôle | Statut | Preuve |
|---|---|---|
| Données | UNKNOWN | En attente d'évaluation S801 |
| Acoustique | UNKNOWN | En attente d'évaluation S801 |
| Architecture | UNKNOWN | En attente d'évaluation S801 |
| Apprentissage | UNKNOWN | En attente d'évaluation S801 |
| Métriques | UNKNOWN | En attente d'évaluation S801 |
| LIVE | UNKNOWN | En attente d'évaluation S801 |

## Verdict

En cours, non promu. Aucun candidat n'est promu par une seule mesure favorable. Le choix sur FIT80 est exploratoire jusqu'à vérification officielle CAL32 de la parité des événements et six métriques ainsi que des mesures CPU/latence physiques.

## Prochaine action

Lancer la lecture et l'évaluation de l'artefact de 80 traces et les six audits ; si les huit politiques échouent aux garde-fous, préserver l'identité. Après cela, investiguer la signature d'une fausse attaque et le réarmement Offset sans décaler une fin de note réelle. Garder le conseiller NVIDIA consultatif seulement.
