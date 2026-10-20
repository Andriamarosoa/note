---
serie: S802
statut: en_cours
type: experience
reference: S708
branche: codex/note-s802-synthetic-native-retrigger-forensics
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S802 — appariement réel attaques synthétiques et vérité FIT80

## Exécution

S802 doit rejouer les trois politiques représentatives S801 depuis les 80 traces signées S800 et classifier chaque attaque additionnelle via un appariement global un-à-un sur les notes GuitarSet. Statut : en attente du run réel et des six audits. Références sources : [reconstruction S800](https://github.com/Andriamarosoa/note/actions/runs/39352880035), [ablations S801](https://github.com/Andriamarosoa/note/actions/runs/39354733640).

## Métriques

| Mesure | Officiel S708 CAL32 | S802 |
|---|---:|---|
| Onset F1 | 74,2639 % | non mesuré |
| Exact-K | 62,0455 % | non mesuré |
| Active Exact-K | 56,2762 % | non mesuré |
| Offset accuracy | 77,3292 % | non mesuré |
| Full Note F1 | 57,1319 % | non mesuré |
| Frame Pitch F1 | 76,4626 % | non mesuré |

Les collisions connues S801, **95/136 consensus, 61/90 poly_gap, 81/118 adaptive**, sont des NoteOff synthétiques lors d'une attaque native du même pitch, **pas** des faux positifs prouvés. Leur TP/FP réel est l'objet de S802. Aucune preuve du pourcentage de faux positifs n'existe encore.

## Comparaison à S708

Aucun changement des poids, du décodeur, des notes MIDI réelles, de la latence CPU ou de la référence officielle S708. Aucune évaluation CAL32 n'est effectuée ; les données de vérité FIT servent uniquement à classer après coup les événements déjà émis. Le relecture de S801 doit donner exactement les six métriques et 12 comptes MIDI FIT précédents.

## Audits

| Rôle | Statut | Preuve |
|---|---|---|
| Données | UNKNOWN | Run S802 à venir |
| Acoustique | UNKNOWN | Run S802 à venir |
| Architecture | UNKNOWN | Run S802 à venir |
| Apprentissage | UNKNOWN | Run S802 à venir |
| Métriques | UNKNOWN | Run S802 à venir |
| LIVE | UNKNOWN | Run S802 à venir |

## Verdict

Non promu et en attente. Aucune conception ajoutant lookahead, confirmation, buffering ou retard des NoteOff n'est recevable.

## Prochaine action

Lire les couples source native/synthétique classés, les taux TP/FP par composition et les distributions d'attaques initiales. Proposer seulement ensuite un éventuel veto par caractéristiques audio présentes à l'instant de création. Tout candidat devra d'abord satisfaire les six métriques FIT, puis une validation indépendante CAL32 et le test physique S793.
