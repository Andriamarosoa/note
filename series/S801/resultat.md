---
serie: S801
statut: termine
type: experience
reference: S708
branche: codex/note-s801-fit-fragment-metric-ablations
run_id: 39354733640
commit_sha: ce470b37a0cc5c21533086cd46e3793a9490b7a3
promotion: false
---

# S801 — résultats réels des huit ablations MIDI256 sur FIT80

## Exécution

S801 dépend directement des archives OOF S800 run [#39352880035](https://github.com/Andriamarosoa/note/actions/runs/39352880035), dont la reconstruction 80/80 et les contrôles signés sont PASS, avec réaudit indépendant PASS au run [#39354212035](https://github.com/Andriamarosoa/note/actions/runs/39354212035). L'évaluation des huit politiques S801 a **réussi** au [run GitHub #39354733640](https://github.com/Andriamarosoa/note/actions/runs/39354733640) (commit `ce470b37a0cc5c21533086cd46e3793a9490b7a3`). Les 80 traces FIT OOF complètes ont été contrôlées par SHA256, les 12 comptes sources FIT reproduits, et les six rôles auditeurs ainsi que le consensus sont PASS. Il n'y a aucun accès CAL32.

## Métriques

| Métrique | S708 CAL32 officiel | S800 FIT80 contrôle | S801 candidat FIT80 |
|---|---:|---:|---|
| Onset F1 | 74,2639 % | 73,2520 % | aucune alternative admissible |
| Exact-K | 62,0455 % | 61,6972 % | aucune alternative admissible |
| Active Exact-K | 56,2762 % | 57,0195 % | aucune alternative admissible |
| Offset accuracy | 77,3292 % | 69,8862 % | aucune alternative admissible |
| Full Note F1 | 57,1319 % | 50,6528 % | aucune alternative admissible |
| Frame Pitch F1 | 76,4626 % | 78,7278 % | aucune alternative admissible |


## Huit ablations mesurées — variations en points de pourcentage sur FIT80 OOF

| Politique | Δ Onset F1 | Δ Exact-K | Δ Active Exact-K | Δ Offset accuracy | Δ Full Note F1 | Δ Frame Pitch F1 | NoteOn synthétiques |
|---|---:|---:|---:|---:|---:|---:|---:|
| Identité source | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Consensus | −0,3741 | +0,0153 | +0,0221 | −0,4997 | −0,6186 | +0,0869 | 136 |
| Veto harmonique | −0,3741 | +0,0153 | +0,0221 | −0,4997 | −0,6186 | +0,0869 | 136 |
| Poly-gap | −0,2393 | +0,0130 | +0,0160 | −0,2901 | −0,3747 | +0,0583 | 90 |
| Adaptative | −0,3352 | +0,0062 | +0,0108 | −0,4240 | −0,5374 | +0,0778 | 118 |
| Garde Onset native | −0,3352 | +0,0062 | +0,0108 | −0,4240 | −0,5374 | +0,0778 | 118 |
| Garde création Offset | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Garde parent harmonique | −0,3741 | +0,0153 | +0,0221 | −0,4997 | −0,6186 | +0,0869 | 136 |

Les gains Exact-K sont très faibles et ne compensent pas les pertes Onset, Offset et Full Note F1. Les scores sont recalculés sur les **vrais événements MIDI256 GuitarSet FIT80**, pas une fixture synthétique. Le veto de non-régression est strict sur chaque mesure ET composition, et non une moyenne permettant d'écraser un échec local.

## Audit des réattaques natives et faiblesses par composition

Le champ du rapport JSON `added_offset_reason_totals` compte en réalité toutes les **raisons d'événement préfixées `s795_`**, dont `s795_multihead_attack` (NoteOn). Il ne faut pas traiter ce dernier compteur comme un nombre de NoteOff. Pour les NoteOff `s795_source_retrigger` : consensus **95/136**, adaptative **81/118**, poly-gap **61/90**. Cela indique des notes synthétiques qui ont dû être fermées lorsqu'une vraie attaque native de la même hauteur s'est produite. La proportion est élevée, mais ne démontre pas à elle seule que la note synthétique était fausse : cet arbitrage demande l'alignement événement-vérité de S802.

Par composition, Active Exact-K **recule sur Jazz1** pour consensus (−0,0345 pt), adaptative (−0,0543 pt) et poly-gap (−0,0493 pt). Sur Jazz1, l'adaptative perd aussi 0,6235 pt d'Onset F1 et 0,8041 pt d'Offset accuracy. Les variations locales invalident toute politique uniforme ou sélection basée sur la seule moyenne FIT.

**Ne pas utiliser les réattaques natives futures comme feature à l'inférence.** Elles seront analysées hors ligne pour comprendre le défaut, puis seules des caractéristiques présentes à la frame de décision pourront guider une hypothèse causale.

## Comparaison à S708

Aucun des scores officiels CAL32 n'a été remis en cause ni modifié. S801 mesure d'abord FIT80 avec les cartes de probabilités OOF authentiques, sans utiliser CAL pour sélectionner le meilleur masque. Aucun hold ni confirmation supplémentaire de NoteOff et aucune modification du modèle LIVE. Les gains éventuels devront être réévalués sans fuite sur CAL et sans latence physique ajoutée.

## Audits

| Rôle | Statut | Preuve |
|---|---|---|
| Données | PASS | [Run S801 #39354733640](https://github.com/Andriamarosoa/note/actions/runs/39354733640), audit indépendant |
| Acoustique | PASS | [Run S801 #39354733640](https://github.com/Andriamarosoa/note/actions/runs/39354733640), audit indépendant |
| Architecture | PASS | [Run S801 #39354733640](https://github.com/Andriamarosoa/note/actions/runs/39354733640), audit indépendant |
| Apprentissage | PASS | [Run S801 #39354733640](https://github.com/Andriamarosoa/note/actions/runs/39354733640), audit indépendant |
| Métriques | PASS | [Run S801 #39354733640](https://github.com/Andriamarosoa/note/actions/runs/39354733640), audit indépendant |
| LIVE | PASS | [Run S801 #39354733640](https://github.com/Andriamarosoa/note/actions/runs/39354733640), audit indépendant |

## Verdict

**S801 terminée et non promue.** Les huit évaluations FIT80 OOF se terminent, mais zéro politique non identitaire franchit le veto de non-régression sur les six mesures et les cinq compositions. Sélection provisoire FIT : `identity`. Le choix n'est pas promu ; CAL32/HELD/TEST sont fermés et aucune latence CPU physique n'a été comparée.

## Prochaine action

Lancer S802 sur les traces FIT OOF réelles : audit forensique de chaque attaque synthétique selon le délai avant réattaque native, la présence de la fondamentale, l'éligibilité acoustique et la vérité d'attaque ; quantifier les sous-groupes par composition. Ne pas entraîner un nouveau modèle ni utiliser la future attaque native dans le décodeur : elle est permise uniquement dans **l'audit hors ligne**. Concevoir ensuite un éventuel veto causal strictement à la frame courante, sans décaler un NoteOff. Garder le conseiller NVIDIA consultatif seulement.
