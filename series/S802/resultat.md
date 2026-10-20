---
serie: S802
statut: termine
type: experience
reference: S708
branche: codex/note-s802-synthetic-native-retrigger-forensics
run_id: 39357068590
commit_sha: 90e73bfc69ddab31588ed20e53137c06a213b007
promotion: false
---

# S802 — appariement réel attaques synthétiques et vérité FIT80

## Exécution

S802 doit rejouer les trois politiques représentatives S801 depuis les 80 traces signées S800 et classifier chaque attaque additionnelle via un appariement global un-à-un sur les notes GuitarSet. Statut : **run #39357068590 SUCCESS**, reproductibilité des six mesures S801 et six audits indépendants avec consensus PASS. La sortie JSON est archivée sous `S802_REAL_FIT80_TRUTH_MATCHED_COLLISION_CAUSES`. Références sources : [reconstruction S800](https://github.com/Andriamarosoa/note/actions/runs/39352880035), [ablations S801](https://github.com/Andriamarosoa/note/actions/runs/39354733640).

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

## Résultats S802 — vraies notes contre faux positifs

Source : [run #39357068590](https://github.com/Andriamarosoa/note/actions/runs/39357068590), authentiques 80 traces OOF GuitarSet FIT80, appariement optimal un-à-un des NoteOn selon la fonction `event_score` (tolérance de 80 ms). Chiffres non issus de simulations.

| Politique | NoteOn synthétiques | TP selon annotations | FP selon annotations | Ferme sur attaque native | TP parmi ces collisions | FP parmi ces collisions |
|---|---:|---:|---:|---:|---:|---:|
| Consensus | 136 | 74 | 62 | 95 | 64 | 31 |
| Poly-gap | 90 | 44 | 46 | 61 | 37 | 24 |
| Adaptive | 118 | 67 | 51 | 81 | 58 | 23 |

**Une collision n'est pas un faux positif.** Pour le consensus, 64 des 95 fermetures `s795_source_retrigger` correspondaient à de véritables attaques correctement appariées selon les annotations. Tout veto qui éliminerait aveuglément ces 95 notes supprimerait donc également des vrais positifs. Ce résultat renforce l'interdiction de patienter pour observer l'attaque native future.

Sur le consensus, **84/95 collisions** se sont produites au plus tard 1 024 échantillons (soit environ 23,2 ms à 44,1 kHz) après l'attaque synthétique ; parmi ces 84, **58 étaient TP** et 26 FP. La proximité temporelle n'est pas, à elle seule, une preuve d'erreur.

### Variations acoustiques et risque par composition

| Composition FIT | Consensus TP synthétiques | Consensus FP synthétiques | Total |
|---|---:|---:|---:|
| BN1 | 10 | 6 | 16 |
| Funk2 | 22 | 3 | 25 |
| Jazz1 | 18 | 17 | 35 |
| Rock3 | 11 | 3 | 14 |
| SS1 | 13 | 33 | 46 |

La forte disparité SS1 contre Funk2 suggère des conditions acoustiques/rythmiques différentes, sans constituer une règle causale utilisable : **l'identifiant de composition et les annotations ne doivent jamais entrer dans le sélecteur LIVE.** Il faudra identifier des caractéristiques calculables sur la frame présente et les généraliser hors composition.

Sur le consensus, les médianes de `on_rise` sont respectivement 0,7151 (TP) et 0,6958 (FP), celles de `general_p` 0,9484 (TP) et 0,9436 (FP), et celles de la caractéristique `harmonic_parent` 0,0079 (TP) et 0,0168 (FP). Leur **fort recouvrement apparent** interdit de déclarer un seuil fixe gagnant à partir de ces seuls résumés. Ces médianes sont descriptives, pas des seuils entraînés.

### Conséquences pour S803

1. Auditer le profil acoustique des FP surtout sur SS1 et Jazz1 contre les TP de Funk2 ; garder les têtes spécialisées indépendantes et le routage par fragment/hauteur.
2. Tester des garde-fous fondés uniquement sur la frame courante (évidence d'attaque, confiance relative entre têtes, parent harmonique), en conservant les six métriques et toutes les compositions.
3. Ne **jamais** attendre les attaques futures, retarder un NoteOn/NoteOff, allonger l'offset, utiliser l'identifiant de composition comme feature ou extrapoler un gain CAL32 depuis l'audit FIT.
4. Tout changement devra être testé sur les sorties MIDI256 réellement émises et le contrat S793 de latence CPU physique avant toute promotion.


## Comparaison à S708

Aucun changement des poids, du décodeur, des notes MIDI réelles, de la latence CPU ou de la référence officielle S708. Aucune évaluation CAL32 n'est effectuée ; les données de vérité FIT servent uniquement à classer après coup les événements déjà émis. Le relecture de S801 doit donner exactement les six métriques et 12 comptes MIDI FIT précédents.

## Audits

| Rôle | Statut | Preuve |
|---|---|---|
| Données | PASS | [Run S802 #39357068590](https://github.com/Andriamarosoa/note/actions/runs/39357068590) |
| Acoustique | PASS | [Run S802 #39357068590](https://github.com/Andriamarosoa/note/actions/runs/39357068590) |
| Architecture | PASS | [Run S802 #39357068590](https://github.com/Andriamarosoa/note/actions/runs/39357068590) |
| Apprentissage | PASS | [Run S802 #39357068590](https://github.com/Andriamarosoa/note/actions/runs/39357068590) |
| Métriques | PASS | [Run S802 #39357068590](https://github.com/Andriamarosoa/note/actions/runs/39357068590) |
| LIVE | PASS | [Run S802 #39357068590](https://github.com/Andriamarosoa/note/actions/runs/39357068590) |

## Verdict

**Terminée, non promue.** La relecture reproduit les comptes FIT80 S800 et tous les comptes et métriques des trois politiques S801. Six audits et consensus PASS. Aucune conception ajoutant lookahead, confirmation, buffering ou retard des NoteOff n'est recevable ; aucune amélioration CAL32 n'est mesurée.

## Prochaine action

Étudier les sous-groupes de faux positifs observés, surtout SS1/Jazz1, sans chercher un simple délai de confirmation. Les distributions initiales des probabilités des têtes se recouvrent fortement entre vrais et faux positifs ; une règle de veto naïve risquerait donc de supprimer de vraies notes. Proposer seulement ensuite un éventuel veto par caractéristiques audio présentes à l'instant de création. Tout candidat devra d'abord satisfaire les six métriques FIT, puis une validation indépendante CAL32 et le test physique S793.
