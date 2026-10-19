---
serie: S789
statut: termine
type: experience
reference: S708
branche: codex/note-s789-deep-data-audit
run_id: 39274713682
commit_sha: e0e5bccb9386d7704f771d18a9a66816e4014ff4
promotion: false
---

# S789 — Résultats

## Exécution

- Six audits profonds : **6 PASS / 6**, consensus **PASS**.
- GitHub Actions : [run #39274713682 — SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/39274713682).
- SHA du code audité : `e0e5bccb9386d7704f771d18a9a66816e4014ff4`.
- Artefact : `S789_SIX_DEEP_DATA_AUDIT_CONSENSUS`, incluant les six rapports distincts.
- Sources immuables : S708 #39094682419, S781 #39256936602, S782 #39257378677.
- Corpus : GuitarSet CAL32, DEV 12 morceaux et HELD 32 morceaux déjà exposés, S781 FIT80 ; pas de TEST ouvert.
- Aucune nouvelle inférence, aucun changement de LIVE, aucun modèle promu.

## Métriques

Référence officielle S708, même sortie MIDI causale à cadence 256 échantillons :

| Caractéristique | CAL32 | DEV déjà exposé | HELD déjà exposé | Écart CAL vers 80 |
| --- | ---: | ---: | ---: | ---: |
| Onset F1 | 74,2639 % | 68,7520 % | 72,9183 % | +5,7361 pts |
| Exact-K | 62,0455 % | 58,9728 % | 62,2400 % | +17,9545 pts |
| Active Exact-K | 56,2762 % | 54,6090 % | 58,7670 % | +23,7238 pts |
| Offset | 77,3292 % | 73,7345 % | 68,6553 % | +2,6708 pts |
| Full Note F1 | 57,1319 % | 50,3095 % | 49,1470 % | +22,8681 pts |
| Frame Pitch F1 | 76,4626 % | 69,2472 % | **80,9591 %** | +3,5374 pts |

**Aucune métrique nouvelle mesurée** par S789 ; ces valeurs sont les métriques historiques S708 recalculées/contrôlées à partir des artefacts, et non les résultats d'un entraînement S789.

### Profondeur par morceau et texture

HELD déjà exposé, 32 pistes : Frame Pitch F1 **85,20 % sur solo** contre **79,20 % sur accompagnement** ; Offset **80,35 % sur solo** contre **62,45 % sur accompagnement**. Par composition, Frame Pitch HELD **82,32 % sur SS3** contre **79,17 % sur Jazz3**. Offset HELD chute jusqu'à **34,31 % sur une piste d'accompagnement SS3** (35 offsets corrects sur 102 éléments admissibles).

DEV déjà exposé, 12 pistes : Frame Pitch **65,02 % sur accompagnement** et **79,35 % sur solo**. L'intervalle bootstrap descriptif au niveau des pistes pour l'Offset HELD est **61,77 % – 74,90 %**. Les pistes d'une même composition ne sont pas indépendantes ; ce n'est **pas** une validation statistique sur nouvelles compositions.

### Racines acoustiques

S782 CAL32 : rappel Onset E2–B2 **54,08 %**, contre **78,08 % sur C4–E5** (−24,01 pts). Voisinage harmonique : rappel **60,80 %**, contre **73,63 %** lorsqu'il n'y en a pas (−12,83 pts). Le diagnostic S782 utilise trois fenêtres d'observation après les attaques : ces mesures **ne peuvent pas devenir directement des décisions strictement LIVE**.

S781 tête K isolée, horloge 1024 (et non MIDI final) : rappel K5 FIT **85,35 %** contre CAL **28,40 %** (−56,95 pts) ; K6 FIT **84,18 %** contre CAL **1,06 %** (−83,13 pts). Sur CAL, **96/189 fragments vrais K6** sont prédits K5, et seulement **2/189** correctement K6. Ce transfert FIT/CAL insuffisant bloque la promotion.

### Effort théorique pour dépasser 80

À dénominateur/FP constants : Offset doit gagner **78 fins correctement identifiées parmi 2 898 évaluées** ; Exact-K **22 678 fragments supplémentaires** ; Active Exact-K **23 782 fragments**. Onset F1 exige au minimum **376 attaques manquées récupérées sans nouveau faux positif**, tandis que Frame Pitch F1 exige **10 837 erreurs FN converties en TP sans faux positif supplémentaire**. Ces calculs sont des bornes arithmétiques, pas des gains réalisables garantis.

## Comparaison à S708

S708 reste la référence. S789 confirme qu'**Offset est numériquement le plus proche de 80 % sur CAL32**, mais qu'il généralise mal à certains accompagnements HELD. **Frame Pitch est la meilleure piste exploratoire vers 80 %**, car son score HELD historique franchit déjà ce seuil ; les différences entre compositions empêchent toutefois d'annoncer une validation indépendante.

## Audits

| Rôle | Statut | Preuve |
| --- | --- | --- |
| Données | PASS | [Run S789](https://github.com/Andriamarosoa/note/actions/runs/39274713682), `audits/role_data.json` : 12+32 pistes recalculées |
| Acoustique | PASS | [Run S789](https://github.com/Andriamarosoa/note/actions/runs/39274713682), `audits/role_acoustics.json` : registres, voisinages, fenêtres post-attaque |
| Architecture | PASS | [Run S789](https://github.com/Andriamarosoa/note/actions/runs/39274713682), `audits/role_architecture.json` : confusions K5/K6 et cadence |
| Apprentissage | PASS | [Run S789](https://github.com/Andriamarosoa/note/actions/runs/39274713682), `audits/role_training.json` : FIT OOF et shift CAL |
| Métriques | PASS | [Run S789](https://github.com/Andriamarosoa/note/actions/runs/39274713682), `audits/role_metrics.json` : six scores et dénominateurs |
| LIVE | PASS | [Run S789](https://github.com/Andriamarosoa/note/actions/runs/39274713682), `audits/role_live.json` : pas de modification du système |

Consensus `consensus/report.json` : `passed=true`, `audits_received=6`, `issues=[]`, promotion interdite.

## Verdict

**AUDITS PASS, PROMOTION REFUSÉE**. Le plus rapide à 80 % numériquement : Offset (77,33 %). Le meilleur candidat de recherche à poursuivre pour un score reproductible proche ou supérieur à 80 % : Frame Pitch, tout en traitant en priorité les accompagnements et les cordes graves. HELD et CAL ont été exposés lors de recherches précédentes, donc le prochain modèle doit être évalué sur un véritable protocole indépendant. Les six métriques officielles >80 % sur le même flux MIDI causal ne sont **pas** atteintes.

## Prochaine action

Créer S790 avec les deux rapports obligatoires dès le démarrage et les six audits. Mesurer une nouvelle tête Frame Pitch avec erreurs par registre/harmonique et validation par composition, FIT seulement pour choisir ; vérifier après intégration réelle au flux MIDI 256 et ne pas ouvrir TEST avant le protocole figé.
