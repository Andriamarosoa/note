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


## Septième conseiller scientifique — correction de l'omission

Le [run indépendant NVIDIA #39358229144](https://github.com/Andriamarosoa/note/actions/runs/39358229144) est **SUCCESS**, modèle réel `z-ai/glm-5.3-flash`, 5 234 tokens d'entrée et 1 909 tokens de réponse (7 143 au total). Le job amont a **rechargé, validé et recombiné les six rapports S802 originaux** du [run scientifique #39357068590](https://github.com/Andriamarosoa/note/actions/runs/39357068590) : consensus PASS. L'artefact GitHub Actions `S802_SEVENTH_NVIDIA_INDEPENDENT_REVIEW` contient l'avis textuel complet, non contraignant et sans clé API. Aucune sortie audio brute, aucune source et aucune prédiction OOF n'ont été envoyées : trois documents `resultat.md` S800–S802 ont constitué le contexte autorisé.

L'avis externe considère que les collisions `s795_source_retrigger` ne peuvent pas être assimilées directement à des faux positifs : **64 des 95** notes synthétiques fermées par attaque native sont correctement appariées en onset. Il souligne qu'un vrai positif en Onset peut toutefois être **faux en Offset** ; cela doit être mesuré séparément, sans retarder les fins de notes.

Recommandations de recherche du **septième conseiller** (toutes non validées) :

1. **S803a** : quantifier ROC/AUC des caractéristiques accessibles à la frame courante, avec incertitude par composition. Le seuil AUC supérieur à 0,75 est un critère *proposé* par le conseiller, pas un résultat.
2. **S803b** : tester un veto mono-caractéristique éventuel avec les six métriques FIT80 et veto de toute régression, après avoir établi qu'il existe une séparation réelle.
3. **S803c** : expliquer pourquoi SS1 accumule 33 FP contre 3 pour Funk2 via caractéristiques acoustiques accessibles causalement, sans jamais injecter l'identifiant de composition au décodeur.
4. **S803d** : auditer spécifiquement la justesse des NoteOff associés aux **64 TP** de collision, même si leurs attaques sont correctes.
5. **S803e** : comparer le recouvrement des faux positifs d'une politique à l'autre, sans utiliser de frame future.

**Précision méthodologique importante :** le modèle tiers a suggéré une possible circularité dans l'appariement des Onset. L'implémentation `scripts.s613_train_strict_live.event_score` utilise un appariement optimal hongrois sur **hauteur et différence de temps**, pas les probabilités de confiance des têtes. Cette inquiétude précise n'est donc pas confirmée par le code vérifié. Le modèle tiers n'a pas lu le code : ses autres hypothèses demandent aussi vérification, elles ne constituent pas des résultats.

L'avis ne mesure aucune amélioration CAL32, aucune latence physique ni aucun gain de F1. La référence S708 est maintenue. Ce septième conseiller externe n'est **pas** l'exécution des six agents personnalisés ChatGPT Work, mais un contrôle scientifique LLM complémentaire. Pour les nouvelles séries, le workflow scientifique doit appeler explicitement `.github/workflows/note-reusable-seventh-adviser.yml` après avoir produit six artefacts PASS ; une série n'est pas considérée comme ayant reçu cet avis tant que le run externe et l'artefact n'ont pas réussi.

## Verdict

**Terminée, non promue.** La relecture reproduit les comptes FIT80 S800 et tous les comptes et métriques des trois politiques S801. Six audits et consensus PASS. Aucune conception ajoutant lookahead, confirmation, buffering ou retard des NoteOff n'est recevable ; aucune amélioration CAL32 n'est mesurée.

## Prochaine action

Étudier les sous-groupes de faux positifs observés, surtout SS1/Jazz1, sans chercher un simple délai de confirmation. Les distributions initiales des probabilités des têtes se recouvrent fortement entre vrais et faux positifs ; une règle de veto naïve risquerait donc de supprimer de vraies notes. Proposer seulement ensuite un éventuel veto par caractéristiques audio présentes à l'instant de création. Tout candidat devra d'abord satisfaire les six métriques FIT, puis une validation indépendante CAL32 et le test physique S793.

## Automatisation 7/7 S802

La configuration scientifique S802 a été modifiée pour exécuter les six rôles indépendants et le modèle NVIDIA GLM simultanément après le job FIT80, sans déclenchement manuel. Le job final rejette l'exécution si un seul des six audits échoue ou si l'avis externe réel manque/échoue. Les six mesures GuitarSet et la référence S708 restent inchangées. **Validation réelle obtenue** : [run scientifique automatique #39359182007](https://github.com/Andriamarosoa/note/actions/runs/39359182007), **SUCCESS**, commit `5fb30f8c159e5c7a9c4bd045880fca95b1297579`. Six rôles indépendants PASS en parallèle de `z-ai/glm-5.3-flash` (réponse NVIDIA réelle archivée), puis `NOTE_SEVEN_OF_SEVEN_PASS S802 z-ai/glm-5.3-flash promotion=false` dans le job final. L'artefact `S802_SEVENTH_NVIDIA_INDEPENDENT_REVIEW` et `S802_SIX_ROLE_CONSENSUS` sont attachés à ce **même run** ; absence ou échec de NVIDIA bloque 7/7. L'ancien workflow d'appel NVIDIA séparé a été supprimé pour éviter des doubles facturations. Les runs historiques #39357068590 et #39358229144 restent des références distinctes et intactes.
