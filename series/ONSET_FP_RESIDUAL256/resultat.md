# Résultats vérifiés — Onset FP résiduels MLP69 natif256

**Workflow final [#39772165153](https://github.com/Andriamarosoa/note/actions/runs/39772165153) : SUCCESS, deux jobs (tests techniques et reconstruction signée des poids), artifact [#12422637841](https://github.com/Andriamarosoa/note/actions/runs/39772165153/artifacts/12422637841)**.

## Résultats du vrai replay audio original, déjà vérifié dans les runs antérieurs

- S613 : TP=8099, FP=4600, FN=3168, F1=67,5874155 %.
- Trois têtes MLP69 sans rétroaction : TP=8054, FP=4411, FN=3213, F1=67,8745997 %.
- Quatre têtes FNRecovery initial : TP=8160, FP=4605, FN=3107, F1=**67,9094541 %**; la quatrième ajoute net **106 TP, 194 FP**, 32 pistes régressent.
- Quatre têtes entraînées sur 100% des négatifs TRAIN : TP=8100, FP=4500, FN=3167, F1=67,87614698 %, **échec** du critère de gain #141; ne pas promouvoir.

## Résultats EXACTS du nouvel audit d'événements Onset MLP69

La CI recharge cinq MLP69 entraînés à PCM256, vérifie leurs SHA256 et splits 3TRAIN/1CAL/1OUTER, et retrouve les **12 699** NoteOn S613 originaux, puis **8 054 TP, 4 411 FP** conservés par les seuils de la tête d'erreur. Source : artefact [#12411673219](https://github.com/Andriamarosoa/note/actions/runs/39741883968/artifacts/12411673219), déjà créé après replay authentique des 80 WAV PCM256.

| Groupe (chevauchement possible) | TP | FP | Taux faux |
|---|---:|---:|---:|
| Accompagnement | 5619 | **3548** | 38,70% |
| Solo | 2435 | 863 | 26,17% |
| Grave E2–B2 | 602 | 624 | **50,90%** |
| Médium C3–B3 | 3561 | **2145** | 37,59% |
| Aigu C4–E6 | 3891 | 1642 | 29,68% |
| **Parent harmonique natif émis STRICTEMENT AVANT, dans 80 ms (=3528 échantillons)** | 678 | 782 | **53,56%** |
| Sans parent harmonique natif récent | 7376 | **3629** | 32,98% |

**Résultat important et négatif pour les veto naïfs :** masquer toutes les attaques du groupe « parent plus grave récemment attaqué » enlèverait 782 FP **mais aussi 678 TP**, F1 évènementiel fictif **66,2356%** (TP7376, FP3629, FN3891). Ce n'est **pas** une amélioration : cette opération est REFUTÉE comme règle de sélection, et n'a pas été promue ni rejouée dans le MIDI. Ces chiffres concernent le groupe EXACT **80 ms**. Une exploration initiale avec 14 hops (~81,27 ms) donnait 784 FP et 688 TP ; elle a été remplacée par le seuil exact 3528 échantillons, d'où la légère correction des nombres.

- La principale source des FP est **déjà dans les trois premières têtes** : 4411 avant FNRecovery. La quatrième tête seule ne peut pas supprimer les erreurs acoustiques que Main et ErrorPrediction acceptent.
- Beaucoup des FP ne se trouvent pas dans le groupe « parent harmonique récent » : **3629 sur 4411**. Il est donc incorrect d'attribuer tous les FP aux harmoniques, aux accords ou à un bug de réarmement.
- Le médium C3–B3 regroupe 2145 FP, la basse E2–B2 a un taux d'erreur particulièrement haut, et les accompagnements concentrent **80,43%** des FP (3548/4411).
- Les annotations GT, dont les relations harmoniques avec les vérités terrain, sont exclusivement postérieures aux prédictions. Les 74D exportés pour une future expérience ont 69D acoustiques d'origine, quatre âges de parents tirés des **anciens NoteOn S613 uniquement**, et P(correct) de la tête MLP69, **aucune GT**.

## Nouvelle hypothèse et statut

L'association entre FP et parents harmoniques récents est une observation réelle, **pas une preuve causale que de nouvelles features résoudront les FP**. Hypothèse H_FP_PARENT_TIMELINE_01 [issue #142](https://github.com/Andriamarosoa/note/issues/142) : la temporalité causale des attaques parentes devrait apporter au réseau neuronal un signal discriminant supplémentaire. Le test prévu doit comparer un vrai réseau sur 74D à un contrôle 69D, à graines, splits, seuils de calibration identiques, suivi d'un **vrai replay audio**. En cas de non-amélioration, audit **REFUTED_OR_INCOMPLETE** conformément à [règle #140](https://github.com/Andriamarosoa/note/issues/140).

**Pas de nouveau réseau entraîné ni de nouveau F1 produit par cette audit descriptive.** Aucune promotion, aucun poids1024, aucun futur audio, aucune autre architecture modifiée ni modification du MIDI officiel. Le corpus FIT80 est partiellement vu historiquement par S613/S701 : pas de prétention de généralisation externe complète.
