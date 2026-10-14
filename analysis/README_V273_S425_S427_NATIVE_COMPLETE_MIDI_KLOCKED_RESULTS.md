# S425–S427 — Amélioration des notes MIDI complètes (Live différé)

**14 octobre 2026.** Le développement concerne uniquement la **transcription MIDI hors ligne** : hauteur de note, instant NoteOn et NoteOff. Le Live causal et ses tests [S424](https://github.com/Andriamarosoa/note/actions/runs/38512810211) sont conservés sans modification. **Le comptage S80 est strictement verrouillé.**

## Résultat : S425, meilleur challenger natif mesuré

| Mesure GuitarSet, 190 pistes / 34 954 notes réelles | S420 | S422 | **S425** |
|---|---:|---:|---:|
| Prédictions NoteOn MIDI | 33 475 | 33 475 | **33 475** |
| Notes avec bonne hauteur et Onset ±50ms | 25 395 | 25 395 | **25 395** |
| NoteOff correct parmi ces notes | 16 140 | 16 340 | **16 543** |
| Précision conditionnelle NoteOff ±50ms | 63,56 % | 64,34 % | **65,14 %** |
| **F1 note MIDI complète ±50ms** | 47,17 % | 47,76 % | **48,35 %** |
| MAE NoteOff sur notes appariées | 91,03ms | 88,40ms | **87,60ms** |
| **Régressions Exact-K** | **0** | **0** | **0** |

**Gain S425 vs S422 : +203 notes complètes à ±50ms, soit +0,59 point de F1**, sans modification des 59 309 prédictions K0..K6 de S80, des hauteurs MIDI, des Onsets S407, des fichiers historiques du Live ou des références sélectionnées.

- [Code S425 — garde de désaccord NoteOff par note et composition](https://github.com/Andriamarosoa/note/blob/codex/s276-idmt-song-disjoint-candidate-audit/scripts/s425_songheld_select_S422_S420_offsets_no_K_regression.py)
- [Run S425 #38513725784](https://github.com/Andriamarosoa/note/actions/runs/38513725784) **success**, artefact `11870639014`. Prédictions `S425_K_LOCKED_59309_native_NoteOff_frozen.npz`, MIDI JSONL `S425_K_LOCKED_native_full_MIDI_note_events.jsonl`, rapport et 4 modèles `joblib`.

### Architecture réellement exécutée

S425 conserve les hauteurs sélectionnées par S419, le K S80 et les Onsets S407 déjà figés. Il apprend, par note, **quand utiliser la durée S420 et quand utiliser celle de S422**, à partir d'indices de désaccord des durées prédites, MIDI, K prédit, temps jusqu'aux attaques voisines et réattaques de même hauteur. Aucun système YourMT3+ n'intervient.

Les exemples de choix positif/négatif sont des notes où **l'un des deux Offsets est correct à ±50ms et pas l'autre**. Seules les annotations JAMS des **autres compositions** sont utilisées pour apprendre le sélecteur. Le seuil de décision de chaque fold est choisi sur une subdivision de compositions de son propre train. En l'absence de gain suffisant sur cette validation interne, le sélecteur s'abstient.

Le modèle **ne crée jamais de NoteOn** et **ne change jamais K** : K0 = 0 nouvelle attaque, mais les notes précédentes peuvent continuer et produire des NoteOff.

**Limite scientifique cruciale :** bien que la dernière tête et le seuil soient évalués sur des compositions différentes, les prédictions S420/S422 amont ont été construites par des modèles cross-fittés qui ne sont pas complètement réentraînés de façon imbriquée pour la méta-sélection. Et S80 a une exposition historique à GuitarSet. **Ce résultat est un gain de développement mesuré, pas une validation indépendante bout-en-bout.**

## S426 — audit indépendant sur les fichiers MIDI gelés

[Code S426](https://github.com/Andriamarosoa/note/blob/codex/s276-idmt-song-disjoint-candidate-audit/scripts/s426_audit_S425_S422_frozen_noteoff_same_GT_K_locked.py) ; [run #38513899842](https://github.com/Andriamarosoa/note/actions/runs/38513899842) **success**, artefact `11870763918`.

Relit les vrais fichiers **JSONL de notes complètes S422 et S425** et les compare aux **mêmes 34 954 identités de note de vérité**. Les données ne sont ni réentraînées ni corrigées.

- **652 corrections** de notes complètes,
- **449 régressions**,
- **+203 notes complètes nettes**,
- **zéro** régression K après relecture des **33 475 notes MIDI réellement exportées** contre les **59 309 valeurs originales de S80**.

| Durée vraie d'une note MIDI | N notes GT | S422 note complète | S425 | Gain net |
|---|---:|---:|---:|---:|
| moins de 100 ms | 2 263 | **670** | 624 | **−46** |
| 100–499 ms | 28 212 | 15 024 | **15 255** | **+231** |
| 500–999 ms | 3 579 | 577 | **595** | **+18** |
| au moins 1000 ms | 900 | 69 | 69 | **0** |
| **TOTAL** | **34 954** | 16 340 | **16 543** | **+203** |

Par fold de composition : fold 0 = 0, fold 1 = +44, fold 2 = **+176**, fold 3 = −17. Cela confirme une amélioration du fold 2 précédemment déficient, sans dissimuler les régressions locales sur le fold 3 et les notes très courtes.

Le grand verrou des longues notes reste entier : **69/900 notes d'au moins 1s complètes** (69/736 appariées par pitch/Onset), malgré le suivi temporel déjà implémenté dans S422.

## S427 — garde supplémentaire courte durée : rejetée

[Code S427](https://github.com/Andriamarosoa/note/blob/codex/s276-idmt-song-disjoint-candidate-audit/scripts/s427_train_only_short_note_offset_safety_guard.py) ; [run #38514140255](https://github.com/Andriamarosoa/note/actions/runs/38514140255) **success**, artefact `11871068865`.

Une famille **pré-déclarée de 24 règles** fondées sur les **durées prédites** uniquement (jamais durées GT en test) tente de restaurer S420 ou S422 sur les notes probablement très courtes. Les autres compositions servent à choisir une règle ; activation seulement si au moins +15 notes correctes sur ces compositions.

**Verdict honnête :** aucune règle n'atteint le gain train minimal sur les quatre folds ; **abstention complète**, zéro note changée et score strictement identique à S425 : **48,35 %**. Ne pas promouvoir S427 comme un nouveau modèle meilleur. Le premier workflow S427 avait échoué sur une assertion CI inversée `no_official_promotion`, après la génération correcte des sorties ; le [second workflow](https://github.com/Andriamarosoa/note/actions/runs/38514140255) a corrigé le test et terminé **success**.

## Références et prochaines priorités

Références sélectionnées laissées intactes :
- Exact-K **S80/S397** : 53 529/59 309 = 90,25 % global, 6 054/7 385 = 81,98 % poly (jeu de développement recoupant historiquement GuitarSet).
- Hauteurs MIDI **S402** : référence active, S419 meilleur challenger de combinaisons de notes.
- Offsets **S408** : référence active choisie par l'utilisateur.
- Transcription MIDI complète **S425 = 48,35 % F1** : meilleur **challenger natif de développement**, non promu officiellement.
- Live historique et pont MIDI Live S424 **préservés**, mais **développement temps réel différé** selon la priorité utilisateur.

Priorités de recherche sur l'identité des notes et leur persistance : 
1. Trajectoires spectrales de vraies voix/cordes afin d'améliorer les notes longues au-delà d'une seconde et distinguer les réattaques (où S425 n'apporte encore aucun gain).
2. Apprentissage plus direct de la hauteur et de l'état actif NoteOn/NoteOff, avec K S80 en entrée **et un décodeur cardinalité bloqué à K**, plutôt que correction de durées seules.
3. Une comparaison réellement **imbriquée et indépendante**, de la source S80 jusqu'aux MIDI sur de nouvelles compositions et une autre source audio, avant toute promotion.
