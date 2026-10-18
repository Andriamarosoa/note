# S733 — Même architecture de validation qu’Exact-K, spécialiste par spécialiste

**Statut : infrastructure seule, non entraînée, non promue.** S708, S707, le Live et TEST restent intacts. Aucune amélioration des métriques n’est affirmée.

## Architecture reprise

Nous reprenons les *mécanismes* de `ClassConditionalAuditSelector` (`scripts/learn_v273_neural_head_selector.py`), qui étaient déjà explorés dans S502–S509 :

- banque de propositions issues de têtes acoustiques, masques de disponibilité par tête et compatibilité K0–K6 ;
- attention **neuronale** entre ces têtes, poids de sélection appris **pour chaque K** ;
- actions `KEEP`, `ADOPT` ou `SUPPRESS` (maintenir, adopter ou rejeter la correction) pour une seule cible ;
- prédiction supervisée du risque de *correction* et du risque de *régression*, profils établis **uniquement sur FIT en OOF** ;
- poids Exact-K de S708 **figés** durant le travail sur Onset ; seule la distribution K prédite est une feature causale, jamais le vrai K.

`src/causal_note/s733_one_target_selector.py` constitue un **adaptateur PyTorch expérimental** de ce principe, et non le même checkpoint TensorFlow entraîné. Aucun poids S733 n’est encore appris ; le passage des décisions au décodeur MIDI reste à tester.

## Entraînement dans l’ordre, jamais simultanément

1. **Onset uniquement** : garder Sustain, Release, Exact-K, Offset S707 et le décodeur figés. Plusieurs têtes peuvent proposer une attaque (dont S731 et S708), le spécialiste apprend à conserver ou corriger sa décision.
2. **Sustain uniquement**, après gel d’un Onset qualifié. Mesurer interruptions prématurées, chevauchements et faux maintiens.
3. **Release**, puis **Offset S707 séparément**, après gel des étapes précédentes ; contrôler les fins correctes, pas simplement la précision des NoteOff.
4. **Identité/pitch des notes**, une fois les générateurs de véritables notes candidates audités ; aucun sélecteur ne peut retrouver une note absente de toutes les propositions.
5. **Exact-K** reconsidéré isolément seulement lorsque les spécialistes précédents sont validés.

**Précision :** il n'existe pas six têtes correspondant mécaniquement aux six scores. Exact-K actif, F1 hauteur/trame et F1 note complète sont des **métriques dérivées** des événements. On entraîne **une seule fonction à la fois**, mais on recalcule systématiquement les **six métriques** après chaque modification.

## Garde-fous et protocole mesuré

- Générer les prédictions de toutes les têtes de manière causale et **sans labels**, puis aligner les annotations de FIT80 pour préparer la supervision des corrections/régressions.
- Calculer les profils OOF par enregistrement/composition, sans jamais donner au réseau les labels de la prise évaluée.
- Sur CAL32 comparer les événements MIDI réellement émis, les six scores, la matrice Exact-K **K0 à K6**, les notes retenues/perdues, les 151 attaques d'octave et le coût CPU.
- Geler la décision avant les tests HELD32 et DEV12 ; ces derniers ont été historiquement consultés et **ne constituent pas un véritable test entièrement neuf**.
- **Zéro promotion** si un des six indicateurs ou les attaques longues/offsets historiquement corrects régressent ; expliciter le compromis si les objectifs sont incompatibles sur les échantillons.
- Garder TEST fermé et ne pas brancher LIVE sans validation de la latence, de la stabilité et du son réel au micro.

## Pourquoi cette voie plutôt que répéter S731/S732 ?

S509 a déjà entraîné un sélecteur d’identité de note mais a montré des régressions (voir les audits S505–S510). S732 entraînait une nouvelle vraisemblance Onset, mais la correction positive restait multipliée par le `count_room` figé de S708 et n’a généré **aucun NoteOn supplémentaire** sur CAL. On doit **tester le passage proposition → décision → MIDI**, et apprendre un veto au bon endroit, pas seulement modifier un posterior que le sélecteur ignore.

Références : `analysis/README_V273_S502_S504_NEURAL_NOTE_HEAD_SELECTION_ARCHITECTURE.md`, `analysis/README_V273_S505_S510_REAL_TRAINED_NEURAL_NOTE_HEADS_SONGHELD.md`, `src/causal_note/s708_cardinality_selection.py`, `src/causal_note/s731_auditory_scene.py`, `src/causal_note/s732_source_discriminator.py`.

**Résultats validés à ce stade : uniquement tests d’intégrité logiciels.** FIT, CAL, HELD et DEV de S733 n’ont pas encore été exécutés, donc aucune performance S733 n'est disponible.
