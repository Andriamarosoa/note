# S518–S520 — Intégration de S516 comme 14ᵉ tête génératrice dans le sélecteur S509

**15 octobre 2026. Recherche expérimentale ; les références officielles S80, V88, S408/S425, le système Live, les onsets et les offsets restent inchangés.** Aucune promotion automatique ou mesure de F1 événementiel complet sans annotations d'événements négatifs.

## Filiations réellement utilisées

- **S505** : corpus réel de 265 attaques annotées associées à 294 groupes musicaux canoniques, répartis en 5 enregistrements guitare IDMT (S80 complet, **45/45** onsets vrais appariés) et 3 enregistrements piano MAESTRO (V88 original, **220/322** onsets vrais appariés, **102** onsets vrais sans groupe).
- **S509** : réseau d'attention par hauteur, 13 têtes H0–H12 (S502). Les probabilités prédites ne concernent **que les pitches déjà proposés par ces têtes**.
- **S515/S516** : 49 hauteurs candidates MIDI40–88 par vrai WAV, 11 caractéristiques frame/harmonique/attaque par hauteur ; modèles ExtraTrees par morceau exclu, permettant de proposer de nouvelles hauteurs réellement absentes des treize têtes.
- **S518** : fusion **inférence-only sans entraînement** de la probabilité S509 lorsqu'elle existe et S516 comme 14ᵉ source ; prédictions d'origine SHA-vérifiées et règles fixes prédéclarées.
- **S519** : modification réelle de l'architecture d'attention de S509 pour **14 têtes**, incluant la probabilité S516 en tant que vote continu, plus les 11 observations spectrales originales par hauteur. Réentraînement neuf du générateur acoustique et du sélecteur neuronal **à chaque fold**, sans réutiliser des poids S509/S516 calibrés sur le morceau tenu à part. Contrôle imbriqué indispensable pour empêcher que les probabilités de métadonnées d'entraînement proviennent d'un générateur ayant appris sur le morceau d'évaluation.
- **S520** : relecture indépendante des probas S519, 8 générateurs, 8 réseaux neuronaux (16 empreintes SHA), classe K0–K6, noms des notes MIDI, corrections/régressions par source.

## S518 — premières mesures obtenues

[Run S518 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38643457520).

| Variante S518 | Guitare IDMT : accords exacts /45 | Piano MAESTRO : accords exacts /220 | Guitare : fixes / regressions | Piano : fixes / regressions |
|---|---:|---:|---|---|
| BP source original | 27 | 125 | — | — |
| S516 seul p≥0,70 | **30** | 128 | 4 / 1 | 6 / 3 |
| Fusion simple des probabilités S509+S516 | 28 | **130** | 2 / 1 | **9 / 4** |
| Fusion ancrée au BP | 27 | 124 | 1 / 1 | 0 / 1 |
| Fusion très prudente | 26 | 124 | 0 / 1 | 0 / 1 |

À noter : les trois probabilités proviennent de modèles déjà formés sur des morceaux voisins. Les variantes ont été choisies avant ouverture des annotations pour le score, mais ce **n'est pas** une véritable sélection neuronale fusionnée encore entraînée. En piano, 130/220 accords exacts est le meilleur S518 ; il n'excède S516 que de 2 accords, avec quatre régressions. La guitare préfère S516 seul. **Aucune de ces politiques ne mérite promotion**.

F1 des notes sur les attaques annotées ≠ F1 événements NoteOn complet, car les 102 attaques MIDI non appariées et les candidats non vérifiés (négatifs inconnus) rendent illusoire un 80 % global. Comparer aussi les F1 de hauteurs et les accords entiers, pas seulement Exact-K.

## S519 — correction de la conception : véritables têtes combinées et validation imbriquée

`scripts/s519_train_nested_S509_attention_with_S516_raw_generator_14head.py`.

### Pourquoi un simple mélange S518 est insuffisant

La qualité S516 varie suivant le timbre, l'instrument et les confusions fondamentales/harmoniques ; S509 est très conservateur et n'a pas accès à des pitch absents de ses 13 têtes. Un vote fixe `0.5 S509+0.5 S516` ne sait pas quand protéger une ancienne note ou admettre une nouvelle voix. Les têtes **doivent coopérer pour chaque hauteur**, et disposer d'un profil de risque issu de l'apprentissage, pas d'une confiance supposée universelle.

### Entrée du réseau

Pour chaque attaque canonique et chaque pitch MIDI40–88 :
- 13 têtes source, chacune 16 indices acoustiques canoniques, 4 types, `head_mask`, proposition oui/non ;
- 14ᵉ tête S516 : 11 observations acoustiques brutes (harmoniques, nouveauté, sustain), cinq indices du contexte, type de tête et probabilité **apprise** en laissant le morceau exclu ;
- contexte global source + spectral + hauteur MIDI + K **prédit acoustiquement** (pas vrai K), soit `[49, 33]`;
- entrée attention `[événements×49, 14, 22]`, produisant une probabilité par hauteur candidate.

Le backbone `compile_model()` vient du vrai réseau d'attention TensorFlow S509 (2 heads). Le module S516 est réentraîné sous le même principe ExtraTrees sur l'audio pour fournir la quatorzième tête ; **les anciens modèles enregistrés S509/S516 ne sont pas directement réutilisés**.

### Validation correcte des poids

Pour chaque composition tenue à part parmi 8 :
1. 7 compositions = train ; la 8ᵉ n'est **jamais accessible** aux modèles ni aux profils générateurs train ;
2. pour chacun de ces 7 morceaux train, entraîner un producteur S516-type sur les **6 autres morceaux train** puis produire ses 49 probabilités hors-apprentissage ;
3. entraîner un générateur S516 sur 7 morceaux train, puis prédire les 49 pitch du morceau tenu à part ;
4. entraîner un réseau S509 avec 14 têtes sur les propositions OOF train et les vraies notes du **train seulement** ; tester une fois sur le morceau tenu à part, sans ajuster les seuils à l'aide de son MIDI ;
5. conserver **8 modèles acoustiques de test et 8 modèles neuronaux d'attention** SHA-frozen ; mesurer à posteriori trois politiques préalables `p≥0,50`, `p≥0,70`, `KEEP+ajout p≥0,85 et S516≥0,75; suppression uniquement si les deux <=0,15`.

Cette validation est plus stricte que le simple rassemblement de prédictions séparément OOF et empêche la contamination du niveau d'entraînement de la fusion. Elle n'élimine cependant **pas** l'exposition historique des modèles amont et des chansons : les 8 fichiers ont été utilisés lors d'anciennes expériences ; il faut un nouveau corpus pour parler de généralisation indépendante.

## S520 — Contrat de non-régression

`scripts/s520_independent_replay_nested_14head_pitch_selection.py` :

- comparer chaque poids enregistré à sa véritable empreinte SHA ;
- reconstruire exactement les notes E3/G3/B3 depuis les probabilités S519 **sans entraînement supplémentaire** ;
- calculer les corrections, régressions, erreurs restant fausses, score de chaque vrai K, exactitudes des notes et accords, F1 hauteurs sur attaques connues, et K S80/V88 d'origine ;
- afficher les notes ajoutées, supprimées, corrigées ou régressées, avec **annotations lues seulement pour l'audit** ;
- interdire la promotion lorsque des régressions persistent, lorsque des attaques vraies ne sont pas couvertes ou lorsque la qualité Live est inconnue.

## Prochain jalon

S519/S520 permettent la fusion des véritables hauteurs. Ils **ne** réparent **pas** les 102 attaques piano manquantes, car ils évaluent des groupes canoniques Basic Pitch. Le futur détecteur *frame+onset* indépendant doit émettre ses propres nouvelles attaques, puis subir une évaluation continue notes+onset avec **négatifs annotés**, permettant un véritable F1 événementiel. Le seuil visé 80 % doit être validé sur des morceaux **jamais utilisés en entraînement ou adaptation**, pas sur les huit titres historiques.

**L'utilisateur a demandé de remplacer la référence après la validation. La référence n'est pas remplacée dans cette série.**
