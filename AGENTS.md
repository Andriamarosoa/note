# Instructions aux agents — NOTE (S713)

Ce fichier coordonne **des sessions d'agents indépendantes** (ChatGPT Work/Codex ou développeurs). Il ne lance PAS d'agents automatiquement. Une issue GitHub = une mission = une branche = une PR DRAFT. Chaque agent lit aussi `docs/audits/S713_CODE_AUDIT_MATRIX.md`.

## Mission commune
Transcription strictement causale guitare PCM → hauteurs MIDI40–88, Onset, Offset, états actifs et Exact-K. Conserver les têtes + sélections natives de NOTE ; `midi` et YourMT3+ sont des comparateurs, pas une greffe de moteur. Ne jamais confondre compteur d'attaques S80 avec cardinalité des notes actives LIVE.

**Références distinctes, pas interchangeables :**
- S80/S397 : Exact-K d'événements d'une architecture historique, sur un protocole distinct.
- S613/S630 : LIVE natif actuellement employé ; ne pas remplacer ses fichiers par défaut.
- S675 : note-off conditionnel 78,08% sur son protocole.
- S699 : tête K shadow à 66,6551% sur GuitarSet DEV (48 970 frames) ; pas intégrée LIVE.
- S708 : meilleur Onset GuitarSet DEV 68,7520% sur 12 morceaux, protocole causal hors ligne, Offset S707 séparé.
- S712 : adaptateur microphone `--s708-model` sur branche expérimental, CI 8 tests OK mais WAV synthétique 0 NoteOn ; pas de mesure de qualité sur un vrai microphone.
- S711 : modèle alternatif plus haut sur HELD Onset, pas promu.

## Invariants NON NÉGOCIABLES
1. **Causalité physique** : décision à `t` calculée uniquement à partir des échantillons déjà reçus ; horodatage de disponibilité jamais avant la lecture audio. Prouver par mutation du futur et partitions PCM irrégulières.
2. **Pas de régression silencieuse** : pour chaque PR, comparer six métriques sur le **même** ensemble/annotations/horodatages : `onset_f1`, `exact_k`, `active_exact_k`, `offset_accuracy`, `full_note_f1`, `frame_pitch_f1`. Ajouter performances accords/solos, notes longues, pitch 40–88, latences CPU/entrée, TP/FP/FN.
3. **Séparation des données** : FIT entraînement uniquement ; CAL sélection d'hyperparamètres ; HELD validation tenue à l'écart ; DEV déjà exposé = diagnostic uniquement ; TEST non ouvert tant que protocole non figé. Aucun accès aux annotations depuis `push`, `step` ou fonctions de prédiction.
4. **Une vérité d'événements** : tous les scores issus du même flux de `note_on`/`note_off` réellement émis, et non de têtes idéales ou d'oracles séparés. N'utiliser ni score K tête brute comme Exact-K décodé, ni Onset time-only comme F1 note+pitch.
5. **Compatibilité** : l'ancienne commande CLI sans `--s708-model`, les références S80 et S675 ne changent jamais dans un audit. Pas de promotion automatique, aucune suppression de workflows historiques.
6. **Falsifiabilité** : toute affirmation chiffrée doit référencer run GitHub, SHA code+checkpoint, corpus, split, critères de correspondance et événements sources. Une CI verte ne prouve pas un F1 ou une latence micro.
7. **Sûreté LIVE** : pas de traitement coûteux dans callback, pas de file infinie, arrêt + NoteOff en cas d'overrun, pas de corruption des événements. Windows/iRig = mesure physique sur poste utilisateur.
8. **Périmètre des modifications** : chaque agent touche seulement ses chemins/branches convenus ; une intégration croisée exige une PR séparée. Ne pas écraser le travail de Work non encore poussé.

## Rôles indépendants et contrat de livraison

### Agent A — Architecture, LIVE et timing
`scripts/live_strict_notes.py`, `src/causal_note/s712_s708_live.py`, `src/causal_note/strict_live.py`, files audio.
Chercher : état de fermeture, continuité PCM, overruns, latence CPU par décision/callback, toutes les options incompatibles, intégration S707, parité offline/LIVE. Livrer au minimum un test WAV réel avec des NoteOn ET NoteOff et rapport p50/p95 + backlog des anomalies.

### Agent B — DSP et harmoniques
`src/causal_note/s706_sharp_spectra.py`, caractéristiques S704/S701, fenêtres FFT.
Chercher : fondamentales graves absentes, aliasing, FFT binning, normalisation, faux partiels/octaves, discrimination notes simultanées, artefacts de phase, exactitude bord de fenêtres. Tests E2/B2/E3, harmoniques seules vs accord réel, stabilité au bloc.

### Agent C — ML, têtes et sélections
`src/causal_note/s705_temporal_heads.py`, `src/causal_note/s708_cardinality_selection.py`, `scripts/s706_train_sharp_temporal_loops.py`.
Chercher : mélange des têtes, poids accords versus solos, collapse K, masques temporels, pertes et sampling, calibration figée, permutation ordre des sélections, under/overcount, state drift. Aucune expérimentation prétendument indépendante à partir de DEV.

### Agent D — Validation, métriques et fuite de données
`scripts/s704_joint_metrics.py`, `scripts/s704_train_joint_loops.py`, `scripts/s708_count_guided_joint_loops.py`, exports de rapports.
Auditer appariements NoteOn-pitch±80ms, Off±50ms, doublons, notes longues, EOF, segmentation par morceau, accumulation frame K, split composition/performer, valeur et incertitude du F1. Refaire 1 petit scénario analytique vérifiable à la main.

### Agent E — Qualité de code et CI
`test/`, `.github/workflows/`, dépendances, outils d'analyse statique.
Chercher erreurs runtime (ex : attribut absent), exceptions et fermeture, contrôle d'entrée de fichier, compatibility PyTorch/NumPy/ONNX, tests CLI réellement exécutés, duplication workflows, artefacts vérifiables, absence d'écrasement des modèles. Proposer tests ciblés économiques, pas 848 workflows relancés.

### Agent F — Relecteur indépendant RED TEAM
Lecture seule des PR A–E. Contre-exemples et chemins d'échec ; refuser toute promotion sans reproduction séparée. Le reviewer n'a pas le droit de choisir l'hyperparamètre sur le jeu qu'il certifie.

## Format exigé pour chaque conclusion
- Sévérité : `P0` blocage/erreur de causalité ou corruption ; `P1` erreur de décision/mesure ; `P2` qualité/performance ; `P3` documentation.
- Localisation exacte fichier + lignes ou lien de commit, précondition reproductible.
- Effet : lequel des six scores / latence / intégrité événementielle risque d'être affecté.
- Preuve minimale : test qui échoue avant correction, puis réussit après, ou `NON REPRODUIT`.
- Changement minimal, résultats comparés même split, et verdict `PASS / FAIL / UNKNOWN`.
- Ouvrir une PR DRAFT seulement ; aucun merge ni mise à jour d'une référence sans revue indépendante.

## Coordination
Agent D définit d'abord le protocole de score et les jeux tenus à l'écart, puis A/B/C travaillent en branches distinctes. E automatise, F vérifie. Les agents ne peuvent pas s'auto-attribuer un passage à 80% sur des données déjà explorées. Si la mesure n'est pas disponible, écrire « non mesuré ».
