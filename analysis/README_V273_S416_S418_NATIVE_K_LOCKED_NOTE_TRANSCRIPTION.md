# S416–S418 — Nouvelle chaîne native de notes MIDI **sans YourMT3+ à l'inférence**

Date : 13 octobre 2026. **Architecture expérimentale, aucune promotion avant résultats.** Le K de S80 est juridiquement/techniquement verrouillé par une égalité exacte par ligne, non par seuil numérique.

## Principe : K n'est ni une harmonie ni un état de note

- **K0** = zéro NOUVELLE attaque, même si des harmoniques et notes antérieures continuent. Un NoteOff peut survenir dans un intervalle K0.
- K1..K6 = exactement 1..6 nouvelles notes produites sur l'événement d'attaque.
- Le nombre K figé est toujours celui de `S80_S397_K5_GUARDED`, **jamais** le K réel JAMS et jamais le comptage suggéré par le modèle de hauteurs.
- Les nouveaux systèmes ne remplacent pas le modèle S80 et ne changent aucune prédiction de celui-ci. L'intégrité sera prouvée par relecture indépendante des notes MIDI **exportées** : chaque `source_row_id` doit émettre exactement son K.

## S416 — réseau natif pour l'identité MIDI

Fichier : `scripts/s416_native_Kfirst_temporal_MIDI_note_decoder.py`.

- Entrée `sequence42x49` : 42 instants et 49 candidats MIDI, sources harmoniques originales du projet + `s80_K` prédit.
- `NativeKTimePitchNet` : convolutions 2D temps/fréquence, résumés physiques pré-/pendant-/après-attaque, vecteurs par hauteur MIDI, intégration précoce de K par embedding/FiLM, **attention entre les hauteurs** puis 49 probabilités de nouveaux pitchs.
- Décodeur `choose_pitches(probabilités, K)` fixé à 0..6, sorties MIDI 40..88.
- Quatre réseaux appris sur **autres compositions GuitarSet**, avec seules les annotations JAMS de l'entraînement : **aucune prédiction, aucun poids, aucun token ni aucune vérité YourMT3+** chargés en entraînement ou en inférence.
- Comparison fixe : S402, 36,76 % d'accords poly complets (ancienne tête de notes). Le nouveau réseau ne doit être retenu que si son gain est prouvé sur les compositions retenues à l'écart.
- Exécution : [run GitHub S416 #38381449092](https://github.com/Andriamarosoa/note/actions/runs/38381449092) (résultat à lire dans son fichier `report.json` : ne pas inventer un pourcentage).

## S417 — Offset natif sur flux audio intégral

Fichier : `scripts/s417_native_S416_MIDI_pitch_plus_continuous_WAV_note_off.py`.

- À chaque nouveau pitch de S416, conserver **Onset S407** et ouvrir un état de note distinct ; source K de S80 inchangée.
- Extraire sur la véritable piste entière GuitarSet MIC des STFT toutes les 10 ms, et ses harmoniques, croissance et décroissance jusqu'à 3,5 s **y compris en K0**, à l'aide de l'extracteur S411.
- Construire un vecteur temporel `59D` par note + `3D` de compétition/réattaque = 62 variables acoustiques et structurelles ; prédire la durée de cette note plutôt qu'un seuil Offset anonyme.
- Comparer deux pertes de régression déclarées AVANT examen des folds tenus à l'écart ; quatre plis séparés par composition, cibles `note_midi` seulement depuis les autres compositions.
- Réduire la probabilité des Offsets trop courts en exploitant les phases de résonance. Attention : aucun comportement causal temps réel n'est revendiqué (accès au futur jusqu'à 3,5 s).
- Exporter de vrais événements MIDI avec `source_row_id`, `pitch`, `onset_sample`, `offset_sample`. Vérifier K par **relecture du JSONL** indépendamment des tableaux de sortie.
- Compare S408 (offset actif choisi par l'utilisateur, F1 note complète 38,14 %), S412 (challenger 41,83 %) et S415 teacher-assisté (67,56 %), **sur les mêmes 190 pistes et JAMS**.
- [workflow GitHub S417](https://github.com/Andriamarosoa/note/actions/workflows/v273-s417-native-complete-MIDI-K-locked.yml) attend explicitement la réussite de S416 ; il ne réinvente pas le comptage.
- La source MIC continue n'est **pas** la source pickup mix des entrées initiales de S80/S258. Cette hétérogénéité est mentionnée dans la comparaison.

## S418 — Inférence native réutilisable sur nouvelles entrées acoustiques S80

Fichier : `scripts/s418_native_Kfirst_MIDI_head_any_new_S80_audio.py`.

- Charge les quatre propres checkpoints PyTorch S416 vérifiés, jamais YourMT3+, et prend exactement `row_id, sequence42x49` et `row_id, s80_K` d'une nouvelle source.
- Moyenne les probabilités des hauteurs puis ne génère que K notes, K0 = liste vide.
- Écrit un CSV MIDI par source et un NPZ de notes avec contrôle indépendant de cardinalité basé sur le **CSV réellement exporté**.
- **Attention méthodologique** : sur GuitarSet, l'ensemble des quatre modèles contient plusieurs réseaux ayant vu chaque composition ; on ne doit donc **pas** lui attribuer les scores de la prédiction hors fold S416. Tester l'ensemble sur nouvelles compositions seulement.
- Il ne calcule ni K à partir du WAV brut, ni les NoteOff ; ce dernier rôle appartient à S417.

## Conditions nécessaires à une promotion

1. **0 régression Exact-K garantie par les notes réellement exportées** sur 59 309 / 59 309 événements ; bilan par K0..K6 identique aux S80 53 529/59 309 global, 6 054/7 385 poly.
2. Comparaison Pitch+Onset F1 et accord exact vs S402, puis note complète et Offset à ±50 ms vs S408/S412, y compris classes `>=1s`.
3. Audit corrections et régressions : ne pas masquer les notes longues réussies par des erreurs sur les courtes. On ne change aucune référence choisie si le modèle régress.
4. **Validation réellement indépendante de la chaîne entière** : S80 a été entraîné/sélectionné partiellement sur GuitarSet ; les folds des têtes seulement ne corrigent pas cet overlap.
5. **S415 n'est pas un modèle autonome** : il utilise directement la sortie YourMT3+ à l'inférence. Il sert seulement d'étalon d'architecture, pas de résultat scientifique propre au réseau S416–S418.
