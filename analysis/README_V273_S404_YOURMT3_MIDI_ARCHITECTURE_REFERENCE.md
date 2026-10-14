# S404 — Audit d'architecture YourMT3+ + `Andriamarosoa/midi` et nouveau prototype K-first

**Date : 13 octobre 2026.** Statut : **architecture vérifiée dans les sources et module PyTorch expérimental écrit**. S80/S397 reste la référence de comptage, S402 la dernière tête de notes **entraînée/mesurée**, jusqu'à une comparaison rigoureuse.

## 1. Sources inspectées — pas seulement les README

**YourMT3+ (Sungkyun Chang, Emmanouil Benetos, Holger Kirchhoff, Simon Dixon; MLSP 2024)** :
- Papier https://arxiv.org/abs/2407.04822 , DOI 10.1109/MLSP58920.2024.10734819 .
- Dépôt officiel d'annonce : https://github.com/mimbres/YourMT3 .
- **Véritable code pré-release indiqué par l'auteur** : https://huggingface.co/spaces/mimbres/YourMT3/tree/main/amt/src/model . L'auteur le renvoie depuis https://github.com/mimbres/YourMT3/issues/2#issuecomment-2255643217 .
- **Fichier source lu** : https://huggingface.co/spaces/mimbres/YourMT3/blob/main/amt/src/model/ymt3.py , en particulier la classe `YourMT3` et `forward`/`inference`, `spectrogram`, `pre_encoder`, `PerceiverTFEncoder`, `MultiChannelT5Decoder`, `task_cond_dec_generate`, `lm_head`.
- Les modules du même code sont `spectrogram.py`, `conv_block.py`, `perceiver_mod.py`, `ff_layer.py`, `projection_layer.py`, `t5mod.py`, `lm_head.py`; pour MIDI, cf. `amt/src/utils` et `task_manager`.
- **Pipeline vérifié :** waveform → log spectrogramme → convolutions/pre-encodeur → attention temps/fréquence Perceiver-TF (option MoE FFN) → projection latente multicanal → décodeur T5 autoregressif → séquences de tokens événementiels MIDI. Le choix du décodeur multicanal est important pour les instruments et annotations partielles. Le pré-encodeur traite du contexte audio plus vaste que les 42 trames locales de S399/S402.
- **Nuance** : le décodeur MT3 ne demande pas d'ordinaire un K (nombre exact de nouvelles notes) externe ; il produit des notes, instants et programmes. Notre **K-first est une extension propre à note**, pas une copie exacte de YourMT3+.

**Dépôt original de l'utilisateur `Andriamarosoa/midi`** :
- https://github.com/Andriamarosoa/midi/blob/main/docs/POLYPHONIC_V2.md
- https://github.com/Andriamarosoa/midi/blob/main/src/polyphonic/model.py : le modèle `build_polyphonic_model` consomme 4096 échantillons (44,1k) et masque temporel, deux convolutions causales, TCN à dilatations 1/2/4/8, `frame` multi-label (sigmoïde), `onset` multi-label (sigmoïde, branche spéciale 512 derniers échantillons), `harmonic_amplitude` et `harmonic_offset_cents` (37 hauteurs MIDI 40..76, 20 partiels chacune).
- https://github.com/Andriamarosoa/midi/blob/main/src/polyphonic/decoder.py : état global des notes par hauteur, seuil d'attaque, accumulation de votes et arrêt avec hystérésis ; relation harmonique `ratio = 2**((pitch_up - pitch_down)/12)` et écart toléré en cents ; une candidate proche d'une harmonique requiert une preuve d'attaque propre avant d'être considérée comme un nouveau NoteOn.
- https://github.com/Andriamarosoa/midi/blob/main/src/process/harmonic.py : fondamentale MIDI -> fréquence, fenêtre Hann + FFT, recherche locale des partiels autour de `h*f0`, interpolation du pic, amplitude relative et écart en cents.
- https://github.com/Andriamarosoa/midi/blob/main/readme/notes/HARMONIQUES_DISCUSSION.txt : les harmoniques **ne sont jamais des vérités de nouvelles notes** et leur trajectoire d'attaque/décroissance compte ; ne pas interpréter comme vraie fondamentale un pic masqué ou un harmonique d'une autre note.
- La supervision harmonique ne provient que des notes décomposables par les six canaux GuitarSet, **masquée** pour les mélanges GAPS, car attribuer les pics du mélange aux bonnes fondamentales sans séparation serait un faux label.
- **Attention** : le modèle produit stable de ce dépôt est décrit comme monophonique pour MIDI 40..76 ; V2 polyphonique reste expérimental. Ne pas importer ses % de top-1 monophonique comme des preuves de détection d'accords K2..K6.

## 2. Comparaison de conception

| Module | Entrées effectivement exploitées | Décisions produites | Défaut/avantage pertinent |
|---|---|---|---|
| S399 | 42×49 activations acoustiques ; K seulement au top-K | 49 scores heuristiques | N'apprend pas les relations notes/harmoniques ; 17,49 % vrais accords poly complets sur GuitarSet |
| S402 (référence actuelle notes) | 352D = audio projeté + K one-hot + K/K² + interaction | 49 probabilités de notes puis sélection de K | K agit réellement sur les probabilités ; mais pas de structure explicite de durée, d'attaque, d'harmoniques ou de décodage d'événements ; **36,76 %** vrais accords poly complets, songholdout pour la tête S402 seulement |
| YourMT3+ | Spectre temps-fréquence et contexte étendu | Tokens événements MIDI multicanaux | Structure musicale riche mais réseau/génération bien plus lourds et sans K externe |
| `Andriamarosoa/midi` V2 | 4096 samples audio causal, masque | Sigmoïdes `frame`, `onset`, harmoniques et offset | Enseigne séparation des notes actives/nouveaux onsets et supervisons harmoniques justifiées |
| **S404 prototype** | S80 K + 42×49 trajectoire acoustique en entrée d'une attention temporelle | 49 sorties note/onset/frame + 49×20 amplitudes/cents, contrainte finale K | Fusion structurelle inspirée des 2 références, encore **non entraînée / non évaluée en qualité** |

## 3. Code S404 implémenté

Fichier : `src/causal_note/k_first_multihead_notes_s404.py`.

```text
audio original 42 × 49                         S80 K∈{0..6}
          |                                           |
          +----------> K embedding / FiLM <-----------+
                           | (AVANT probabilités MIDI)
               encodage temporel Transformer
                           |
              queries apprises par pitch (49)
                  cross-attention temporelle
                    /          |         \\
                   /           |          \\
          frame_logits     onset_logits   harmonic_amp + cents
              49              49               49 × 20
                    \         /
                  note_logits 49
                        |
               sélection de K notes (0..6)
```

- `KFirstMultiHeadNoteNet.forward(harmonic_time, predicted_k)` reçoit **K dès l'encodeur** (embedding et modulation FiLM) et l'utilise aussi dans les 49 requêtes des notes ; la sélection top-K **n'est pas son unique utilisation**.
- `multitask_loss` calcule des pertes multi-label indépendantes pour notes actives et nouvelles attaques. Le troisième terme est `masked_binary_loss` appliqué seulement aux harmoniques dont l'origine physique est mesurée et annotée ; masque zéro = **aucun apprentissage harmonique à partir du mélange**.
- `select_exact_k_notes` garantit **K0 -> []**, et pour K1..K6 exactement K hauteurs distinctes MIDI 40..88. Le module prototype ne résout pas encore les deux mêmes notes jouées par des cordes distinctes.
- Ce prototype utilise **l'attention**, pas les poids de YourMT3+, et n'implémente ni T5 ni la génération autoregressive `Program/Pitch/TimeShift/Velocity`. Pas de changement de licences/code repris ; ce sont des idées architecturales comparées.
- Les 42 trames originelles peuvent inclure de l'information **après l'attaque** ; contrairement au `midi` V2, S404 n'est donc PAS certifié causal ni utilisable en live à faible latence. Il faut d'abord construire et auditer une entrée causale réelle ou accepter un mode hors ligne.

## 4. Plan d'évaluation avant promotion

1. **Ne pas toucher S80** : référence de nombre K0–K6 = 90,25 % global et **81,98 % poly GuitarSet**, avec contamination de reconstruction déjà documentée ; garder également `S402` (36,76 % poly accord complet) en comparateur.
2. Construire un nouveau jeu d'apprentissage `frame`, `onset`, harmoniques physiques depuis les annotations temporelles, **sans** lire les notes dans les entrées modèle au test. Pour les harmoniques, utiliser la donnée hexaphonique sous supervision seulement. Ne jamais faire de `harmonic_valid=1` pour une attribution non démontrée.
3. Entraîner S404 sur compositions distinctes : groupes par **composition**, pas par événements ou vues du même morceau. En validation, évaluer vrai K0 comme absence de *nouvelle attaque* même si les notes déjà actives résonnent.
4. Comparez S402, S404 sans K et S404 avec K, ainsi que `frame` seul et `frame + onset` et supervision harmonique masquée. Ne jamais choisir/optimiser le modèle sur les morceaux du test final.
5. Mesurer par K0..K6 : exact-K inchangé, exact accord MIDI, F1 par hauteur et attaque, fausses notes, notes manquées, régressions par composition, temps d'exécution et erreurs d'octave/harmoniques.
6. Vrai benchmark indépendant : nouvelles compositions, et **S80 recalculé sur un jeu sans overlap d'entraînement**. Une bonne tête S404 conditionnée par un K d'entraînement contaminé ne devient pas automatiquement un système indépendant.

**Décision au 13 octobre 2026 :** S404 est une **base d'architecture expérimentale inspirée par les deux systèmes réels** et ne remplace pas la tête S402 sans apprentissage puis preuve de gain net.
