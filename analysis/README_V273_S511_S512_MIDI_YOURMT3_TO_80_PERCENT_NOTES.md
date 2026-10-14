# S511–S512 — Boucle de recherche vers 80 % de détection des notes : midi, YourMT3+ et nouveaux candidats

**Date 14 octobre 2026. Statut expérimental : ne pas promouvoir S80, ne pas modifier Live.**
L'objectif demandé est au moins 80 % **F1 événementiel des notes** (hauteur + onset sur un ensemble de morceaux indépendants). L'accuracy Exact-K et l'exactitude des **accords complets** doivent être contrôlées séparément. Il est interdit d'annoncer que 80 % est atteint à partir d'une simple borne supérieure d'oracle ou de la précision conditionnelle sur des attaques connues.

## Sources vérifiées directement

### `Andriamarosoa/midi` (code et résultats)

- `src/polyphonic/model.py` et `docs/POLYPHONIC_V2.md` : prédicteur causal CNN/TCN multilibellé avec têtes *frame*, *onset* et 20 harmoniques par pitch ; pertes harmoniques **masquées quand les annotations de partiels ne sont pas fiables**, et entraînement via GuitarSet + GAPS + Guitar-TECHS, validation par morceau/joueur.
- `src/polyphonic/decoder.py` : état global **par hauteur MIDI**, hystérésis activation/libération, délai de réattaque, protection contre les harmoniques sauf preuve d'une vraie attaque indépendante, non-nativité d'une corde pour l'identité de la note.
- `src/stream/onset_detector.py` : preuve physique d'attaque par flux spectral positif, croissance RMS, seuils robustes/médiane + MAD, réarmement et refroidissement.
- `runs/polyphonic/polyphonic_v2_2_guitarset_gaps_guitar_techs_20260722_024644/reports/FINAL_PRODUCT_V2_2.md` : **F1 événement onset = 0,242221**, F1 onset+offset = 0,068950, **1 623** notes appariées sur 4 916 annotations, 6 862 notes générées sans correspondance ; **1 932 notes manquantes ont un support spectral fort**. Cela démontre que le modèle `midi` V2.2 n'est pas une solution prête à atteindre 80 %, même si son architecture est instructive.
- L'intégration doit préserver la distinction entre preuves « note présente » et « note nouvellement attaquée ». Un pic acoustique seul ne prouve ni une note ni un vrai onset. Ne pas créer des labels harmoniques par attribution aveugle aux sources mélangées.

### YourMT3+ (Chang et al., 2024, arXiv 2407.04822)

- Papier : https://arxiv.org/abs/2407.04822 ; dépôt officiel : https://github.com/mimbres/YourMT3
- Architecture scientifique : encodeur hiérarchique **PerceiverTF** en temps-fréquence, attention croisée spectrale, **Mixture-of-Experts**, décodeur multicanal d'événements temporels de type MT3/T5, augmentation intra-piste/inter-piste pour l'entraînement multi-instrument.
- Le dépôt de présentation référencé ne fournit pas un checkpoint compatible avec notre tronc `note` déjà installé. **Ce prototype S511 n'utilise donc PAS un YourMT3+ préentraîné.** Aucune métrique YourMT3+ n'est attribuée à S511.
- À tester dans une architecture entraînée ultérieurement : pré-encodeur spectral plus experts conditionnels par instrument/registre et contraintes de note-on / note-off, séparation d'annotations incomplètes, sans copier naïvement la tokenisation T5 dans notre sortie live E3/G3/B3.

## Borne réelle identifiée dans `note` (S508)

Sur les **45** attaques guitare annotées et appariées, les 13 têtes n'atteignent que **31/45 accords exacts au maximum** lorsqu'on choisit une seule tête, et **34/45** lorsque les hauteurs déjà observées sont recombinées par oracle. Dans **11/45** attaques, au moins une vraie hauteur est absente de toutes les propositions. Sur 220 attaques piano appariées, les plafonds sont **136/220** (tête entière) et **172/220** (combinaison par hauteur), et **48/220** possèdent au moins une vraie hauteur manquante ; **102** vraies attaques piano n'ont même pas de groupe de notes Basic Pitch apparié.

**Ces plafonds de sélection ne sont pas des résultats de modèle.** Ils prouvent que le goulot ne se corrige pas en ajoutant seulement des sélections neuronales aux mêmes notes. De nouveaux producteurs de fréquences fondamentales et de centres temporels d'attaque sont nécessaires.

## S511 — Nouveau producteur de hauteurs depuis le vrai WAV, indépendant de Basic Pitch

`scripts/s511_new_raw_harmonic_pitch_onset_generators.py` : reprenant les concepts généraux de `midi`, l'algorithme calcule des spectres STFT et un **score de support harmonique** sur MIDI40–88, au plus huit partiels par fondamentale, avec pondération décroissante, indice de fondamentale, normalisation robuste, évolution de l'énergie par pitch et flux spectral de nouvelle attaque. Les candidates `top6` et `top12` sont choisies par score acoustique **avant toute ouverture XML/MIDI**. Les vrais WAV sont téléversés depuis leurs sources originales officielles et vérifiés SHA contre S277 (guitare IDMT) et S470 (piano MAESTRO si activé), sans redistribution des WAV.

S511 est un **générateur de candidates audio**, et non un détecteur de notes certifiées. Son fenêtrage STFT est offline et non causal : **ne pas prétendre préserver une latence Live testée**. La simple présence de top12 augmente généralement le rappel oracle mais peut réduire la précision lorsque des harmoniques et résonances sont confondues avec des vraies notes.

## S512 — Validation a posteriori déjà codée

`scripts/s512_audit_original_audio_harmonic_candidate_note_recall_vs_XML.py` évalue sur 45 vraies attaques IDMT (21 poly), après gel S511 :
- appariement temporel un-à-un des attaques acoustiques et XML ±80ms ;
- rappel réel des hauteurs manquantes et **nouvelles hauteurs réellement retrouvées** hors Basic Pitch ;
- précision/rappel/F1 des pitches **conditionnels sur des attaques positives connues**, et véritables ensembles complets de notes ;
- corrections, régressions et sous-ensemble poly ; tous les 45 vrais onsets sont dans le dénominateur ;
- événements acoustiques non appariés = **inconnus**, pas des faux onsets confirmés.

L'étape d'entraînement ultérieure doit récupérer de nouveaux jeux d'enregistrements avec **positifs et vrais négatifs** d'attaque, plus de diversité que 5 IDMT et 3 MAESTRO déjà explorés. Créer des têtes apprises `frame`, `onset`, `harmonic`, `independent-instrument-expert`, `new-note-vs-sustain`; apprendre les poids/abstentions seulement sur des splits par morceaux/joueurs, et traiter la direction des erreurs sur chaque source.

## Règles de décision

- Aucune promotion avant dépassement de **80 % de F1 note événementiel** sur des enregistrements **réellement indépendants**, à la fois précision et rappel instrumentés, plus audits Exact-K et ensembles complets.
- Interdit d'utiliser les étiquettes test déjà exposées pour choisir `top6/top12`, le seuil, les hyperparamètres ou l'architecture puis de revendiquer un test aveugle.
- Aucune modification de `freeze_local_combo`, S80 officiel, Live, onset/offset, ni de checkpoint source tant qu'une validation indépendante et la preuve de gain non régressif ne sont pas démontrées.
