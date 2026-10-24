---
serie: S840
statut: en_cours
type: experience
reference: S708
branche: codex/note-s840-onset-neural-latent-transposition256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S840 — Transposition latente apprise par IA pour détecter les notes (PCM256)

## Objectif
Créer **une vraie tête de transposition IA sans DSP de pitch-shift**, afin de produire de nouvelles preuves acoustiques concernant les notes de guitare, surtout les notes superposées et les fondamentales faibles. On part du témoin indépendant Onset PCM256 **69,2584331 % F1 sur 80 WAV GuitarSet** (7 607 TP, 3 093 FP, 3 660 FN). Le HGB rejeté, qui ne pouvait proposer que 802 nouvelles attaques récupérables et plafonnait à 73,8636 % dans un scénario parfait, ne doit pas servir de seule source de candidats.

## Hypothèse
Le signal audio brut ne doit pas être transposé numériquement. Un réseau neuronal peut apprendre une transformation **latente conditionnée** par un décalage souhaité en demi-tons, en apprenant les relations entre de **véritables enregistrements** de notes et d'accords transposés. L'encodeur apprendra des représentations du son brut, un transporteur neural apprendra la transposition, et un décodeur neural retournera des hypothèses sur les **hauteurs d'origine**. Ces hypothèses pourront ensuite servir de nouvelles propositions Onset, mais ne constituent ni des attaques MIDI ni un décodeur Onset déjà calibré.

## Modifications
- `src/onset256/learned_latent_transposer256.py` : un seul bloc audio [batch,256] à la fois ; encodeur CNN causal au sens de disponibilité à la frontière du bloc (pas d'audio futur), `nn.Embedding` du déplacement −12..+12 demi-tons, transporteur appris et contrôleur de qualité de chaque vue, décodeur original 49 notes.
- La transposition de la représentation ne contient **aucun calcul fixe de la fréquence**, FFT, phase vocoder, resampling, changement de vitesse ni génération de son audio. Les petits calculs sur les indices de **vérité terrain** servent uniquement à vérifier les déclarations de paires source/cible.
- Sorties jumelles : hypothèses enrichies par l'IA et tête de contrôle identique **sans transposition** pour ablation appariée. Pas de retour vers Exact-K, Active Exact-K, Offset, Frame Pitch et Full Note.
- `scripts/train_onset_neural_transpose256.py` : apprentissage supervisé à partir d'une archive externe NPZ de **paires de vrais enregistrements indépendants**, jamais de son transposé artificiellement par DSP. Supervision du latent transposé par représentation d'un véritable enregistrement cible et par son étiquette de note.
- Les données obligatoires contiennent, par paire, `source_pcm`, `target_pcm`, `source_note_mask`, `target_note_mask`, `semitones`, `source_track`, `target_track`, `split`, avec manifeste SHA256 et déclarations de provenance. L'authenticité audio doit être vérifiée par ailleurs ; une déclaration de provenance ne remplace pas un audit physique des enregistrements.
- `tests/test_onset_neural_latent_transposition256.py` : causalité, absence d'attente, refus 128/512/1024, backpropagation réelle, paires GT cohérentes, absence de fuite d'identité des prises et séparation des hypothèses vs événements MIDI.
- Création d'une branche S840 indépendante après l'audit Onset 69,2584 ; tous les checkpoints existants, scripts Live, événements MIDI et cinq autres axes restent gelés.

## Protocole
1. Apprendre sur de vrais exemples de sons **distincts enregistrés**, associés à leurs notes avant/après transposition, et non à des audio synthétiquement pitch-shiftés pour prétendre que la transposition est apprise. Une seule entrée PCM mono float32 de 256 valeurs à 44100 Hz (5,80499 ms) est accessible pour chaque décision ; **aucune fenêtre PCM brute plus longue**, sans S708 entraîné1024, sans lecture du futur.
2. Le seul état d'inférence est l'horloge et le modèle : pas de file, pas de hold, pas de note retardée ni de mémoire audio brute. La capacité de reconnaître E2 à partir de 256 échantillons est une hypothèse à tester, pas une garantie physique.
3. TRAIN, CAL et OUTER doivent être disjoints au niveau **des identités des deux enregistrements** de chaque paire. Le modèle est choisi exclusivement par le F1 proxy **par trame** sur CAL ; OUTER jamais consulté avant le gel du modèle.
4. Le F1 proxy par trame du module n'est **pas Onset event F1**. Toute proposition de correction exige ensuite un nouveau décodeur Onset indépendant et un replay comparatif de 80 WAV avec appariement NoteOn ±80 ms, compte TP/FP/FN, audits audio des FN et groupes SOLO/COMP/K0..K6. Aucun score >80 avant cette mesure.
5. Mesurer la différence entre **même encodeur sans vue** et encodeur + vues latentes (±7/±12 ou offsets choisis exclusivement par CAL). Critique obligatoire : l'IA peut simplement apprendre un raccourci de décodage de notes ; l'alignement aux **vrais enregistrements cibles** doit être évalué indépendamment et une ablation d'identité/de changement aléatoire testée.
6. La contrainte Live exige un budget p95 et p99 inférieur à 5,805 ms sur CPU Windows/iRig avec l'ensemble de la chaîne. **Une décision à cadence 256 ne prouve pas que le calcul finit en 5,805 ms.**
7. Veto de promotion : un gain individuel Onset n'a pas le droit de régresser les cinq autres caractéristiques ou d'augmenter la latence ; aucune sortie musicale publique avant preuve appariée.

## Audits obligatoires
| Domaine | Contrôle |
|---|---|
| Données | vrais enregistrements pairs, SHA original, split des prises et vérification manuelle de la provenance |
| Acoustique | entrée uniquement 256 PCM, aucune FFT ni fenêtre supérieure, zéro futur |
| Architecture | encodeur, transporteur, décodeur et pondération appris ; autres cinq axes gelés |
| Apprentissage | paires source-cible physiquement distinctes, vrais GT source-cible, CAL hors TRAIN, OUTER fermé |
| Métriques | F1 proxy ≠ F1 événement ; au moins 80 WAV et TP/FP/FN comparables avant promotion |
| LIVE | timestamp au présent, zéro hold ; p99/hardware à évaluer, aucun MIDI émis |

## Statut
**Prototype expérimental, pas modèle entraîné sur les vrais WAV et pas amélioration de score prouvée.** Les tests unitaires artificiels contrôlent le chemin et les gradients, mais ils ne constituent aucune donnée de guitare, aucun F1 et aucune performance Live réelle. La référence actuelle reste 69,2584331 % Onset F1, chaîne acoustique directe PCM256. Cette branche constitue la première mise en œuvre de l'hypothèse de transposition latente apprise, pas une autorisation d'intégrer au MIDI.
