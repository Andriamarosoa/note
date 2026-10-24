---
serie: S843
statut: en_cours
type: experience
reference: S708
branche: codex/note-s843-real-audio-transpose-reconstruction256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S843 — Tête indépendante de reconstruction et de transposition acoustique neuronale, PCM256

## Objectif
Créer une **véritable tête neuronale capable d'estimer des sons de corde de guitare** depuis le son capté au microphone, puis d'explorer une transposition apprise en espace latent pour distinguer fondamentales, harmoniques et vraies notes superposées. Refuser tout oscillateur sinusoïdal, échantillon synthétisé comme vérité terrain, phase-vocoder, transformation DSP de fréquence, rééchantillonnage ou faux renforcement qui inventerait un E3 absent. Surtout ne jamais prétendre que le son reconstruit est physiquement identique au son réel : il s'agit d'une estimation neuronale **supervisée par une prise hexaphonique réelle de la même guitare**.

## Hypothèse
Le microphone mono contient la somme acoustique complexe ; GuitarSet dispose en même temps de six **vrais signaux enregistrés sur les cordes** via capteur hexaphonique (version debleeded). Un réseau CNN+GRU causal peut apprendre à estimer les six prises réelles à partir du PCM micro et produire une preuve de note physique par voie. La transformation neuronale conditionnée par demi-tons peut mettre en doute une harmonique confondue avec une vraie attaque ; son intérêt reste à mesurer, la tête ne crée pas directement de NoteOn.

## Modifications
- `src/onset256/real_audio_transpose_head256.py` : **CNN raw audio**, mémoire `GRUCell` *strictement passée*, décodeur appris de six formes d'onde de 256 échantillons, six confidences de présence de corde, tête de notes MIDI 40..88, décodeur neuronal inverse mic, transporteur latent appris ±12 et arbitre de vues contre le même encodeur sans transpose.
- `CausalRealAudioTransposeStream256.push` exige **exactement un nouveau bloc de 256 échantillons float32 à 44,1kHz par appel**. Le modèle utilise un état neuronal conservé du passé, pas de 1024 nouveaux PCM, pas de fenêtre brute FFT, pas de file ni de lookahead, pas de NoteOn, pas de MIDI. Toute estimation sort à l'horloge du bloc courant, mais le temps CPU p95/p99 n'est pas automatiquement garanti.
- `scripts/s843_train_real_hex_transpose256.py` : entraîne avec les **archives d'origine GuitarSet** :
  - mono micro `audio_mono-mic.zip` SHA256 `237cdc58353d25c3c9683f4565a0f1cf2db30a9051abca545a919f8f1296dc28`,
  - 6 canaux `audio_hex-pickup_debleeded.zip` MD5 officiel `c31d97279464c9a67e640cb9061fb0c6`,
  - annotations `annotation.zip` SHA256 `8daa02e6417ccca1685feb44b135e95928ad7037e5032ecb326b5791856fda99`.
  Les checksums **doivent** correspondre. Un fichier autre qu'un enregistrement GuitarSet signé est rejeté par défaut. Ce sont des prises réelles simultanées, pas des fichiers audio DSP transposés.
- Les partitions sont strictement séparées **par guitariste** : TRAIN joueur 04, CAL joueur 05, OUTER joueurs 00–03. Le score indépendant de cette nouvelle tête sera à calculer hors entraînement. La référence Onset 69,2584331 % reste un benchmark de développement historique ; le score ne change pas sans replay intégral.
- `reconstruction_loss` supervise réellement les 6 sorties audio par les six vraies voies hex enregistrées, plus la représentation de l'activité notes/cordes issue de JAMS. `real_shift_pair_loss` apprend les codes de transposition sur deux prises réelles distinctes (+12 / −12), **sans imposer une même forme d'onde phase-par-phase entre performances différentes**.
- `tests/test_s843_real_audio_transpose_head256.py` : tests causaux, formes d'onde, gradients de reconstruction et transposition, train split, refus entrées non-256, checksums GuitarSet, pas de MIDI.
- Aucun changement dans les têtes officielles Onset/Offset/Exact-K/Active Exact-K/Frame Pitch/Full Note. Aucune sélection qui influe sur leur décodage. Référence du benchmark historique sauvegardée.

## Protocole
1. Acquérir les trois archives GuitarSet originales **sans modifier leurs octets** ; SHA256/MD5 obligatoires. Auditer le mapping canal note_midi↔corde et l'alignement physique microphone/hex : l'ordre des six annotations JAMS est actuellement supposé correspondre aux six hex ; si l'audit d'alignement échoue, refuser toute conclusion de précision de séparation.
2. Entraîner le modèle sur des séquences **de plusieurs blocs passés**, un nouveau bloc de 256 échantillons par décision ; mémoire GRU causale autorisée, pas d'audio futur. Le modèle ne reçoit jamais les étiquettes ou les pistes hex au moment de l'inférence.
3. Mesurer à part : qualité des six reconstructions (L1/erreur PCM sur hex), absence de notes hallucinéés, F1 de présence réelle par corde, F1 de note par bloc, ablation avec/sans transposition conditionnée, statistiques E2/E3/E2+E3.
4. Un modèle générant des sons plausibles mais des E3 fictifs doit être rejeté. Sur les trois catégories isolé/accord, comparer faux positifs *harmoniques* et vrais positifs récupérés ; ne pas accepter uniquement un meilleur son.
5. Enfin seulement entraîner un générateur indépendant d'Onset à partir de ces preuves et comparer **sur les 80 WAV GuitarSet originaux, 306536 frontières PCM256**, TP/FP/FN, F1 événement ±80ms, non-régression des autres axes, et validation hors corpus.
6. Mesurer CPU p95/p99 et latence réelle du pipeline Windows/iRig. 5,805 ms de cadence **n'est pas** un délai physique garanti.

## Audits obligatoires
| Type | Validation |
|---|---|
| Données | 3 archives originales signées, debleeded hex six canaux, JAMS source, alignement des prises, partitions joueurs sans chevauchement |
| Acoustique | entrées 256 seules ; signaux reconstruits estimés à partir de vraies prises ; transparence sur la différence microphone/hex |
| Architecture | réseau indépendant CNN+GRU/transporteur latent, contrôle sans vues, pas de connexion MIDI |
| Apprentissage | paires WAV hex réelles uniquement, code de transposition appris sans DSP, séparation TRAIN 04/CAL 05/OUTER 00..03 |
| Métriques | RMSE/L1 audio, TP/FP/FN E2/E3/E2+E3, puis vrai Onset F1 sur 80 WAV après création d'un détecteur |
| LIVE | exactement 256 nouveaux échantillons par décision, 0 lookahead, 0 attente ; p99 CPU Windows non encore vérifié |

## Statut
Architecture, entraîneur et tests ajoutés comme **recherche isolée**, sans promotion. Aucune prétention qu'une forme d'onde décodée par IA est identique à la prise physique. Le premier test technique sert à vérifier les gradients, le respect du bloc 256 et le non-régression du code, **pas** le F1 musical. Le score témoin d'Onset PCM256 reste **69,2584331 %** tant qu'un replay complet n'a pas mesuré de gains.
