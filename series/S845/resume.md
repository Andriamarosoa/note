---
serie: S845
statut: termine
type: experience
reference: S708
branche: codex/note-s845-raw-wave-acoustic-decoder256
run_id: 40089022537
commit_sha: 0a1653d50289c9f8b67c88d6e3067300959f0cd2
promotion: false
---

# S845 — Décodeur IA qui conserve la forme d'onde vraie du microphone (PCM256)

## Objectif
Faire une nouvelle expérience scientifiquement fondée après S844 : la tête de transposition / reconstruction du vrai son GuitarSet avait prouvé son apprentissage sur les trois archives réelles micro/hex/JAMS et des indices de notes E2/E3 faibles mais positifs, **sans gain des vues transposées**. Surtout, sa reconstruction des vraies voies hex était *moins bonne qu'une sortie silence* : MAE 0,029785044 contre 0,029205378 (différence −0,000579666 en faveur du silence). L'objectif immédiat de S845 est de **dépasser le silence sur MAE audio d'origine**, sans faux F1 Onset ni modification du MIDI.

## Hypothèse
Le décodeur S844 reconstruit six formes d'onde de 256 valeurs essentiellement à partir d'un état latent global, puis le masque par des probabilités de corde, ce qui favorise une sortie presque muette (moyenne |sortie|=0,001868 pour moyenne |cible hex|=0,029205). Une **voie locale convolutionnelle apprise** directement sur les 256 **vrais échantillons de microphone** et conditionnée par le contexte GRU passé peut préserver davantage la phase et le timbre. Ce n'est ni une transposition DSP, ni un générateur d'oscillateurs, ni une garantie que le son estimé est égal à la prise hex : ce sont des estimations entraînées sur des signaux vrais.

## Modifications
- `src/onset256/real_audio_transpose_head256.py` : ajout d'un filtre temporel CNN appris `local_string_filters` et gains `string_condition` en complément du décodeur latent six voies. La tête **ne réduit plus artificiellement le niveau sonore par la confiance de présence de corde** ; les présences restent évaluées séparément. La transposition ±12 du latent est maintenue ; les représentations déplacées restent des *estimations*, pas de nouveaux vrais enregistrements.
- `tests/test_s845_raw_waveform_acoustic256.py` : conserve exactement 256 nouveaux échantillons, vérifie que l'audio brut influence vraiment les six sorties, que le décodeur apprend sur les cibles hex, que le masquage silencieux est impossible, que le futur et le MIDI sont absents.
- Protocole réel S844 conservé : GuitarSet trois archives avec empreintes officielles, TRAIN guitariste 04, CAL guitariste 05, OUTER guitaristes 00–03, 16 pistes TRAIN, 16 CAL, 64 OUTER, 96 séquences × 8 blocs, GT E2/E3/accord, poids train-only, 8 époques.
- Décodeur et toute l'expérience restent isolés ; **aucune des six architectures de NOTE, aucun modèle Onset de travail et aucun MIDI officiel ne sont modifiés**.

## Protocole
1. Re-vérifier les fichiers audio réels, JAMS et hex par SHA/MD5. Refuser tout audio synthétisé, tout signal DSP transposé, tout S708 entraîné1024.
2. Exécuter tests du vrai signal, backprop et causalité. Chaque décision voit exactement 256 nouvelles valeurs à 44100 Hz, sans audio du futur ni attente, avec GRU passé seulement.
3. Apprendre le modèle nouveau à partir des vraies voies micro/hex simultanées. Sélection du checkpoint uniquement sur CAL, puis geler avant OUTER. Aucun test sur OUTER pour modifier poids/seuils.
4. Mesurer sur OUTER les MAE audio hex vs **silence** et comparer explicitement à S844. Si reconstruction reste pire que silence, S845 est **rejeté**, même si un autre proxy de note augmente.
5. Mesurer E2, E3, E2+E3 sur les mêmes 49 152 décisions PCM256 à 64 pistes et les mêmes partitions. Comparer avec et sans vues transposées, avec TP/FP/FN pour E2/E3/accord. Les F1 sont par **présence de note**, jamais Onset événement ±80ms.
6. N'envisager une promotion Onset qu'après un vrai décodeur de NoteOn et replay exhaustif sur les 80 WAV signés et jeu externe, sans régression des autres axes. Aucune conclusion de p99 Windows/iRig sans mesure.

## Audits obligatoires
| Domaine | Preuve |
|---|---|
| Données | vraies prises hex/micro de GuitarSet, SHA/MD5 officiels et GT complets |
| Acoustique | entrée strictement PCM256, CNN vrai signal et GRU passé, MAE vs silence |
| Architecture | transport latent IA et contrôle sans transposition, six axes et MIDI gelés |
| Apprentissage | poids classes TRAIN seul ; sélection checkpoint CAL seul ; OUTER réservé |
| Métriques | MAE audio, TP/FP/FN E2, E3, E2+E3 actifs; jamais confusion avec Onset |
| LIVE | zéro futur/hold, budget 5,805 ms, p99 complet non certifié |

## Statut
**Terminé, non promu.** [Run #40089022537](https://github.com/Andriamarosoa/note/actions/runs/40089022537) : 16 TRAIN, 16 CAL, 64 OUTER, 49 152 blocs PCM256 ; MAE audio S845 0,029422 contre 0,029205 pour le silence (échec), note F1 transposée E2 14,31 %, E3 17,22 %, E2+E3 10,05 %, fortes erreurs. La sortie estimée moyenne vaut 0,000856 contre 0,029205 pour la vraie cible hex : son quasi silencieux. Aucun modèle Onset/MIDI promu. Le résultat Onset actuel demeure **69,2584331 % F1** (7607 TP, 3093 FP, 3660 FN). S844 a montré que la transposition ne battait pas l'encodeur sans vues sur les graves ; S845 cible donc en priorité le signal reconstruit. Toute forme d'onde sortie du modèle est une **estimation neuronale à valider**, pas un « vrai son » garanti.

## Utilisation — vrai microphone vers six WAV reconstruits par IA (prévisualisation)

Le script `scripts/s845_export_real_audio_estimates256.py` produit six WAV estimés sur les cordes, un WAV micro estimé et, en option, six WAV estimés de transposition neuronale −12 / +12. Les noms contiennent toujours `estimated_` ; **ce n'est pas une preuve qu'un son de corde original a été physiquement isolé**, et les vues transposées ne sont pas de vrais enregistrements indépendants. Ces sorties servent uniquement aux audits acoustiques et à l'écoute critique ; aucun NoteOn/NoteOff/MIDI.

La source d'entrée doit être une **vraie prise de guitare** WAV mono, 44100Hz, PCM16 ou float32. Ne pas fournir un son fabriqué pour revendiquer la reconstruction. Utiliser **uniquement un checkpoint issu de l'apprentissage S845**, car les poids S843/S844 n'ont pas les nouvelles convolutions locales et ne sont pas compatibles :

```bash
python -m scripts.s845_export_real_audio_estimates256 \
  --wav chemin/ma_guitare_reelle.wav \
  --model chemin/s845_trained/s843_real_hex_neural.pt \
  --out chemin/estimations_s845
```

Le nom `s843_real_hex_neural.pt` est le nom de sortie hérité du trainer, **pas** une autorisation de charger les anciens poids S843. Vérifier que le checkpoint a été extrait de l'artefact **S845** et non d'un run antérieur. L'inférence refuse un fichier ou un tenseur incompatible. Les WAV peuvent être écoutés en les comparant à la vraie source ; si leur énergie est quasi nulle ou leur forme d'onde infidèle, le résultat est un **échec**, pas un résultat à promouvoir.

La sortie JSON contient les cinq hypothèses de notes les plus probables à chaque frontière PCM256 ; ce ne sont pas des Onset NoteOn et aucun score réel de détection n'est induit. Les tests du script avec signaux artificiels ne servent qu'à vérifier la robustesse du logiciel, jamais à revendiquer une vraie qualité de guitare.
