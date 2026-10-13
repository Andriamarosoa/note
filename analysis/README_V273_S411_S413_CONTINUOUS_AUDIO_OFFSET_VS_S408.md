# S411–S413 — S408 figé comme référence Offset, essai avec audio continu

**13 octobre 2026.** Demande : conserver S408 comme **référence active de développement** tout en cherchant une meilleure précision Offset. Aucun changement de référence Exact-K S80 ni de note MIDI S402 ou temps S407. Aucune promotion publique sans preuve externe.

## Baseline utilisateur : S408, figée

Ref `analysis/v273_active_development_reference.json → active_offset_detection_head`.

| Référence active | Résultat sur GuitarSet développement |
|---|---|
| S80 K0..K6 | 90,2544 % global / 81,97698 % poly K2..K6 |
| S402 hauteurs | 36,7637 % accord poly entièrement exact |
| S407 Onset temporel | 84,8266 % F1 ±20 ms sans condition sur MIDI |
| **S408 Offset (actif par choix utilisateur)** | **13 050/25 120 = 51,9506 %** Offset correct ±50 ms sur notes déjà appariées en pitch+Onset |
| **S408 note complète** | **38,1417 % F1** MIDI+Onset+Offset ±50 ms |
| S408 MAE Offset parmi les notes appariées | 105,425 ms |
| S408 vrais offsets >=1s corrects | 0/737 |

**Aucune comparaison ne peut convertir la précision temporelle conditionnelle en Exact-K ou en note complète.** 25 120 notes candidates S80/S402/S407 correspondent à une vraie attaque de même hauteur à ±50 ms sur 34 954 notes JAMS, parmi 33 475 notes prédites.

## S411 : extraction continue de vrais WAV, y compris les intervalles K0

- Source : `scripts/s411_extract_continuous_GuitarSet_WAV_pitch_survival_features.py`.
- [Run GitHub S411 #38374781473](https://github.com/Andriamarosoa/note/actions/runs/38374781473) **success**, artifact `11799939667`.
- Source authentifiée : **GuitarSet audio_mono-mic.zip 656 927 981 octets**, MD5 officiel `275966d6610ac34999b58426beb119c3`. URL miroir HuggingFace `EZMONYI/music-ai-human-test-audio/external_controls/guitarset_source_v1/audio_mono-mic.zip` ; provenance officielle GuitarSet Zenodo `3371780`.
- 190 morceaux audios, **spectre STFT calculé en continu toutes les 10 ms** avec harmonique MIDI 40..88 et 6 partiels, et croissance/décroissance, même entre événements ; 33 475 vecteurs continus multi-horizons de pitch associé à une prédiction existante.
- **IMPORTANT** : le micro mono S411 **n'est pas le pickup mix utilisé historiquement par S258**. Le système doit être réentraîné/évalué sur ces nouvelles variables, sans les présenter comme les caractéristiques source S80 originales.
- Aucune annotation JAMS n'est ouverte dans l'extraction, aucun modèle prédictif K/note/timing n'est modifié, aucune mesure Offset par S411 seul. Ce n'est pas encore un modèle temps continu réel complet par corde.

## S412 : comparaison mesurée contre S408 figé

- [Run GitHub S412 #38375133317](https://github.com/Andriamarosoa/note/actions/runs/38375133317) **success**, artifact `11800518747`.
- `scripts/s412_evaluate_real_continuous_WAV_sustain_note_offset_against_S408.py`.
- Quatre folds de compositions (pas de chansons identiques dans train/test de cette tête), 2 modèles pré-déclarés : entraînement ordinaire et entraînement surpondérant les longs offsets, évalués directement ou mélangés à 50% avec S408.
- Cibles d'apprentissage à partir des durées JAMS seulement des notes à MIDI et Onset correctement appariés dans les **compositions d'apprentissage**. Les durées vraies du fold de test ne sont jamais entrées dans les modèles.
- **Même K, même MIDI, même Onset, même ensemble de vraies notes appariées** que S408 : 33 475 notes proposées, 25 120 appariées.
- Métadonnées de performance :
  
| Mode | NoteOff correct ±50ms parmi notes Onset/pitch appariées | Précision conditionnelle | F1 note complète ±50ms | MAE |
|---|---:|---:|---:|---:|
| **S408 référence** | 13 050 | **51,95 %** | **38,14 %** | 105,43 ms |
| **S412 WAV continu ordinaire** | **14 313** | **56,98 %** | **41,83 %** | **95,87 ms** |
| S412 continu pondéré pour notes longues | 12 643 | 50,33 % | 36,95 % | 105,69 ms |
| S412 continu ordinaire, mélange 50% | 13 916 | 55,40 % | 40,67 % | 99,24 ms |
| S412 pondéré, mélange 50% | 12 977 | 51,66 % | 37,93 % | 99,84 ms |

S412 standard : **+1 263 NoteOff corrects nets à ±50 ms** vs S408 ; gain relatif en point **+5,03**, F1 note complète **+3,69 points**. La correction pondérée longue améliore un type de note au prix d'une régression globale et n'est pas retenue comme challengeur principal.

| Durée vraie | S408 / nombre | S412 ordinaire / nombre | Limite |
|---|---:|---:|---|
| <100 ms | 503/1 247 (40,34%) | **630/1 247 (50,52%)** | Amélioration |
| 100–499 ms | 12 305/20 385 (60,36%) | **13 248/20 385 (64,99%)** | Amélioration |
| 500–999 ms | 242/2 751 (8,80%) | **435/2 751 (15,81%)** | Amélioration, insuffisant |
| >=1000 ms | 0/737 | 0/737 | **Verrou majeur toujours entier** |

Références de méthodologie : `scripts/s409_audit_S408_note_offset_errors_by_duration_and_K.py`, `scripts/s413_compare_S412_vs_user_reference_S408_paired_note_offset_errors.py` (audit de corrections/régressions par durée, fold, K, octave sur predictions figées).

## Décision de conception et limites

S408 reste **active_offset_detection_head**, l'utilisateur l'a explicitement désignée comme référence ; le meilleur S412 mesuré est `development_challenger` et **ne remplace pas silencieusement S408**.

Pour dépasser S412, surtout les fins de notes >1s :
- Ne pas entraîner un simple modèle global de durée en surpondérant les longues notes ; les régressions 100–499 ms dépassent les corrections utiles.
- Prédire la **survie de chaque note active** à chaque pas temporel, à partir des 10 ms du signal STFT et de son historique depuis le NoteOn (puis extraire NoteOff au franchissement calibré + ajustement local en millisecondes).
- Le K de S80 conserve son rôle : **seules de nouvelles notes peuvent être créées si K>0 ; quand K0, toutes les notes déjà actives peuvent continuer, décroître ou se terminer**. Ne jamais supprimer les frames K0 du réseau d'Offset.
- Tester notes étouffées, réattaques, vrais offsets >=1000 ms, changements de source microphone/pickup, accords K2..K6 et doublons MIDI sur différentes cordes.
- Calibrer et comparer sur un ensemble musical réellement inédit pour **tout** le système S80+notes+onsets+offsets. Le décompte S80 actuel a un chevauchement historique avec GuitarSet, et les développements S408/S412 ont été guidés par ses résultats : la validation song-held des têtes **ne prouve pas** une généralisation globale sur nouveaux morceaux.

**Statut** : réussite mesurée de l'essai continu S412 dans GuitarSet développement, **pas de promotion officielle**.
