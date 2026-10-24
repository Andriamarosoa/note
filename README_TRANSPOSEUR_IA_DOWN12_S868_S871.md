# NOTE — Transposeur IA des aigus vers les graves (-12), S868-S871

Objectif rectifié le 24 octobre 2026 : descendre d'une octave (-12 demi-tons), au lieu de monter (+12). Le système S863/S866 de l'octave montante n'est pas réutilisable comme transposeur descendant en changeant un paramètre : sa transformation favorise les harmoniques montantes.

## Résultats expérimentaux authentiques

| Série | Méthode | Notes justes ±0,5 st sur CAL (six) | Notes justes sur six notes déjà auditées | Nettement plus graves (au moins 5 st) |
|---|---|---:|---:|---:|
| S868 | TCN neural audio direct | 0/6 | 0/6 | 0/6 |
| S869 | TCN gated | 0/6 | 0/6 | 0/6 |
| S870 | Décodeur GRU récurrent | 0/6 | 0/6 | **5/6** |
| S871 | Contrôle audit séparé | — | 0/6 pour les trois IA | Standard offline : 6/6 -12 exact |

La plupart des notes S868/S869 restent à leur hauteur originale. Le décodeur S870 descend beaucoup trop bas sur plusieurs notes, parfois vers 30-60 Hz, et ne tombe pas sur la cible. **5/6 plus graves n'est pas un score de transposition musicale**. Chaque sortie standard -12 a été mesurée par YIN (minimum 30 Hz, trame 4096) et vérifiée par rapport à la prise originale.

## Méthode et limites

- Sources : 39 enregistrements physiques GuitarSet de la banque S847, découpés en 27 TRAIN, 6 CAL et 6 notes OUTER déjà inspectées dans les séries +12. Pas d'évaluation aveugle inédite.
- La référence cible de l'apprentissage est générée avec librosa pitch_shift n_steps=-12 **hors ligne uniquement** ; le chemin d'inférence IA n'appelle aucun pitch-shift DSP, formule de note ni banque de clips.
- Les flux reçoivent strictement 256 nouveaux PCM à 44100 Hz par décision ; S868-S869 utilisent 2048 échantillons de passé, S870 un état GRU passé, sans future audio ni hold supplémentaire.
- Les tests de géométrie PCM256 et de continuité de flux ont été exécutés localement. Les temps CPU sont variables et n'ont pas été validés sur Windows/iRig ; délai acoustique bout-en-bout inconnu.
- Les six têtes officielles NOTE et MIDI n'ont été ni modifiés ni promus. La référence historique Onset 69,2584331 % F1 n'a pas été réévaluée.

## Artefacts et exécution

- Code de streaming expérimental : research/transpose256/s868_down12_inference256.py
- Tests : research/transpose256/test_s868_down12_inference256.py
- Code d'entraînement réel, checkpoints, mesures JSON et comparaison audio dans l'archive conversationnelle NOTE_S868_S871_TRANSPOSEUR_AIGU_VERS_GRAVE_PCM256.zip.
- Audio : original aigu, standard -12, S868, S869 et S870, volumes normalisés seulement pour l'écoute.

**Aucun modèle promu :** la prochaine voie requiert une véritable génération subharmonique/phase sous supervision forte avec validation sur autres guitaristes et accords enregistrés. Ne pas annoncer une réussite si la fondamentale est seulement plus grave mais incorrecte.