# S852 — Résultat : réseau neural causal audio-vers-audio PCM256

Statut: **terminé / refusé / non promu**, 24 octobre 2026.

| Modèle | CAL L1 | MAE OUTER banque | Erreur médiane de hauteur OUTER (6 notes) | Corrélation enveloppe IA vs standard sur WAV externe | p99 CPU local |
|---|---:|---:|---:|---:|---:|
| wave_l1 | 0,069507 | 0,067889 | 11,877 demi-tons | 0,8722 | 3,77 ms |
| spectral | 0,069735 | 0,068717 | 11,875 demi-tons | 0,8852 | 2,65 ms |

Le choix de CAL est wave_l1. Les deux IA sortent une note environ une octave **sous** le standard +12 : elles conservent largement la fondamentale de la prise originale. La dynamique peut sembler bonne; elle ne prouve PAS la transposition correcte. Les sorties ont RMS/standard de 0,189 et 0,203, donc très affaiblies. Test de futur causal et contrat PCM256 OK. P99 CPU local, pas Windows/iRig. Les 6 notes OUTER sont distinctes de TRAIN/CAL.

- Benchmarks S852–S855 effectués localement sur six notes réelles OUTER, séparées en identités de notes (27 TRAIN / 6 CAL / 6 OUTER) à partir de la banque authentique S847; non comparables au score Onset 80 WAV.
- Code de recherche, poids, sons et JSON exportés dans une archive autonome de la conversation; **non intégrés à NOTE**.
- Contrat: 256 PCM nouvellement reçus par décision à 44 100 Hz; mémoire passée permise, zéro lookahead, zéro hold, aucune modification des six têtes NOTE/MIDI.
- Standard +12 par librosa **hors ligne seulement** comme cible d'apprentissage/comparaison; aucune transposition DSP en inférence.
- Sur seulement ~9,94 s d'extraits de notes GuitarSet, ce pilote ne démontre pas de généralisation à un vrai mix polyphonique.
