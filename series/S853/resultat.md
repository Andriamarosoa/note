# S853 — Résultat : pertes spectrales à contexte audio long

Statut: **terminé / refusé / non promu**, 24 octobre 2026.

| Modèle | CAL spectral loss (échelles propres) | Erreur médiane de hauteur OUTER | Corrélation enveloppe WAV externe |
|---|---:|---:|---:|
| log_spectral | 0,267019 | 11,871 demi-tons | 0,94645 |
| linear_spectral | 0,847370 | 11,878 demi-tons | 0,86339 |

Les deux fonctions de perte ne sont pas comparables numériquement entre elles ; CAL a choisi log_spectral à l'intérieur de sa variante. L'enveloppe ressemble davantage au standard, mais la note reste à sa fréquence d'origine : **pas de transposition +12**.

Perte spectrale pendant l'entraînement sur sortie 8 × 256 (2048 échantillons), mais l'inférence est strictement par nouveaux blocs de 256, sans futur audio. Enseignement : une forte similarité d'enveloppe peut masquer une erreur fondamentale de hauteur.

- Benchmarks S852–S855 effectués localement sur six notes réelles OUTER, séparées en identités de notes (27 TRAIN / 6 CAL / 6 OUTER) à partir de la banque authentique S847; non comparables au score Onset 80 WAV.
- Code de recherche, poids, sons et JSON exportés dans une archive autonome de la conversation; **non intégrés à NOTE**.
- Contrat: 256 PCM nouvellement reçus par décision à 44 100 Hz; mémoire passée permise, zéro lookahead, zéro hold, aucune modification des six têtes NOTE/MIDI.
- Standard +12 par librosa **hors ligne seulement** comme cible d'apprentissage/comparaison; aucune transposition DSP en inférence.
- Sur seulement ~9,94 s d'extraits de notes GuitarSet, ce pilote ne démontre pas de généralisation à un vrai mix polyphonique.
