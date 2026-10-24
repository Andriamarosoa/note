# S854 — Résultat : réseau neuronal à portes et supervision d'harmoniques

Statut: **terminé / refusé / non promu**, 24 octobre 2026.

| Modèle | CAL loss (identique entre variantes) | Médiane erreur hauteur OUTER | Corrélation enveloppe WAV externe | RMS IA / standard |
|---|---:|---:|---:|---:|
| gated_harmonic | **0,470711** | **11,816 demi-tons** | 0,62479 | 0,831 |
| ungated_harmonic | 0,486291 | 11,860 demi-tons | 0,65898 | 0,868 |

Le réseau à portes multiplicatives est le meilleur sur CAL, avec une légère amélioration de la hauteur par rapport à S853, mais loin d'une octave transposée. La transformation conserve en grande partie les fréquences de la source. Le supervision harmonique par vérité MIDI ne s'applique **qu'à la perte offline** ; l'inférence reste un vrai réseau audio-vers-audio causal sans synthétiseur ni banque. Aucune promotion.

- Benchmarks S852–S855 effectués localement sur six notes réelles OUTER, séparées en identités de notes (27 TRAIN / 6 CAL / 6 OUTER) à partir de la banque authentique S847; non comparables au score Onset 80 WAV.
- Code de recherche, poids, sons et JSON exportés dans une archive autonome de la conversation; **non intégrés à NOTE**.
- Contrat: 256 PCM nouvellement reçus par décision à 44 100 Hz; mémoire passée permise, zéro lookahead, zéro hold, aucune modification des six têtes NOTE/MIDI.
- Standard +12 par librosa **hors ligne seulement** comme cible d'apprentissage/comparaison; aucune transposition DSP en inférence.
- Sur seulement ~9,94 s d'extraits de notes GuitarSet, ce pilote ne démontre pas de généralisation à un vrai mix polyphonique.
