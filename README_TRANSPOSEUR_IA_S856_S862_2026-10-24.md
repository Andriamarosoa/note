# Transposeur IA PCM256 — Audits et boucles S856 à S862 (24 octobre 2026)

Statut : **7 séries terminées localement ; aucune promue**. Les scripts Python, les poids entraînés, les vrais WAV, les JSON et la démonstration audio sont fournis dans l'archive conversationnelle **NOTE_S856_S862_BOUCLES_NEURALES_PCM256.zip**. Ce dépôt contient les rapports de résultats, pas une intégration NOTE ni une prétendue validation GitHub Actions de ces modèles.

## Résumé des mesures

| Série | Résultat et décision |
|---|---|
| [S856](series/S856/resultat.md) | Nouvelle architecture : erreur YIN ~11,89 demi-tons, refus |
| [S857](series/S857/resultat.md) | MLP symétrie paire : 2f/f0 +5,53 dB, refus |
| [S858](series/S858/resultat.md) | Modèle pair + filtres appris : 2f/f0 +26,60 dB mais impaires parasites |
| [S859](series/S859/resultat.md) | Audit : ratio paires/impaires seulement +0,44 dB, p99 CPU local S858 ~1,17 ms |
| [S860](series/S860/resultat.md) | Meilleur candidat labo choisi sur CAL : log-mel médian CAL 12,373 dB |
| [S861](series/S861/resultat.md) | Pénalité renforcée : log-mel CAL 12,427 dB, rejeté face à S860 |
| [S862](series/S862/resultat.md) | Mesures communes + 4/4 tests unitaires ; aucune transposition musicale prouvée |

Sur les six notes OUTER **déjà auditées** : log-mel S858 13,702 dB → S860 11,707 dB, RMS médian 18,0 % → 46,0 %, mais erreur YIN ~11,89 demi-tons et dominance totale paires/impaires seulement 0,13 dB. **Ne pas déclarer un pourcentage de succès de transposition, ces chiffres ne le justifient pas.**



**Périmètre strict.** Modèles expérimentaux indépendants : exactement 256 nouveaux PCM à 44,1 kHz par hop, contexte passé uniquement, aucun futur ni hold, aucun changement de NOTE (Onset, Offset, Exact-K, Active Exact-K, Frame Pitch, Full Note) ni du MIDI. Métriques de labo sur banque S847 de 39 notes réelles (27 TRAIN / 6 CAL / 6 OUTER distinctes en hauteur, mais de même collection/guitariste) ; les 6 OUTER ont déjà été examinées en itération, donc ne valent pas jeu inédit après conception. Teacher standard +12 est **offline seulement**. Aucun modèle promu et aucun nouveau F1 officiel.

## Recherche scientifique
- [Neurodyne (Interspeech 2025)](https://www.isca-archive.org/interspeech_2025/gu25_interspeech.html) — représentations indépendantes de la hauteur et cohérence cyclique (principalement voix)
- [PESTO (TISMIR 2025)](https://reference-global.com/article/10.5334/tismir.251) — équivariance à la transposition pour reconnaissance de hauteur, pas waveform généré
- [DDSP neural polyphonic guitar (DAFx 2024)](https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_49.pdf) — conditionnement corde par corde et synthèse guitare
- [Review différentiable audio](https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2023.1284100/full) — produits d'intermodulation des waveshapers

**État final :** scores de laboratoire améliorés, mais aucune promotion. Ne pas confondre réussite technique, octaves spectrales, F1 Onset, et transposition musicale fidèle.
