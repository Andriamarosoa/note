# Transposeur NOTE PCM256 — Audits S852 à S855

S851 a prouvé un échec de transcription des notes et l'incapacité du système à reproduire une même performance transposée : le système rejoue une banque. Onset référence de travail **69,2584331 % F1**, jamais modifiée dans ces études.

| Boucle | Question | Verdict |
|---|---|---|
| [S852](series/S852/resultat.md) | Un petit TCN causal apprend-il directement la sortie WAV +12 ? | Non, conserve les fondamentales |
| [S853](series/S853/resultat.md) | Perte STFT longue sur 2048 sorties d'entraînement ? | Non, améliore surtout enveloppe |
| [S854](series/S854/resultat.md) | Portes neuronales et contrainte harmonique hors ligne ? | Non, erreur médiane 11,82 st |
| [S855](series/S855/resultat.md) | Banque avec note cible idéale, sans erreur de classifieur ? | 0,763 enveloppe médiane sur 27 notes isolées, pas de preuve de performance fidèle |

**Architectures S852-S854 réellement exécutées dans une expérience locale**, sources et poids joints au rapport téléchargeable de cette conversation. Ce dépôt contient les rapports de résultats mais **pas encore les scripts et checkpoints du pilote**, afin de ne pas prétendre à une validation GitHub Actions inexistante. Ne pas utiliser des scores de laboratoire comme métriques officielles NOTE.

### Recherche scientifique
- [BRAVE: Designing Neural Synthesizers for Low-Latency Interaction (2025)](https://arxiv.org/abs/2503.11562) : conception neuronale à faible latence, sans preuve PCM256 pour NOTE.
- [Neurodyne, Interspeech 2025](https://www.isca-archive.org/interspeech_2025/gu25_interspeech.html) : représentation pitch-invariante et cohérence cyclique.
- [DDSP, ICLR 2020](https://arxiv.org/abs/2001.04643) : synthèse différentiable mais difficulté polyphonique.
- [GuitarSet original](https://zenodo.org/records/3371780) : microphone et pickup hexaphonique, annotations par corde, CC BY 4.0.

### Audit causal et décisions
- Une période complète de E2 (~82,41 Hz à 44,1 kHz) représente ~535 échantillons ; un bloc de 256 seul est insuffisant pour lire le cycle entier. Les expériences utilisent uniquement l'historique déjà reçu, sans ajouter d'attente.
- Les résultats avec un WAV transposé par traitement standard ne signifient pas que le standard respecte l'inférence Live sans délai : c'est un **professeur/témoin offline**, jamais déployé.
- **Aucune version promue.** F1 Onset et scores officiels des autres têtes non réévalués puisqu'ils n'ont pas été touchés.
- Auditer l'alignement acoustique JAMS/hex, créer un jeu de **paires de performances** avec une vraie référence source-cible, imposer une validation GT des hauteurs, de la polyphonie et de la qualité audio avant intégration.
