# NOTE — boucle IA de transposition S863 à S867 (24 octobre 2026)

**État : cinq audits/expériences locaux terminés, aucune promotion NOTE.** Cette branche contient un moteur shadow de replay PCM256 et ses tests structurels, plus le bilan des entraînements. Les poids entraînés et audios comparatifs sont dans l'archive conversationnelle `NOTE_S863_S867_TRANSPOSEUR_IA_PCM256.zip` ; ils ne sont pas déployés ni committés en binaire dans GitHub.

| Série | CAL (6 notes) | 6 notes OUTER déjà inspectées | Distance log-mel OUTER médiane vs standard |
|---|---:|---:|---:|
| S860 référence | 0/6 | 0/6 | **11,707 dB** |
| [S863](series/S863/resultat.md) | 4/6 | 3/6 | 15,374 dB |
| [S864](series/S864/resultat.md) | 4/6 | 2/6 | non retenu |
| [S865](series/S865/resultat.md) | 4/6 | 3/6 | non évalué comparablement |
| [S866](series/S866/resultat.md) | 4/6 | **3/6** | 15,079 dB |
| [S867](series/S867/resultat.md) | audit commun | **3/6** | comparateur fixé |

**Avertissement :** 3/6 n'est **pas** « 50 % d'accuracy générale du transposeur ». Les six notes ont été réutilisées, les jeux ne sont pas indépendants par guitariste, et aucune polyphonie réelle nouvelle n'a été validée. Même sur ces notes, la similarité spectrale de S866 est moins bonne que celle de S860.

## Causalité, latence et indépendance

Chaque décision reçoit **256 nouveaux échantillons** de 44,1 kHz, jamais le futur. Tests du chemin d'inférence de prévisualisation passés localement (3 tests). Le FIR appris de S863 centre son énergie ~5,8 ms dans le passé ; S866 environ 1,2 à 3,9 ms. **Le temps de calcul p99 de 0,434 ms mesuré localement pour S863 ne démontre pas une latence acoustique nulle.** Aucun résultat de PC Windows/iRig ni F1 officiel NOTE n'est mesuré ici.

Les six têtes NOTE et le MIDI sont intacts : n'utiliser que `research/transpose256/s863_live_shadow256.py` comme code expérimental. Aucun déploiement, aucune intégration ni promotion.

## Inférence reproductible

Télécharger l'archive indiquée, en extraire les poids réels `s863.pt` / `s866.pt`, la source `guitare_originale.wav` et le script. Installer numpy, scipy, torch CPU ; exécuter :

```bash
python s863_live_shadow256.py --model s866.pt --input-wav guitare_originale.wav --output-wav essai_S866.wav
```

Le modèle ne calcule aucune hauteur MIDI et ne rejoue aucune banque ; il génère 256 nouveaux échantillons audio par bloc. Le code reste **une expérience**, pas un programme de scène approuvé. La compilation/test structurel GitHub ne remplace pas une comparaison musicale.



## Cadre méthodologique
Données : 39 vraies notes physiques GuitarSet de la banque S847, 27 TRAIN, 6 CAL et 6 notes OUTER déjà consultées dans les boucles antérieures. Les références +12 sont produites par transposition standard **hors ligne** ; jamais pendant l'inférence de l'IA. Les notes MIDI vraies servent aux pertes d'apprentissage et aux mesures, jamais au réseau de sortie. La réussite est définie par une erreur YIN absolue ≤0,5 demi-ton face à MIDI source+12. Ce petit score n'est pas une estimation fiable de la réussite musicale en général.

Exactement 256 échantillons PCM nouvellement reçus à 44,1 kHz ; uniquement le passé et le bloc acquis, sans futur ni file d'attente ajoutée. Ne pas confondre cela avec une latence acoustique nulle : les filtres causaux peuvent introduire du déphasage ou un retard de groupe. Les six têtes officielles NOTE et le MIDI ne sont pas modifiés ; **aucune promotion**.


## Recherche et limites
[BRAVE, modèle neural audio basse latence](https://fcaspe.github.io/brave/) ; [guitare avec reconnaissance de technique temps réel, EURASIP 2025](https://link.springer.com/article/10.1186/s13636-025-00413-6). Ces travaux illustrent le compromis contexte/latence et ne prouvent pas que notre architecture satisfait les contraintes Live.

### Étapes exigées avant toute promotion
Nouvelles prises jamais consultées ; validation de l'alignement JAMS/HEX ; tests véritables d'accords et d'intermodulation ; mesure des transitoires et retard acoustique exact sur Windows/iRig ; comparaison de hauteur **et** de fidélité ; aucun compromis accepté sur les autres têtes NOTE.
