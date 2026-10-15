# S631 — Vraie politique de latence adaptative entraînée (ATTENDRE/ÉMETTRE/REJETER)

**État : entraîné, audité et REJETÉ comme nouvelle référence.**

Source de vérité : [GitHub Actions run 38801078124](https://github.com/Andriamarosoa/note/actions/runs/38801078124), statut `success`, artifact `S631-genuine-trained-adaptive-latency-DEV-only`.

## Objectif et architecture

Entraîner effectivement un modèle causal qui choisit pour chaque candidat NOTE entre :
- **WAIT** : accumuler une nouvelle observation acoustique passée/actuelle ;
- **EMIT** : émettre maintenant ce NoteOn à l'heure effective de décision ;
- **REJECT** : rejeter la proposition et éviter un faux événement MIDI.

Origine des candidats : attaques montantes, à partir des sorties S614 et des caractéristiques PCM natives (pas de source externe). Trame HOP=256 à SR=44100. Budget d'observation borné à 28 trames (~162,54 ms). Chaque candidat possède son âge observé, son maximum de vraisemblance d'attaque/maintien, ses caractéristiques harmoniques courantes, ses évolutions et son contexte de comptage. **Aucune durée finale NoteOff n'est utilisée à l'inférence.**

Tête entraînée : `HistGradientBoostingClassifier` scikit-learn, 3 classes [WAIT,EMIT,REJECT]. Les étiquettes d'action sont des décisions *supervisées de référence approximatives* dérivées des véritables attaques et de pénalités de délai sur TRAIN ; ce **n'est pas** une politique d'arrêt optimal par renforcement, et la fonction de perte directe F1+lambda×latence n'a pas encore été entraînée.

Les états et probabilités observés par le contrôleur sont causaux. Les labels ne sont lus qu'au moment de l'entraînement ou de l'évaluation et jamais à l'inférence.

## Intégrité et séparation des données

- Original S614 SHA256 : `7ccf046e3a1870eaca3a108e6f9565298aeeadf43732a2f83559542c03f5f17f` **inchangé**.
- S80 historique : inchangé.
- Nouveaux poids S631 SHA256 : `b959d12a5cc4b573bdcc2cf1d7d8aaf4978b6fa26d1e4a181b94a2319d259a35`.
- 35 enregistrements TRAIN : 17 URMP et 18 GuitarSet.
- 17 DEV séparés par composition : 5 URMP, 12 GuitarSet. Jeux DEV déjà utilisés historiquement pour S614, **pas nouvellement indépendants**.
- **Aucune source du groupe publié TEST à 22 extraits n'a été ouverte pour sélectionner S631.**
- 93 632 exemples d'états de décision : 19 410 WAIT, 20 894 EMIT, 53 328 REJECT.
- CI : compilation, tests de causalité, de mémoire bornée et des contrats natifs S614 réussis.

## Validation DEV — vrais NoteOn émis et appariés à ±80 ms

| Variante | GuitarSet TP / FP / FN | GuitarSet F1 | URMP TP / FP / FN | URMP F1 |
|---|---|---:|---|---:|
| S614 référence | 998 / 585 / 582 | **63,10 %** | 243 / 149 / 256 | **54,55 %** |
| Émettre >=0,40 / rejeter >=0,60 | 1051 / 1471 / 529 | 51,24 % | 285 / 310 / 214 | 52,10 % |
| Émettre >=0,55 / rejeter >=0,75 | 1030 / 1090 / 550 | 55,68 % | 264 / 225 / 235 | 53,44 % |
| Émettre >=0,70 / rejeter >=0,60 | 908 / 670 / 672 | 57,50 % | 226 / 140 / 273 | 52,25 % |
| **Émettre >=0,70 / rejeter >=0,75** | **945 / 693 / 635** | **58,73 %** | **232 / 145 / 267** | **52,97 %** |

8 configurations évaluées. Aucune ne bat S614 simultanément sur GuitarSet et URMP. **Ne pas promouvoir.**

## Latence adaptative mesurée (depuis la naissance du candidat)

Sous le réglage émission 0,70 / rejet 0,75 / maximum 28 trames :

- GuitarSet DEV : médiane 0 ms, p95 **34,83 ms**, max **87,07 ms** ;
- URMP DEV : médiane 0 ms, p95 **28,73 ms**, max **52,24 ms**.

Ces durées sont le temps **supplémentaire entre naissance du candidat et émission de la note**, et **PAS** le délai total entre l'attaque physique réelle, la réception microphone et la sortie MIDI. Les scores F1 ci-dessus utilisent les temps effectifs de NoteOn émis.

La médiane à 0 ms montre que la politique a souvent choisi d'émettre immédiatement. Elle ne prouve pas qu'une attente adaptative a été bien exploitée.

## Verdict et prochaine étape

1. Le principe d'une latence par note est implémenté et une politique a été **réellement entraînée** sur des enregistrements annotés, avec une inférence strictement causale.
2. L'objectif de gain de précision n'est **pas atteint** : l'ajout de vrais NoteOn vient avec trop de faux positifs ; durcir le seuil supprime aussi des TP.
3. L'état actuel est une imitation supervisée de règles d'action issues des annotations : un meilleur apprentissage de la fonction de décision temporelle (risque de fausse note, coût de délai, valeur d'information d'une trame future), la calibration des probabilités et une meilleure tête acoustique de proposition seraient nécessaires.
4. Avant toute promotion : entraîner une véritable politique de coût/risque sur TRAIN, sélectionner sur DEV, geler la règle, puis évaluer sur **sources inédites** sans réutiliser le test S614 ouvert.

Le S614 officiel reste **69,03 % de F1 NoteOn ±80 ms sur les 22 sources de son test historique** ; aucune comparaison S631 sur ce test n'est revendiquée.

## Reproduire

```bash
export PYTHONPATH=.:src
python -m unittest -v test.test_strict_live test.test_s631_adaptive_latency
python -m scripts.s631_train_adaptive_policy --out /tmp/S631
```

Le rapport JSON et le checkpoint entraîné sont dans l'artifact du run ci-dessus.
