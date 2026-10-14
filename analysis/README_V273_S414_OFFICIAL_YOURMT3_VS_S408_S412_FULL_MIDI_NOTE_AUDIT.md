# S414 — VRAIE comparaison YourMT3+ / S408 / S412, même GuitarSet et mêmes notes

Date : 13 octobre 2026. Résultat : **YourMT3+ surpasse largement S408 et S412 pour l'identification MIDI, les onsets et les offsets**, particulièrement les notes de durée >=1 seconde. Ceci ne change **pas** les références officielles ni la référence S408 sélectionnée par l'utilisateur.

## Provenance exacte

- YourMT3+ checkpoint officiel `YPTF.MoE+Multi (noPS)`, poids SHA256 `ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25`, révision de source `5e66c1ea173a8186e0d20432b841d3180cc015b5`.
- **Ancien run YourMT3+ réellement exécuté** : [#37605163312](https://github.com/Andriamarosoa/note/actions/runs/37605163312), 4 artefacts `yourmt3-exactk-fold-0/1/2/4`. Chaque fold contient `notes.jsonl` (note MIDI, onset, offset, numéro de source, affectation native) produit par le décodage officiel avec `scripts/run_yourmt3_exactk.py`, sans post-traitement appris ni correction supplémentaire.
- **Mêmes 190 pistes GuitarSet** (players 00..04, fold 3/player05 exclus), mêmes 34 954 annotations JAMS officielles `note_midi`; archive MD5 `b39b78e63d3446f2e54ddb7a54df9b10`.
- Prédictions **S408 figées** et meilleur challenger **S412** : [#38375133317](https://github.com/Andriamarosoa/note/actions/runs/38375133317), artefact `11800518747`. S408/S412 utilisent le même K de S80, mêmes pitch S402, mêmes onsets S407 ; seul l'Offset varie.
- [NOUVEAU run S414 #38377420107](https://github.com/Andriamarosoa/note/actions/runs/38377420107) **success**, artefact `11801872144`. Code `scripts/s414_compare_official_yourmt3_full_notes_vs_S408_S412_guitarset.py`.
- Les sorties de YourMT3+ n'ont pas été reconstruites, synthétisées ni réexécutées : leurs notes et durées archivées ont été utilisées **sans modification**. Les poids de S80/S402/S407/S408/S412 ne changent pas.

## Protocole commun, différentes prédictions

Sur chaque piste complète, associer **une seule fois** une note prédite à une vraie note de **même hauteur MIDI** avec onset absolu à <=50 ms. Mesurer :
- Onset+pitch F1 : TP appariées / nombres de notes prédites et vraies, F1 = 2TP/(Np+Ng).
- Offset correct conditionné : parmi les notes appariées par pitch+onset, Offset à <=50 ms.
- **F1 note complète** : même appariement de hauteur+Onset ET Offset à <=50 ms, F1=2TP_complet/(Np+Ng).
- Contrôle : Offset autorisé à <=max(50 ms, 20% de la durée réelle).
- Durées et comparaisons par classe temporelle.
- YourMT3+ est évalué une fois sur **toutes les notes effectivement décodées**, même hors événements natifs (éviter une sélection avantageuse). Une variante ne garde que les notes attribuées aux fenêtres natives pour vérifier l'effet du périmètre.

Ce benchmark est **comparabilité musicale** (mêmes WAV GuitarSet et JAMS), et non **égalité complète d'architecture et d'audio d'entrée** : YourMT3+ reçoit le pickup mix complet et un large contexte de décodage ; S408 est dérivé de S258 pickup mix (fenêtres locales) et S412 ajoute la source microphone mono continu. Les latences sont différentes. YourMT3+ a inclus GuitarSet dans son corpus de pré-entraînement, S80 comporte également un recoupement historique : **aucune conclusion indépendante sur de nouvelles musiques**.

## Résultats vérifiés

| Mesure | S408 actif | S412 challenger | YourMT3+ officiel toutes notes | YourMT3+ affectées aux groupes natifs seulement |
|---|---:|---:|---:|---:|
| Notes prédites | 33 475 | 33 475 | 34 781 | 33 218 |
| Notes JAMS vraies | 34 954 | 34 954 | 34 954 | 34 954 |
| Notes Pitch+Onset appariées <=50ms | 25 120 | 25 120 | 30 594 | 29 414 |
| Pitch+Onset F1 | 73,4192 % | 73,4192 % | **87,7436 %** | 86,2935 % |
| Offset juste <=50ms parmi Pitch+Onset appariées | 13 050 | 14 313 | **26 252** | 25 233 |
| **Offset exact conditionnel** | 51,9506 % | 56,9785 % | **85,8077 %** | 85,7857 % |
| **F1 note complète <=50ms** | 38,1417 % | 41,8331 % | **75,2907 %** | 74,0275 % |
| F1 note complète onset<=50ms, Offset<=max(50ms,20% durée) | 41,8887 % | 46,0916 % | **76,9943 %** | 75,7261 % |
| Offset MAE sur notes appariées | 105,43 ms | 95,87 ms | **41,79 ms** | 41,74 ms |

**Différence principale vs meilleur challenger S412** : YourMT3+ **+33,46 points** de F1 note complète ; **+28,83 points** d'Offset conditionnel ; +14,32 points de Pitch+Onset F1.

## Fins de notes : le défaut critique de notre modèle

| Durée annotée (vrai GT) | S408 | S412 | YourMT3+ |
|---|---|---|---|
| <100 ms | 503/1 247 = 40,34 % | 630/1 247 = 50,52 % | 1 281/1 622 = **78,98 %** |
| 100–499 ms | 12 305/20 385 = 60,36 % | 13 248/20 385 = 64,99 % | 21 792/24 869 = **87,63 %** |
| 500–999 ms | 242/2 751 = 8,80 % | 435/2 751 = 15,81 % | 2 614/3 273 = **79,87 %** |
| >=1000 ms | 0/737 = **0 %** | 0/737 = **0 %** | **565/830 = 68,07 %** |

Les différents dénominateurs reflètent des notes effectivement appariées par Pitch+Onset pour chaque prédicteur. Ce tableau n'est **pas** une comparaison conditionnelle à un ensemble commun de notes, mais le benchmark de note complète ci-dessus est évalué sur la **même vérité totale de 34 954 notes** avec le même protocole, et donc comparable au niveau du système.

## Lecture architecturale, ce que l'on sait vs ce que l'on suppose

1. **Observation mesurée** : YourMT3+ identifie davantage de hauteurs MIDI au bon moment ET maintient les offsets bien plus précisément. S408/S412 terminent trop tôt les notes longues.
2. **Hypothèse plausible, non encore démontrée par ablation sur note** : le grand contexte spectrotemporel et l'encodage/décodage en événements MIDI du Transformer autoregressif aident à maintenir la durée, en plus d'une perception du mix améliorée. L'ancienne ablation YourMT3+ Exact-K sur encodeur linéaire vs décodeur T5 montre une différence, **pas** une preuve suffisante sur le mécanisme des Offsets.
3. **Prochaine architecture à tester** : Ne pas dupliquer le comptage Exact-K S80 ; le K prédit reste une entrée de la prédiction de notes. Ajouter aux têtes de notes un **suivi de note active sur plusieurs secondes et événements NoteOn/NoteOff liés**, plus une mémoire et attention sur le flux STFT y compris K0. Évaluer à composition disjointe ; hauteurs, débuts, durées et échecs >=1s ; réaliser ablations de mémoire, attention et état par hauteur/voix.
4. **Méthode incontournable** : évaluer sur de nouveaux WAV et nouveaux morceaux non présents dans le pré-entraînement de **aucun** comparateur pour conclure à une généralisation. Ne pas confondre les chiffres Exact-K historiques (YourMT3+ 86,54% global / 54,54% poly) avec ceux d'Offset et de note complète mesurés ici.

**Décision** : l'utilisateur a choisi S408 comme référence Offset de développement, elle reste inchangée. S412 est le meilleur challenger interne actuel. YourMT3+ devient **l'étalon comparatif externe mesuré** pour la qualité de transcription complète, et aucun score de note complète plus élevé que 41,83 % n'est revendiqué pour le système `note` à ce stade.
