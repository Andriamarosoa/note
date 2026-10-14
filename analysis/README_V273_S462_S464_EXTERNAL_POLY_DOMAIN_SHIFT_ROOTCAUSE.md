# S462–S464 — Pourquoi l'Exact-K polyphonique perd en précision sur IDMT ?

**Source:** Andriamarosoa/note ; 14 octobre 2026. **Type:** diagnostic rétrospectif en lecture seule. Aucun entraînement, aucun seuil de modèle ajusté à IDMT, aucune promotion ni modification Live/onset/offset.

## Ce qui est réellement comparé

- Référence GuitarSet reconstruction S397 / S80 : 53 529/59 309 = 90,25 % Exact-K, et 6 054/7 385 = 81,98 % vrai poly. Les modèles S80 ont un historique d'entraînement/exposition sur GuitarSet, donc **pas** un test compositionnel indépendant au sens strict.
- IDMT-SMT-GUITAR V2 Dataset3 : 5 premiers extraits audio de **10 secondes** de cinq WAV, ré-échantillonnés mono à 44,1 kHz par la procédure S277 V88 ; vrais poids originaux S80 S297 rejoués. Ces données ont déjà été évaluées en S299 ; pas un nouveau jeu aveugle.
- Parmi **506 clusters d'événements natifs V88**, S451/S453 retrouvent 47 attaques XML appariées, 45 dont toutes les hauteurs sont dans la plage MIDI40–88 ; sur celles-ci S80 réussit 25/45 Exact-K et sur les 21 vrais poly K2–K4 réussit 7/21. Les **459 événements XML non appariés ne sont jamais traités comme de vrais K0**.

## S462 — problème avéré de fragmentation / surproduction des événements acoustiques

[GitHub Actions S462 : succès](https://github.com/Andriamarosoa/note/actions/runs/38546964936).

Le regroupement des attaques XML utilise strictement **20 ms**, comme S451. L'algorithme historique assigne chaque groupe XML une seule fois à un événement V88 à ±**80 ms**. La nouvelle recherche **optimale un-à-un** (matching biparti, couverture maximale) indépendamment appliquée aux vrais horodatages confirme exactement :

| Mesure, 5 extraits × 10 s | Valeur vérifiée |
|---|---:|
| Attaques XML regroupées dans la partie 0–10 s | **47** |
| Attaques XML appariées par méthode originale S451 | **47** |
| Attaques appariées par nouvel optimum un-à-un 80 ms | **47** |
| Clusters/événements natifs V88 | **506** |
| Événements proches (≤80 ms d'une attaque XML) | **88** |
| Événements à plus de 80 ms de chaque attaque XML | **418** |
| Parmi ces événements éloignés, prédits K0 par S80 | **417** |
| Total des événements prédits K0 par S80 | **456** |

Répartition exacte (événements natifs / attaques XML / événements prédits K0) : **140/6/134 ; 15/13/1 ; 88/7/81 ; 98/12/84 ; 165/9/156**. Dans quatre des cinq enregistrements, il y a environ 9 à 23 événements V88 par attaque XML. La cinquième (15/13/1) se comporte différemment : l'ampleur du phénomène **dépend fortement de l'enregistrement**.

**Ce que S462 prouve :** sur les segments IDMT, la chaîne native V88 produit beaucoup d'événements sans attaque instrumentale XML proche et S80 attribue K0 à la quasi-totalité de ceux éloignés des annotations. La très faible couverture des 506 events par les annotations n'est **pas** le résultat d'un décalage global de synchronisation qui masquerait les attaques : toutes les 47 attaques regroupées ont été retrouvées avec la tolérance existante. Il faut conserver la distinction **détection d'événements** et **classification K** dans toute future métrique.

**Ce que S462 ne prouve pas :** les 418 événements éloignés ne sont pas forcément des faux événements *acoustiques* (résonances, bruit, sustain, sons non annotés), ni des vrais K0. Un examen d'énergie waveform est nécessaire pour les qualifier.

## Inspection du code original : comment V88 produit ces candidats

Le flux réel dans `scripts/evaluate_v273_fresh_player05.py` utilise `runtime_records`, qui invoque `_candidate_groups` de `scripts/train_v86_state_transition_proposals.py`. Ce générateur :

- conserve **chaque maximum local** du flux de probabilité d'attaque V8.4 si son score dépasse un `floor` historiquement choisi sur GuitarSet ;
- ajoute aussi les **fronts montants du score dépassant 0,40** ;
- ne fusionne dans cette phase que les pics séparés d'au plus **4 échantillons audio** (≈0,091 ms à 44,1 kHz), ce qui ne constitue pas une détection de l'absence de retrigger pendant tout le sustain ;
- le constructeur de V90 (cf. `scripts/train_v90_structured_cluster_cardinality.py`) regroupe ensuite ces propositions selon une fenêtre de **40 ms ancrée au premier candidat**, sans critère explicite exigeant une nouvelle augmentation d'énergie physique au lieu d'un son maintenu.

Ces propriétés sont directement vérifiables dans les sources, pas inférées d'un résultat de score. Elles offrent un **mécanisme précis à tester** : un même son soutenu comportant des oscillations locales du score d'attaque peut créer une succession de groupes V88, dont S80 estime ensuite K0. Un tel événement reste présent dans les fichiers natifs même si aucune note n'est finalement émise.

La source IDMT n'exigeait **aucun rééchantillonnage** des cinq fichiers WAV originaux : tous sont 44,1 kHz PCM16, quatre mono et un stéréo converti en mono. Un écart de fréquence d'échantillonnage ne saurait donc expliquer les résultats. L'effet d'une éventuelle stéréophonie n'explique pas non plus la forte surproduction sur plusieurs WAV déjà mono.

**Ne pas remplacer ce constat par un filtre K0 naïf** : 7/45 vrais événements annotés en plage MIDI sont précisément mal classés K0 par S80, dont 3/21 accords polyphoniques. Les éliminer détruirait de vrais onsets.

## S463 — décalage acoustique conditionnel au vrai K et lié aux erreurs

[GitHub Actions S463 : succès](https://github.com/Andriamarosoa/note/actions/runs/38547288114).

S463 réutilise **les mêmes vecteurs originaux** 58D résumé, 245D flux, 490D morphologie, 294D physique, 3D géométrie, enregistrés par S258 GuitarSet et S283 IDMT. Aucune transformation ML supplémentaire. L'audit calcule pour chaque **vrai K** les percentiles des caractéristiques GuitarSet et la proportion des valeurs IDMT hors de l'intervalle [P01; P99] de leur K réel.

Sur les **45 événements MIDI éligibles IDMT** (S80 K bon 25, faux 20), proportions des caractéristiques hors du support fréquent de GuitarSet chez les deux groupes :

| Caractéristiques | S80 K correct (25) | S80 K faux (20) |
|---|---:|---:|
| Résumé58 | **0,76 %** | **10,17 %** |
| Flux245 | 1,42 % | 3,22 % |
| Morphologie490 | 1,47 % | 2,86 % |
| Physique294 | 1,93 % | 4,05 % |
| Géométrie3 | 1,33 % | 1,67 % |

Ces métriques font apparaître un **décalage des caractéristiques acoustiques, particulièrement celles des résumés**, plus important parmi les événements dont le K est erroné. La comparaison est stratifiée par vrai K, pour ne pas attribuer automatiquement à la source des différences de proportions K1/K2/K3/K4.

L'observation reste descriptive et non causale. Les caractéristiques sont corrélées et 20 erreurs constituent un faible effectif. La comparaison de distributions ne démontre pas à elle seule quelle variable corrigerait ces 20 erreurs.

## Interprétation : deux phénomènes distincts, simultanés

1. **Excès de clusters événementiels** sur quatre WAV, majoritairement neutralisés en K0 par S80. Ne pas intégrer les 418 events hors annotation dans le calcul de l'Exact-K comme vrai K0 ; la segmentation/acquisition doit être audité indépendamment.
2. **Difficulté de classifier le K même aux attaques annotées** (20 K incorrects sur 45). Les 58 variables de résumé sont plus atypiques par rapport à GuitarSet dans ces erreurs, ce qui rend plausible un problème de normalisation/timbre/harmoniques. S453 montrait que les erreurs existaient dès V88/S5 : 4/21 poly corrects V88 → 7/21 S80. Un correcteur final statique ne résout pas ce problème.

Les scores 81,98 % (GuitarSet) et 33,33 % (IDMT) diffèrent de **48,65 points**, mais ils n'ont pas le même mode d'échantillonnage ni le même statut d'exposition d'entraînement. Ce n'est **pas** une estimation valide de l'effet causal du micro ou de l'instrument.

## Diagnostic complémentaire S464

Script `scripts/s464_validate_IDMT_farK0_audio_rms_against_recording.py`, workflow `.github/workflows/v273-s464-audio-RMS-on-excess-IDMT-K0-events.yml`. L'audit compare sans réentraînement le **RMS de chaque vrai WAV original vérifié par SHA**, le contraste attaque/préattaque, les événements annotés proches et les 417 K0 prédits à plus de 80 ms des attaques. Il ne filtre pas d'événements et ne choisit aucun seuil de correction.

Son verdict ne doit être recopié ici **qu'après succès vérifié**.

## Décision et prochains critères

**Ne pas promouvoir S462–S464 comme une nouvelle architecture K.** Avant de concevoir S465 (un classificateur résiduel), auditer sur *un seul protocole commun* et de nouveaux WAV jamais annotés au moment de geler les modèles :

- rappel des vrais onsets et nombre d'extras par minute, avec définition de l'événement de référence ;
- précision du K conditionnelle à une attaque XML correctement appariée, par K1/2/3/4/5/6 ;
- comparaison acoustique par type de source et vrai K, par composition et par niveau d'énergie ;
- évaluation des régressions et correctifs **par transitions K2↔K3, K3↔K4**, et des erreurs K0 hors véritables événements ;
- vérification que la chaîne `son -> E3 / accord [E3,G3,B3]` ne détériore ni S80 ni la précision des notes en fin de traitement.

**L'ordre des faits importe** : le nombre élevé d'événements natifs concerne la segmentation V88 ; les erreurs 20/45 concernent la classification K sur de vraies attaques. Les résoudre requiert potentiellement deux mécanismes différents.
