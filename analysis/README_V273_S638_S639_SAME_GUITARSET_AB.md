# S638/S639 — NOTE vs MIDI : vrai F1 causal sur le même GuitarSet

**Branche :** `codex/s634-midi-live-transport`.

## Protocole gelé

- **Modèle NOTE :** S614 natif original SHA256
  `7ccf046e3a1870eaca3a108e6f9565298aeeadf43732a2f83559542c03f5f17f`.
- **Modèle MIDI :** V2.2.1 TFLite original SHA256
  `4a4df49d34c1fd359fab98c20bb598f37c7f42f22493c70923a2e829176df744`.
- **Code MIDI d'origine :** révision
  `536536f54290fdc4f63db28aa063d2e324121642`.
- **Sources :** GuitarSet WAV microphone réels + annotations JAMS
  `note_midi`, téléchargement vérifié par empreintes SHA256.
- **Prétraitement :** un **unique** WAV de 44 100 Hz mono PCM16 écrit
  une fois, lu par les deux modèles. Aucune retouche temporelle
  ni séparation d'instrument par oracle.
- **Inférence MIDI :** décodeur exact `src.polyphonic.transcribe`,
  paramètres V2.2.1 gelés, sans automatic model level, gain=1.
- **Inférence NOTE :** `StrictLiveStream` S614, sans assistant S632.
  Le S80 historique reste préservé.
- **Score :** S635 common original MIDI causal matcher,
  intervalles de délai 12/18/24/35/46/80/250 ms, NoteOn hauteur
  identique, appariement 1:1 sans reconnaissance anticipée.
- **Plage de référence commune** : MIDI 40–76. Les vérités et
  prédictions hors plage sont comptées séparément, puis exclues
  symétriquement du F1 comparatif.
- Pas d'alignement rétrospectif des sorties MIDI après le fait ;
  les temps sont les temps d'émission du décodeur originel.

### Test pilote — trois compositions DEV distinctes

[CI S638 pilote : réussite](https://github.com/Andriamarosoa/note/actions/runs/38811275351)
[Rapport JSON du pilote](https://github.com/Andriamarosoa/note/actions/runs/38811275351/artifacts/12009170702).

Extraits :
- `04_BN3-119-G_comp` : 135 références, 106 NoteOn NOTE, 109 MIDI ;
- `04_Funk3-112-C#_comp` : 283 références, 285 NOTE, 216 MIDI ;
- `04_Rock1-130-A_comp` : 190 références, 195 NOTE, 92 MIDI.

Trois fichiers audio vérifiés chacun par SHA256 commun entre modèles.

| Score causal 80 ms | NOTE S614 | MIDI V2.2.1 |
|---|---:|---:|
| TP | **329** | 73 |
| FP | **257** | 344 |
| FN | **279** | 535 |
| Précision | **56,14 %** | 17,51 % |
| Rappel | **54,11 %** | 12,01 % |
| F1 | **55,11 %** | 14,24 % |

F1 à 35 ms : NOTE **37,86 %**, MIDI **7,22 %**.

**Résultat direct vérifié :** NOTE S614 est nettement meilleur que
MIDI V2.2.1 sur **ces trois extraits précis** sous le protocole
causal S635 commun, sans mélange de benchmarks historiques.

### Limites cruciales

Ce pilote est un **DEV déjà exposé historiquement** au projet NOTE,
et les intersections exactes avec l'ensemble d'entraînement de
MIDI ne sont pas auditées. Il ne peut donc pas être présenté
comme une nouvelle évaluation indépendante ni comme une preuve
de supériorité pour tout instrument, toute captation ou tout style.

**Ne pas confondre** le F1 original S614 de 69,03 %
sur son TEST historique (tolérance NoteOn ±80 ms symétrique)
avec ce F1 causal retard-only obtenu sur trois extraits
différents. Le protocole S635 exige que le NoteOn émis
n'arrive pas avant l'attaque de référence ; cela explique
une partie de la différence de métriques.

Les décisions de promotion du modèle, les poids S80/S614/S632
et leurs seuils restent inchangés. Il reste à tester de
nouveaux enregistrements non utilisés et à mesurer
la latence microphone→MIDI sur matériel physique.

## Extension S638 et audit S639

Le workflow d'extension S638 rejoue les 12 morceaux
GuitarSet DEV (trois compositions, quatre variations chacune),
sans ajuster les paramètres après le pilote.
Les artefacts comprennent un manifeste avec les prédictions
événementielles et les labels, des statistiques par clip
et les scores aggrégés pour toutes les échéances.

`scripts/s639_audit_same_audio_disagreement.py` examinera
ensuite les vrais événements de référence appariés par les deux
moteurs, par NOTE seulement, par MIDI seulement ou manqués
par les deux, ainsi que les registres graves/médium/aigus et la
densité locale des attaques. La densité locale d'attaques
n'est **pas** l'Exact-K S80 historique.

## S638 — Bilan intégral des 12 enregistrements (terminé)

[CI complète réussie](https://github.com/Andriamarosoa/note/actions/runs/38811643000).
[Manifeste, prédictions, métriques et rapport JSON](https://github.com/Andriamarosoa/note/actions/runs/38811643000/artifacts/12008897339).

**12** véritables enregistrements GuitarSet DEV regroupés
dans **3 compositions distinctes** avec leurs rôles
d'accompagnement/solo. **1 580** vraies notes sur
la plage commune MIDI 40–76, mêmes WAV PCM-16 et
SHA256 entre les deux moteurs pour chaque source.

| Métrique F1 causal, 80 ms | NOTE S614 | MIDI V2.2.1 |
|---|---:|---:|
| TP | **901** | 397 |
| FP | **682** | 1 150 |
| FN | **679** | 1 183 |
| Precision | **56,92 %** | 25,66 % |
| Recall | **57,03 %** | 25,13 % |
| F1 | **56,97 %** | 25,39 % |

À 35 ms : NOTE **39,84 %** (630 TP, 953 FP,
950 FN) ; MIDI **17,59 %** (275 TP, 1 272 FP,
1 305 FN).

Cette fois les deux modèles ont vu les mêmes 12 vrais audios
et ont été évalués avec la même règle de score : le
résultat est comparable **sur ces morceaux et pour ce protocole**.
Il ne prouve aucune généralisation sur des nouvelles compositions.
Les scores historiques NOTE 69,03 % et MIDI 17,51 % ne doivent
pas remplacer ces mesures.

### Résultats par enregistrement, NoteOn F1 à 80 ms

| Enregistrement | NOTE S614 | MIDI V2.2.1 |
|---|---:|---:|
| BN3-119-G accompagnement | 63,07 % | 18,03 % |
| BN3-119-G solo | 72,07 % | 48,95 % |
| BN3-154-E accompagnement | **18,87 %** | 6,04 % |
| BN3-154-E solo | 82,58 % | 53,19 % |
| Funk3-112-C# accompagnement | 50,00 % | 14,83 % |
| Funk3-112-C# solo | 71,52 % | 45,10 % |
| Funk3-98-A accompagnement | 60,37 % | 9,72 % |
| Funk3-98-A solo | 76,92 % | 46,15 % |
| Rock1-130-A accompagnement | 57,66 % | 9,93 % |
| Rock1-130-A solo | 58,24 % | 41,21 % |
| Rock1-90-C# accompagnement | 56,25 % | 26,45 % |
| Rock1-90-C# solo | **83,33 %** | 53,42 % |

Les accompagnements (accords/polyphonie) et notamment
BN3-154-E sont un défaut manifeste à investiguer :
NOTE tombe à 18,87 % sur cet extrait, alors qu'il
atteint 82,58 % sur la version solo correspondante.
C'est une piste de conception, pas une preuve causale
suffisante d'un défaut unique.

## S639 — Audit des corrections et désaccords note par note

[Workflow S639 validé](https://github.com/Andriamarosoa/note/actions/runs/38812452658).
[Rapport de taxonomie vérifié](https://github.com/Andriamarosoa/note/actions/runs/38812452658/artifacts/12008748131).

Le script a téléchargé **les événements figés de S638**,
recalculé indépendamment le score, et vérifié
les invariants de partition des 1 580 annotations
pour la tolérance causale 80 ms :

| Statut de note vraie | Nombre |
|---|---:|
| NOTE ET MIDI corrects | **331** |
| NOTE correct, MIDI manqué | **570** |
| MIDI correct, NOTE manqué | 66 |
| Les deux manquent | **613** |
| **Total** | **1 580** |

Les **504 TP nets** d'avance de NOTE représentent
570 récupérations qui lui sont propres contre
66 qui sont propres à MIDI. Cela n'implique PAS que
MIDI puisse apporter 66 corrections triviales à
NOTE : il faudrait apprendre un sélecteur dont les
caractéristiques peuvent distinguer ces cas sans
nouveaux FP.

### Registres à améliorer

| Registre (MIDI) | NOTE TP/FP/FN | NOTE F1 | MIDI F1 |
|---|---|---:|---:|
| Grave 40–52 | 177 / 193 / 232 | **45,44 %** | 14,78 % |
| Médium 53–64 | 535 / 396 / 339 | **59,28 %** | 29,74 % |
| Aigu 65–76 | 189 / 93 / 108 | **65,28 %** | 33,12 % |

Le grave reste donc le registre le plus difficile pour
NOTE sous la métrique commune, malgré son avance sur MIDI.

## Décision

**Aucune promotion du système de reconnaissance à l'issue de S638/S639**.
S80, S614, S632 et MIDI V2.2.1 restent à leurs versions d'origine.
Le transport S634/S636 reste expérimental pour le matériel réel.

L'étape pertinente est un audit causal acoustique des
613 notes manquées par les deux, en particulier :

- relever si une attaque réelle a été candidate et rejetée,
  ou si les têtes n'ont jamais produit son signal ;
- séparer les accompagnements des parties solo et examiner
  BN3-154-E ;
- par registre, distinguer fondamentales graves, harmoniques
  concurrentes et notes simultanées ;
- conserver les 901 vrais événements NOTE existants et mesurer
  chaque nouveau faux positif avant toute expérience de promotion.

**Objectif 80 % F1 LIVE toujours non atteint**. La valeur
officielle reste 69,03 % sur le TEST historique du projet NOTE ;
le nouveau protocole commun donne 56,97 % sur DEV et
ne remplace pas la valeur officielle.

