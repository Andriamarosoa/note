# S606–S609 — Validation de S601 sur GuitarSet, IDMT et MAESTRO

**Le même S601 figé atteint 80,06 % sur GuitarSet microphone, 82,11 % sur IDMT Dataset3 et 76,26 % sur MAESTRO. Les 80 % ne sont donc pas validés de façon générale.** Le score GuitarSet dépasse de justesse le seuil en moyenne ; accompagnement, Rock et trois guitaristes restent sous 80 %. Le score IDMT porte sur seulement 88 notes et cinq prises du même instrument.

Les 84,91 % de S605 restent le résultat des cinq extraits URMP réservés. Il ne faut pas en déduire une qualité équivalente sur tous les instruments. Aucun apprentissage, ajustement de seuil, remplacement de source ou décalage temporel n’a été effectué pour les nouveaux résultats ci-dessous.

[Protocole enregistré avant les données](README_V273_S606_S609_FROZEN_TRANSFER.md) · [Liste exacte des 77 fichiers](v273_s606_cross_corpus_preregistration.json) · [Première CI réussie](https://github.com/Andriamarosoa/note/actions/runs/38688389092) · [Provenance et horodatages](evidence/v273-s606-s609/CI_provenance.json)

## Résultats principaux : NoteOn, hauteur MIDI40–88 exacte, ±80 ms

F1 micro sur tous les événements des extraits, avec appariement un à un et comptage de tous les faux positifs et faux négatifs. Les trois politiques traitent exactement le même audio et la même vérité.

| Corpus | Fichiers | Durée évaluée | Vraies notes | BP par défaut | S600 global figé | S601 figé |
|---|---:|---:|---:|---:|---:|---:|
| GuitarSet | 60 | 1280.48 s | 7498 | 76,11 % | 79,25 % | **80,06 %** |
| IDMT | 5 | 70.62 s | 88 | 72,15 % | 74,07 % | **82,11 %** |
| MAESTRO | 12 | 288.00 s | 3301 | 70,70 % | 75,47 % | **76,26 %** |

| Corpus, S601 | TP | FP | FN | Précision | Rappel |
|---|---:|---:|---:|---:|---:|
| GuitarSet | 5711 | 1058 | 1787 | 84,37 % | 76,17 % |
| IDMT | 78 | 24 | 10 | 76,47 % | 88,64 % |
| MAESTRO | 2083 | 79 | 1218 | 96,35 % | 63,10 % |

**Ce sont des mesures hors ligne de NoteOn uniquement.** Elles ne valident ni NoteOff, ni Exact-K complet, ni latence microphone→MIDI, ni fonctionnement causal. Les fonctionnalités Live et la référence S80 restent inchangées.

## La moyenne GuitarSet masque des limites

| Partie | BP par défaut | S600 global | S601 |
|---|---:|---:|---:|
| Accompagnement | 72,56 % | 75,62 % | 76,08 % |
| Solo | 84,35 % | 87,42 % | 88,58 % |

| Style du nom de fichier | BP par défaut | S600 global | S601 |
|---|---:|---:|---:|
| BN | 79,36 % | 80,76 % | 83,79 % |
| Funk | 74,82 % | 81,05 % | 79,01 % |
| Jazz | 78,44 % | 81,83 % | 83,24 % |
| Rock | 76,44 % | 76,52 % | 74,26 % |
| SS | 72,54 % | 76,82 % | 83,25 % |

Le Rock régresse de 76,44 à 74,26 % ; le gain global ne doit pas le cacher. La F1 augmente sur 46 fichiers, diminue sur 13 et reste identique sur un. Le guitariste 00 atteint 74,85 %, 02 atteint 79,21 % et 04 atteint 78,95 %. Les trois autres dépassent 80 %, sans que cela certifie d’autres guitares physiques.

| Guitariste | BP par défaut | S601 |
|---|---:|---:|
| 00 | 72,84 % | 74,85 % |
| 01 | 76,04 % | 82,27 % |
| 02 | 77,39 % | 79,21 % |
| 03 | 78,37 % | 82,66 % |
| 04 | 73,41 % | 78,95 % |
| 05 | 80,11 % | 83,98 % |

## IDMT Dataset3 : cinq prises, un seul instrument

Les cinq enregistrements totalisent 70,62 s. Ils sont tous plus courts que la limite de 24 s : leurs enregistrements entiers sont donc évalués. `pathetique_poly` est effectivement stéréo et a été moyenné en mono selon la règle fixée ; les quatre autres fichiers sont mono, tous en PCM16 à 44,1 kHz. La [description officielle](https://zenodo.org/records/7544110) indique que ces cinq prises utilisent le même instrument. **Ce n’est pas une validation des sept guitares de la collection complète.**

| Prise IDMT | BP par défaut | S601 |
|---|---:|---:|
| nocturneNr2 | 65,38 % | 75,56 % |
| pathetique_mono | 94,74 % | 100,00 % |
| pathetique_poly | 65,22 % | 81,08 % |
| promenade_mono | 92,86 % | 100,00 % |
| quintfall | 67,57 % | 75,00 % |

Les deux prises nommées `mono` atteignent 100 % sur 22 vraies notes ; les trois autres prises, réunies, atteignent seulement 76,71 %. Les versions mono et poly de Pathétique ne constituent pas deux compositions indépendantes.

## Piano : rappel insuffisant et deux régressions

S601 atteint 96,35 % de précision, mais seulement 63,10 % de rappel sur 3 301 notes. Il réduit les faux positifs de 516 à 79 tout en perdant quatre vrais positifs nets. Dix extraits progressent en F1 face à BP et deux régressent : Schubert (53,75→46,50 %) et Rachmaninoff (70,70→63,67 %).

| Compositeur, extrait de 24 s | BP par défaut | S600 global | S601 |
|---|---:|---:|---:|
| Alexander Scriabin | 72,28 % | 82,11 % | 92,68 % |
| Claude Debussy | 64,79 % | 69,44 % | 96,15 % |
| Domenico Scarlatti | 68,78 % | 79,20 % | 75,73 % |
| Franz Liszt | 76,97 % | 83,89 % | 87,95 % |
| Franz Schubert | 53,75 % | 53,85 % | 46,50 % |
| Frédéric Chopin | 80,13 % | 82,31 % | 93,06 % |
| Johann Sebastian Bach | 79,42 % | 89,52 % | 97,26 % |
| Joseph Haydn | 77,28 % | 86,72 % | 86,84 % |
| Ludwig van Beethoven | 73,96 % | 78,22 % | 80,35 % |
| Robert Schumann | 61,36 % | 67,22 % | 66,81 % |
| Sergei Rachmaninoff | 70,70 % | 66,15 % | 63,67 % |
| Wolfgang Amadeus Mozart | 83,65 % | 87,69 % | 91,37 % |

## Toutes les corrections et régressions de S601 face à BP

| Corpus | Vraies notes gagnées | Vraies notes perdues | FP supprimés | FP ajoutés | Variation TP | Variation FP |
|---|---:|---:|---:|---:|---:|---:|
| GuitarSet | 504 | 475 | 1169 | 476 | +29 | -693 |
| IDMT | 0 | 1 | 36 | 8 | -1 | -28 |
| MAESTRO | 304 | 308 | 470 | 33 | -4 | -437 |

Les fichiers `individual_note_audit` contiennent chaque vraie note gagnée ou perdue, identifiée par source, hauteur et temps. Le rejeu hors ligne rattache chaque note correcte de BP perdue au candidat d’origine : 457 des 475 pertes GuitarSet, l’unique perte IDMT et 301 des 308 pertes MAESTRO ont une probabilité inférieure au seuil figé de 0,5. Les autres relèvent de l’espacement des prédictions ou de la compétition dans l’appariement.

| Faux négatifs de S601 | Aucun candidat à ±80 ms | Tous les candidats sous 0,5 | Espacement/appariement | Total FN |
|---|---:|---:|---:|---:|
| GuitarSet | 281 | 1426 | 80 | 1787 |
| IDMT | 8 | 2 | 0 | 10 |
| MAESTRO | 95 | 1099 | 24 | 1218 |

Ce diagnostic est descriptif : aucun seuil n’a été changé après observation des erreurs. Les 1 099 faux négatifs piano disposant seulement de candidats sous le seuil montrent que le filtrage appris sur URMP est une priorité de recherche ; ils ne garantissent pas qu’abaisser le seuil améliore la F1, car cela réintroduirait aussi des faux positifs.

## Réattaques et tolérance temporelle

Les paires ci-dessous proviennent des annotations : même hauteur et même corde/canal, écart strictement inférieur à 100 ms. Elles ne constituent pas un audit acoustique manuel de chaque réattaque. Une paire est comptée récupérée seulement si ses deux notes sont appariées à ±80 ms.

| Corpus | Paires annotées | Deux notes retrouvées par BP | Par S600 global | Par S601 |
|---|---:|---:|---:|---:|
| GuitarSet | 108 | 4 | 31 | 9 |
| IDMT | 0 | 0 | 0 | 0 |
| MAESTRO | 54 | 0 | 25 | 3 |

Le filtre S601 fait moins bien que S600 sur ces paires rapides. L’absence de telles paires dans le petit test URMP de S605 ne permettait pas de mesurer cette faiblesse.

| Corpus, F1 S601 | ±30 ms | ±50 ms | ±80 ms |
|---|---:|---:|---:|
| GuitarSet | 71,82 % | 75,99 % | 80,06 % |
| IDMT | 70,53 % | 78,95 % | 82,11 % |
| MAESTRO | 69,41 % | 73,29 % | 76,26 % |

Le seuil de 80 % n’est atteint sur aucun des trois lots à ±50 ms. Il ne faut pas confondre la tolérance ±80 ms utilisée ici avec la métrique ±50 ms de nombreux articles, ni avec une mesure de latence Live.

## Indépendance, annotations et reproductibilité

Le contrat S602, le checkpoint S601 et ses 168 variables sont inchangés : seuil 0,5, espacement 100 ms, aucun décalage. Les 77 sources ont été fixées au commit `6372069c3d200486c39e6e7c283677b1ace7defe` avant l’inférence de transfert. Les trois gels CI précèdent le début du score à **05:54:21 UTC** le 15 octobre 2026. Le premier essai CI a réussi ; les reprises réseau n’ont entraîné ni exclusion ni remplacement de source.

La tête S601 a été entraînée sur URMP uniquement. GuitarSet et MAESTRO font partie des corpus d’apprentissage du réseau de base selon le [tableau 1 de l’article Basic Pitch](https://arxiv.org/pdf/2203.09893). Le recoupement exact avec les poids distribués n’est pas établi ici. IDMT et GuitarSet ont également servi à des expériences historiques du projet. Il s’agit donc d’un **transfert de la nouvelle tête figée**, pas d’une preuve d’indépendance totale vis-à-vis du préentraînement ou de tout l’historique du dépôt.

Les durées GuitarSet varient de 9,4 à 24 s après le départ fixé à 5 s. Les 12 morceaux MAESTRO sont de nouvelles sources par rapport aux 35 fichiers des huit manifestes listés, issus du split test et de durées totales 29–300 s ; ce n’est pas l’intégralité du test officiel. Un seul extrait par compositeur ne caractérise pas toutes ses œuvres.

Les trois fichiers signalés par les errata publics GuitarSet sont absents de la sélection. Aucun correctif d’annotation ni exclusion après observation du score n’a été appliqué. Les notes hors MIDI40–88 sont exclues de la mesure et publiées séparément : 0 GuitarSet, 1 IDMT et 114 MAESTRO. Un doublon d’unisson exactement simultané a été retiré sur GuitarSet et un sur IDMT ; aucun sur MAESTRO.

Les archives téléchargées ont toutes été vérifiées contre les SHA256 fournis par GitHub. Le rejeu hors ligne reproduit exactement les rapports et toutes les décisions de S601 ; **693 vérifications d’appariement indépendantes** (77 fichiers × 3 politiques × 3 tolérances) concordent. Les 784 vraies notes correctes de BP perdues par S601 sont toutes expliquées et conservées.

Les sorties, vérités, annotations dérivées, audits et rapports sont conservés durablement dans [le dossier de preuves](evidence/v273-s606-s609/), avec un [index d’empreintes](evidence/v273-s606-s609/EVIDENCE_MANIFEST.json). Les grandes listes sont compressées en gzip sans modification du JSON original. Les sorties réseau brutes et probabilités candidates sont également disponibles dans les artefacts CI jusqu’à leur expiration ; aucun WAV n’est publié.

## Verdict de recherche

**S601 est meilleur que BP en F1 micro sur ces trois cohortes, mais il ne valide pas encore l’objectif de 80 % sur tous les instruments et conditions.** Il faut traiter les pertes sur accompagnement/Rock, les attaques rapides et le rappel piano, puis figer une nouvelle version avant une nouvelle validation indépendante. Réutiliser ces 77 extraits pour choisir la prochaine variante en fera des données de développement, pas un nouveau test indépendant. Les autres guitares physiques, les autres sous-ensembles IDMT, les prises utilisateur et les conditions Live restent à évaluer.

Aucune promotion de S80, aucun remplacement de la référence Exact-K et aucune modification du Live ne découlent de ces résultats.

## Annexe : les 60 prises GuitarSet, sans retrait des régressions

| Fichier | BP par défaut | S601 | Écart F1 en points |
|---|---:|---:|---:|
| 00_BN1-129-Eb_solo_mic.wav | 72,31 % | 83,48 % | +11.17 |
| 00_BN1-147-Gb_comp_mic.wav | 74,49 % | 81,08 % | +6.59 |
| 00_Funk1-97-C_comp_mic.wav | 73,58 % | 70,33 % | -3.25 |
| 00_Funk2-108-Eb_solo_mic.wav | 74,03 % | 82,42 % | +8.38 |
| 00_Jazz2-187-F#_comp_mic.wav | 67,55 % | 80,75 % | +13.20 |
| 00_Jazz3-150-C_solo_mic.wav | 62,20 % | 75,00 % | +12.80 |
| 00_Rock2-85-F_comp_mic.wav | 74,54 % | 61,93 % | -12.61 |
| 00_Rock3-117-Bb_solo_mic.wav | 74,73 % | 79,09 % | +4.36 |
| 00_SS2-88-F_solo_mic.wav | 90,32 % | 94,21 % | +3.89 |
| 00_SS3-98-C_comp_mic.wav | 69,10 % | 81,35 % | +12.25 |
| 01_BN1-147-Gb_comp_mic.wav | 75,98 % | 78,87 % | +2.89 |
| 01_BN3-154-E_solo_mic.wav | 87,31 % | 95,60 % | +8.29 |
| 01_Funk1-114-Ab_comp_mic.wav | 63,27 % | 79,38 % | +16.12 |
| 01_Funk3-112-C#_solo_mic.wav | 81,31 % | 86,06 % | +4.75 |
| 01_Jazz1-200-B_comp_mic.wav | 76,19 % | 86,67 % | +10.48 |
| 01_Jazz2-187-F#_solo_mic.wav | 82,87 % | 83,67 % | +0.80 |
| 01_Rock1-90-C#_solo_mic.wav | 91,71 % | 91,26 % | -0.45 |
| 01_Rock3-148-C_comp_mic.wav | 68,69 % | 66,22 % | -2.47 |
| 01_SS1-68-E_comp_mic.wav | 61,08 % | 76,38 % | +15.30 |
| 01_SS3-98-C_solo_mic.wav | 87,36 % | 89,45 % | +2.09 |
| 02_BN1-129-Eb_solo_mic.wav | 87,50 % | 92,47 % | +4.97 |
| 02_BN3-119-G_comp_mic.wav | 76,21 % | 72,68 % | -3.53 |
| 02_Funk1-114-Ab_comp_mic.wav | 73,91 % | 70,32 % | -3.59 |
| 02_Funk2-119-G_solo_mic.wav | 87,12 % | 91,12 % | +4.01 |
| 02_Jazz3-137-Eb_comp_mic.wav | 84,62 % | 75,62 % | -9.00 |
| 02_Jazz3-137-Eb_solo_mic.wav | 95,10 % | 93,88 % | -1.23 |
| 02_Rock3-117-Bb_comp_mic.wav | 77,84 % | 74,71 % | -3.12 |
| 02_Rock3-117-Bb_solo_mic.wav | 78,57 % | 88,89 % | +10.32 |
| 02_SS1-100-C#_solo_mic.wav | 89,11 % | 95,83 % | +6.72 |
| 02_SS1-68-E_comp_mic.wav | 46,61 % | 74,67 % | +28.06 |
| 03_BN1-129-Eb_solo_mic.wav | 90,91 % | 95,74 % | +4.84 |
| 03_BN3-119-G_comp_mic.wav | 75,84 % | 80,42 % | +4.58 |
| 03_Funk3-112-C#_solo_mic.wav | 88,37 % | 89,25 % | +0.88 |
| 03_Funk3-98-A_comp_mic.wav | 70,28 % | 76,21 % | +5.93 |
| 03_Jazz3-150-C_comp_mic.wav | 79,31 % | 91,46 % | +12.15 |
| 03_Jazz3-150-C_solo_mic.wav | 92,42 % | 91,97 % | -0.45 |
| 03_Rock1-130-A_comp_mic.wav | 70,29 % | 73,14 % | +2.85 |
| 03_Rock2-85-F_solo_mic.wav | 85,44 % | 92,63 % | +7.19 |
| 03_SS2-88-F_comp_mic.wav | 78,86 % | 81,67 % | +2.81 |
| 03_SS3-98-C_solo_mic.wav | 88,11 % | 87,58 % | -0.53 |
| 04_BN1-147-Gb_solo_mic.wav | 88,37 % | 93,18 % | +4.81 |
| 04_BN2-166-Ab_comp_mic.wav | 78,37 % | 84,40 % | +6.03 |
| 04_Funk1-114-Ab_solo_mic.wav | 73,47 % | 80,95 % | +7.48 |
| 04_Funk3-98-A_comp_mic.wav | 70,56 % | 75,78 % | +5.21 |
| 04_Jazz1-130-D_comp_mic.wav | 77,30 % | 80,29 % | +2.98 |
| 04_Jazz2-187-F#_solo_mic.wav | 91,11 % | 94,74 % | +3.63 |
| 04_Rock3-117-Bb_solo_mic.wav | 76,54 % | 77,78 % | +1.23 |
| 04_Rock3-148-C_comp_mic.wav | 67,47 % | 70,20 % | +2.73 |
| 04_SS2-107-Ab_solo_mic.wav | 94,55 % | 99,10 % | +4.55 |
| 04_SS3-98-C_comp_mic.wav | 61,84 % | 73,87 % | +12.03 |
| 05_BN1-129-Eb_comp_mic.wav | 81,57 % | 89,83 % | +8.26 |
| 05_BN3-119-G_solo_mic.wav | 84,62 % | 85,71 % | +1.10 |
| 05_Funk2-119-G_comp_mic.wav | 79,73 % | 82,03 % | +2.30 |
| 05_Funk2-119-G_solo_mic.wav | 85,00 % | 93,71 % | +8.71 |
| 05_Jazz3-137-Eb_solo_mic.wav | 90,91 % | 93,71 % | +2.80 |
| 05_Jazz3-150-C_comp_mic.wav | 67,71 % | 76,30 % | +8.59 |
| 05_Rock1-130-A_comp_mic.wav | 85,46 % | 77,75 % | -7.71 |
| 05_Rock3-117-Bb_solo_mic.wav | 93,46 % | 93,33 % | -0.12 |
| 05_SS1-68-E_comp_mic.wav | 63,24 % | 79,80 % | +16.56 |
| 05_SS3-84-Bb_solo_mic.wav | 91,06 % | 91,06 % | +0.00 |
