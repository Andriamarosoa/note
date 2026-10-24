---
serie: S842
statut: termine
type: experience
reference: S708
branche: codex/note-s842-real-guitarset-neural-transpose256
run_id: 40083706274
commit_sha: d99beb4186d9faf5eeff3a06ac4a3e5e92e8593d
promotion: false
---

# S842 — Vraie transposition neuronale latente et test E2 / E3 / E2+E3 (échec scientifique documenté)

## Exécution
[GitHub Actions #40083706274](https://github.com/Andriamarosoa/note/actions/runs/40083706274) **SUCCESS** : modèle entraîné à partir des **vrais 80 WAV GuitarSet signés**, cache GT vérifié par SHA, apprentissage PyTorch CPU de huit époques, test sur compositions OUTER disjointes, neuf tests S840+ cinq nouveaux tests S842 PASS. Le succès CI signifie que l'expérience a abouti techniquement : **elle ne signifie pas que la nouvelle méthode bat la référence**.

Archive GitHub Actions : `S842_REAL_GUITARSET_NEURAL_OCTAVE_PCM256`. Le rapport `/tmp/s842_model/resultat.json` indique la mesure hors entraînement du proxy multiclasse, et `/tmp/s842_evaluation/probe_report.json` indique TP/FP/FN E2/E3/conjonction avec seuils figés sur CAL.

**Origine vraie des données**, sans son artificiellement transposé : 8 945 couples de fenêtres **originales, distinctement enregistrées**, contenant une unique hauteur GT pendant l'intégralité d'un bloc brut de 256. Le transport neuronal est supervisé par des hauteurs réellement séparées de ±12 demi-tons et par l'embedding d'un enregistrement cible différent. Les paires sont **faiblement sémantiques** (performances différentes), et ne prouvent pas une véritable transposition audio physiquement fidèle du même timbre ou de la même attaque.

| Groupe isolé par composition | Paires audio originales | Usage |
|---|---:|---|
| BN1/Funk2/Jazz1 | **5 833** | entraînement |
| Rock3 | **1 836** | calibration / sélection de modèle |
| SS1 | **1 276** | évaluation externe au training de S842 |
| **Total** | **8 945** | 256 échantillons bruts par paire |

## Métriques
### Comparaison des reconnaissances ciblées E2, E3 et E2+E3 sur les 251 fenêtres GT SS1 OUTER

Chaque cellule affiche le F1 de classification **de note active par bloc** ; ce ne sont **pas** des métriques Onset F1 ±80ms.

| Cible | Encodeur sans transposition | IA transposition latente ±12 | Écart |
|---|---:|---:|---:|
| E2 (MIDI40) | **12,2222 %** (TP 11/FP 15/FN 143) | **0 %** (TP 0/FP 0/FN 154) | **−12,2222 points** |
| E3 (MIDI52) | **77,8589 %** (TP 160/FP 91/FN 0) | **66,8493 %** (TP 122/FP 83/FN 38) | **−11,0096 points** |
| E2+E3 actifs ensemble | **13,4831 %** (TP 6/FP 20/FN 57) | **0 %** (TP 0/FP 0/FN 63) | **−13,4831 points** |

Les vraies classes positives dans ce sous-échantillon sélectionné sont 154 pour E2, 160 pour E3 et 63 pour E2+E3. La sous-population de 251 fragments est constituée exprès pour le diagnostic, pas représentative du temps réel entier. Une même fenêtre peut appartenir aux trois catégories à la fois.

### Proxy sur les paires OUTER, toutes les 49 hauteurs

| Score d'activité des notes sur les 1 276 paires SS1 | F1 |
|---|---:|
| Encodeur sans transposition | **0,8186 %** |
| Avec transposition latente ±12 | **3,7951 %** |

Le gain du très faible proxy global 49 hauteurs **ne compense pas** le résultat ciblé E2/E3, qui est le but de cette série. Ne pas donner 3,7951 % comme une amélioration d'Onset ; la dynamique des NoteOn (attaques et réattaques) n'a pas été évaluée. Ce proxy de notes actives est expérimental, très faible et obtenu dans le même benchmark historiquement exploré.

### Témoin Onset protégé — valeurs *inchangées*

| Référence | Mesure |
|---|---:|
| **Onset F1 PCM256 officiel de travail** | **69,2584331 %** (7 607 TP, 3 093 FP, 3 660 FN) |
| HGB ajouté | 69,2489759 % : rejeté car +3 FP, aucun TP |
| Onset F1 de S842 | **NON MESURÉ** : pas de générateur NoteOn, pas de replay F1 nouvellement effectué |
| Exact-K | code et modèle inchangés |
| Active Exact-K | code et modèle inchangés |
| Offset | code et modèle inchangés |
| Frame Pitch F1 | code et modèle inchangés, proxy S842 isolé |
| Full Note F1 | code et modèle inchangés |

## Comparaison à S708
S708/H0 historique est à 73,252019 % F1 sur FIT80, mais son entraînement d'origine était à 1024 échantillons. Le modèle S842 est séparé : **entrée brute exactement 256 échantillons** à 44,1 kHz, aucun poids S708, pas de spectrogramme ni de futur. Il n'y a aucun gain Onset mesuré par rapport au contrôle indépendant 69,2584331 %. Ne pas confondre 5,805ms de cadence et coût CPU : le benchmark physique Windows/iRig n'a pas été réalisé.

## Audits
| Rôle | État | Preuve / résultat |
|---|---|---|
| Données | PASS | 80 WAV originaux et GT SHA vérifiés, paires issues de véritables enregistrements, TRAIN/CAL/OUTER disjoints par composition |
| Acoustique | PASS | exactement 256 nouveaux échantillons, transport latent neuronal appris, aucun pitch-shift DSP ni audio futur |
| Architecture | PASS | nouveau réseau isolé, même encodeur témoin, fusion de vues effectivement entraînée, cinq autres architectures et MIDI inchangés |
| Apprentissage | PASS | 5 833 paires TRAIN, 1 836 CAL, 1 276 OUTER, huit époques, seuils E2/E3 appris sur CAL uniquement |
| Métriques | FAIL | F1 E2/E3/superposition de la tête transposée inférieur au témoin sans transposition |
| LIVE | UNKNOWN | p99 CPU et vraie latence Windows/iRig non mesurés ; aucune intégration au MIDI |

## Verdict
**S842 ENTRAÎNEMENT RÉEL RÉUSSI, HYPOTHÈSE D'AMÉLIORATION E2/E3 RÉFUTÉE EN L'ÉTAT, PROMOTION REFUSÉE.** Le modèle transposé ne récupère **aucune** des 154 fenêtres positives E2 ni aucune des 63 positives E2+E3 dans le diagnostic SS1 alors que le témoin en reconnaît respectivement 11 et 6. Cette constatation ne prouve pas que toute IA de transposition est inutile.

Aucune cause précise n'est encore attribuable sans analyse supplémentaire des probabilités et poids de vues par individu. Les hypothèses ouvertes sont : (a) l'audio brut 256 (< une période complète d'E2 ~535 échantillons) peut être insuffisant sans état causal passé ; (b) les paires sémantiques de différents enregistrements ne constituent pas des transpositions physiques suffisamment comparables ; (c) un entraînement dominé par les hauteurs communes ne transfère pas aux cas graves superposés ; (d) fusion/calibration CAL → OUTER non robuste. Aucune de ces hypothèses n'est encore une preuve causale.

## Prochaine action
Ne pas relancer à l'identique les mêmes seuils. Auditer les E2/E3 positifs perdus et les poids des vues dans SS1 vs Rock3, puis essayer **un état récurrent strictement causal de blocs 256** (nouvel input 256 à chaque hop, mémoire des blocs précédents uniquement, décision immédiate sans attente) et/ou superviser directement E2/E3/superpositions avec véritables cas négatifs harmoniques. Tout nouveau candidat doit surpasser son propre contrôle sur des vraies notes séparées, puis proposer de vrais NoteOn sur l'ensemble des 80 WAV et des données jamais vues, sans toucher aux autres axes ni au MIDI avant la preuve.
