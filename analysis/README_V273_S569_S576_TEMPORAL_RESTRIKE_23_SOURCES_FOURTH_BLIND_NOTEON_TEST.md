# S569–S576 — Tête temporelle réattaque/résonance 52D, 23 musiques, quatrième validation MIDI indépendante

**15 octobre 2026. Branche expérimentale de [PR #41](https://github.com/Andriamarosoa/note/pull/41), PR DRAFT, aucun merge/promotion S80.**

## Origine du problème et hypothèse testée

L'architecture S562 à 34 indices spectraux sait retirer une partie des faux harmoniques du médium, mais S568 a prouvé qu'elle a supprimé **8 vraies notes médiums** dans la troisième cohorte malgré une baisse de 30 FP. Une absence de prédiction n'est pas une preuve de doublon : les auteurs ont observé des répétitions authentiques d'une même note à des intervalles inférieurs à 100 ms. Le correcteur temporel doit donc être **appris**, en faisant intervenir l'évolution physique acoustique de la hauteur avant et après un événement V88, plutôt qu'en comparant des votes d'experts statiques (S553 rejeté).

## S569 — deux vrais nouveaux modèles de hauteur avec contexte temporel

[S569 TRAINING SUCCESS #38670001388](https://github.com/Andriamarosoa/note/actions/runs/38670001388). Le train contient **23 enregistrements MAESTRO réellement annotés antérieurement** : 3 S469 +5 S537 +5 S545 +5 S554 +5 S561. Aucune annotation, audio ou choix de seuil du prochain ensemble S570 n'a été consulté pour entraîner.

- **52 caractéristiques par note MIDI40..88 à chaque candidat V88 K0..K6** : 34 caractéristiques physiques S562 (niveau, attaque, nouveauté, maintien, concurrentes harmoniques et hauteur relative) plus 18 mesures de **dynamique temporelle réelle**, calculées à partir des lignes MIDI-blind originales 49×11 : énergie/attaque/maintien/nouveauté à l'événement sonore précédent pour la *même hauteur*, différences actuel−précédent, deux indices à l'avant-dernier événement, décroissances exponentielles du décalage temporel 80/250 ms, rapports de force/nouveauté, et présence/activité/confidence Basic Pitch déjà calculées sans MIDI. Les tableaux sont classés chronologiquement **indépendamment pour chaque source**, y compris en cas d'ordre d'entrée modifié ; aucune caractéristique MIDI vérité n'entre à l'inférence.
- **Tête A** : apprentissage de la véritable identité de note sur les candidats V88 dont l'attaque a été appariée à un vrai accord MIDI à ±80ms, les autres hauteurs MIDI40..88 à cet événement étant des classes négatives. Conserve toujours les BP births du modèle préentraîné.
- **Tête B** : mêmes événements réels appariés + tous les autres événements V88 non appariés, avec poids de leurs étiquettes négatives diminué à 0,3 pour éviter de prétendre à tort qu'une résonance est silence.
- Pour chaque tête, les **23 plis de validation croisée excluent intégralement l'enregistrement et toute autre œuvre partageant un compositeur ou co-compositeur**, et chaque vrai événement MIDI et chaque fausse note prédite est compté. Deux modèles finaux supplémentaires sont entraînés sur la totalité des 23 enregistrements avec SHA de poids.
- **10 véritables re-frappes MIDI de même hauteur à moins de 100ms** constatées sur les anciennes annotations. La déduplication absolue à 100ms ne peut donc pas être considérée comme sûre universellement. La nouvelle tête utilise un **contexte acoustique** mais les politiques d'émission en aval appliquent encore le filtre historique 80/100ms ; ce filtre reste une source de perte de vrais NoteOn.

| S569, validation croisée **interne**, ±80ms | TP | FP | FN | F1 NoteOn |
|---|---:|---:|---:|---:|
| **Tête A : attaques appariées** | **1507** | **626** | **1179** | **62,54 %** |
| Tête B : attaques et événements acoustiques non appariés | 1414 | 638 | 1272 | 59,69 % |

**Important** : ces scores sont des contrôles internes rétrospectifs du même corpus d'apprentissage, non une généralisation aveugle ; ils ne sont pas comparables directement au 58,87 % S562 réalisé sur **18** chansons seulement.

## S570–S575 — quatrième vraie validation SHA-frozen, nouveaux compositeurs

- [S570 PASS #38670092733](https://github.com/Andriamarosoa/note/actions/runs/38670092733) : sélection déterministe sur le seul CSV officiel MAESTRO v3 de cinq morceaux du split officiel **TRAIN**, avec exclusion de **tous les 23 chemins d'audio précédemment exposés et compositeurs/co-compositeurs associés**. Nouvelle musique : *George Frideric Handel, Leoš Janáček, Muzio Clementi, Alban Berg, Percy Grainger*. Aucun WAV ni MIDI S570 ouvert ou inspecté à cette étape. Cette nouvelle cohorte est en dehors de tout entraînement S546/S562/S569. **Ce n'est pas un résultat sur le split officiel MAESTRO TEST** ni la preuve d'absence absolue d'exposition dans l'histoire de chaque branche du dépôt.
- [S571 PASS #38670185242](https://github.com/Andriamarosoa/note/actions/runs/38670185242) : ancienne pile native V88 et candidats de notes K0..K6 à partir des 5 véritables WAV ; aucune vérité MIDI et poids historiques freeze_local_combo inchangés.
- [S572 PASS #38670258232](https://github.com/Andriamarosoa/note/actions/runs/38670258232) : notes Basic Pitch préentraîné, aucune vérité MIDI.
- [S573 PASS #38670454679](https://github.com/Andriamarosoa/note/actions/runs/38670454679) : original 49×11 WAV physiques pour toutes les hauteurs 40..88 à chaque candidat V88, y compris K0. Les premiers lancements S572/S573 bloqués par l'ordre de résolution des dépendances ont été corrigés, sans inspection des MIDI ni adaptation de note.
- [S574 PASS #38670528441](https://github.com/Andriamarosoa/note/actions/runs/38670528441) : vérification de **11 vrais poids entraînés** S546×8, S562×1 et S569×2, par SHA original. Sept politiques définies **avant** annotation : BP births seules, ancienne S546 à 100ms+harmony, S562 physique34 à 100ms+harmony, S569 tête A et tête B chacune à 80ms sans veto et 100ms+harmony 0,8. Seuil d'entrée NoteOn MIDI à p≥0,70 fixe. Les 7 politiques ont produit leurs prédictions **sur les véritables cinq WAVs S570** sans jamais lire les MIDI. Les événements de notes et leurs SHA ont été gelés dans un artefact immuable **avant** lecture d'annotation.
- [S575 PASS #38671119840](https://github.com/Andriamarosoa/note/actions/runs/38671119840) : les cinq fichiers MIDI ont été ouverts seulement APRÈS le contrôle des empreintes S574. Appariement note exacte MIDI40..88 et début ±30/50/80ms, 1:1, toutes les notes supplémentaires FP, toutes les vraies notes manquées FN. **299 vraies notes nommées** sur les cinq nouvelles œuvres.

### Scores complets S575, nouveau vrai test ±80 ms

| Modèle et politique figés | TP | FP | FN | Précision | Rappel | F1 NoteOn |
|---|---:|---:|---:|---:|---:|---:|
| Basic Pitch naissances seules | 52 | 26 | 247 | 66,67 % | 17,39 % | 27,59 % |
| S546 original (8 musiques) + 100ms+harmony | 158 | 150 | 141 | 51,30 % | 52,84 % | **52,06 %** |
| S562 physique34 (18 musiques) + 100ms+harmony | 164 | 115 | 135 | 58,78 % | 54,85 % | **56,75 %** |
| S569 **tête A temporelle52** + 80ms | 179 | 130 | 120 | 57,93 % | 59,87 % | 58,88 % |
| **S569 tête A temporelle52 + 100ms+harmony** | **177** | **115** | **122** | **60,62 %** | **59,20 %** | **59,90 %** |
| S569 tête B (événements non appariés) + 80ms | 166 | 110 | 133 | 60,14 % | 55,52 % | 57,74 % |
| S569 tête B + 100ms+harmony | 163 | 96 | 136 | 62,93 % | 54,52 % | 58,42 % |

**Vrai gain apparié** : meilleure tête A S569 à 100ms/harmony vs ancien S562 physique34 avec exactement le même filtrage et les mêmes cinq nouvelles sources = **+3,15 points de F1**, **+13 TP**, **0 FP supplémentaires**, **−13 FN**. Comparée à la base S546 ancienne sur ces mêmes musiques, +7,84 points de F1 mais modèles avec nombres différents de chansons d'entraînement. La tête B retire 19 FP par rapport à S562 mais perd 1 TP, score +1,68 point de F1 ; plus conservative et moins de rappel.

F1 par compositeur, S569 tête A vs S562 34D, 100ms : **Handel +7,74 pp, Janáček +5,72 pp, Berg +2,60 pp, Clementi +2,27 pp, Grainger −0,10 pp**. L'objectif de **non-régression absolue** n'est pas atteint, même sur cette cohorte.

Le meilleur score **observé sur une AUTRE cohorte** avec une autre architecture était **60,02 % sur S554** ; ce score n'est PAS la comparaison appariée de S569, puisqu'il n'y était pas déployé en mode prospectif. Le score S569 à 59,90 % n'est donc pas une baisse de performance face au « 60,02 % » d'une autre collection.

## S576 — audit indépendant post-MIDI, par registre

[S576 PASS #38671279513](https://github.com/Andriamarosoa/note/actions/runs/38671279513) a indépendamment rejoué les sept résultats S575, vérifié chaque SHA MIDI et exact chaque TP/FP/FN total par trois registres et ±30/50/80ms. 299 notes MIDI réelles : grave 101 (MIDI40–59), médium 124 (MIDI60–71), aigu 74 (MIDI72–88).

| Registre | S569 tête A temporelle52 vs S562 physique34, tous deux 100ms/harmony | TP | FP | F1 delta |
|---|---|---:|---:|---:|
| Grave | Augmentation significative du rappel et également des FP | **+20** | +15 | **+9,12 pp** |
| **Médium** | Filtrage reste trop important : perte de 4 vraies notes | **−4** | −4 | **−1,25 pp** |
| Aigu | Supprime plusieurs faux pics mais perd aussi des TP | **−3** | **−11** | **+1,26 pp** |

Autrement dit, le gain global de S569 résulte **surtout du grave (+20 TP)**. **La faiblesse du médium n'est pas guérie**, elle reste régressive malgré l'ajout des 18 caractéristiques temporelles. Le compromis de la tête B entraîne une régression supplémentaire de médium −5 TP pour −1 FP face à la tête A.

## Décision scientifique et suite

**Retenir S569 tête A à 52D comme meilleur candidat de développement récent, pas comme référence officielle ni système « 80 % ».** L'architecture porte un gain réellement mesuré sur cinq nouvelles œuvres, mais un léger cas régressif Grainger et des erreurs persistantes sur le médium.

Pour aller plus loin, séparer réellement la décision temporelle **nouveau NOTE_ON de la même hauteur** du classifieur d'identité de hauteur : tête explicite d'événement de ré-frappe, labels vrais MIDI rapides + contrôles harmonique/offset/silence par tranche musicale ; apprentissage sur plusieurs instruments et sources dont les attaques ne sont pas issues du seul MIDI du piano MAESTRO. Tester les mêmes événements re-frappes, F1 plein flux et exact-K ; le budget de 10 vraies re-frappes rapides dans 23 anciens WAV est insuffisant pour prouver la robustesse d'une nouvelle tête spécialisée. Idéalement préenregistrer d'autres instruments (guitare IDMT/GuitarSet) pour une nouvelle validation et les comparer sur un protocole NoteOn strictement identique.

**La référence officielle S80 / Exact-K, freeze_local_combo, original GuitarSet et pipeline Live n'ont pas été changés, fusionnés ni réévalués intégralement ici.** Rien de ces tests n'établit 80% NoteOn F1 ni non-régression S80. PR #41 reste **DRAFT**.
