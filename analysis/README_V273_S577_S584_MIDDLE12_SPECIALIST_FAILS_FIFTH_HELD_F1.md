# S577–S584 — Tête dédiée au médium : validation croisée puis rejet sur nouveau corpus NoteOn

**15 octobre 2026 — SEPT workflows GitHub Actions vérifiés. PR #41 en brouillon. Référence S80, GuitarSet Exact-K, freeze_local_combo et Live : INCHANGÉS.**

## Pourquoi S578

La tête temporelle S569 avait amélioré le NoteOn sur la cohorte S570 à **59,90 %**, mais l'audit S576 a montré une **régression de −1,25 point de F1 au médium MIDI60..71** par rapport à S562 physical34, même si le score global progressait grâce au grave. L'objectif est de séparer les indices d'existence d'une hauteur fondamentale des indices d'une nouvelle attaque / ré-frappe véritable. Le passage S578 teste une tête réellement entraînée sur les notes MIDI **60 à 71 uniquement**, sur le son réel, sans remplacer les probabilités graves ou aiguës.

## S577 : cinquième cohorte métadonnées, source tenue à l'écart

[S577 enregistré et vérifié, run SUCCESS #38675435536](https://github.com/Andriamarosoa/note/actions/runs/38675435536). Cinq nouveaux morceaux de l'original **MAESTRO TRAIN** — **Felix Mendelssohn, César Franck, Johann Strauss / Alfred Grünfeld, Carl Maria von Weber, Johann Pachelbel**. Le protocole exclut les 28 enregistrements précédents S469/S537/S545/S554/S561/S570 **et tous leurs compositeurs/co-compositeurs composés**. Le CSV officiel est contrôlé SHA. Les cinq fichiers MIDI ont été exclus de tout entraînement S569/S578.

**Nuance de sélection et données :** la première tentative S577 avec une fenêtre de durée étroite ne trouvait pas cinq auteurs ; une fenêtre à 3 600 secondes a trouvé des sources, mais un WAV était trop volumineux pour l'ancienne limite de téléchargement. Le filtre de durée sur le CSV a été ramené à ≤1 000 s, puis cette liste finale a été de nouveau gelée **avant** toute prédiction/annotation MIDI de cette cohorte. Cette adaptation technique après constat du poids d'un fichier WAV ne constitue pas une optimisation sur les annotations ou les résultats de détection ; elle signifie néanmoins qu'il ne faut pas présenter ce processus comme si aucun attribut audio n'avait jamais été observé avant la réinscription. Aucun MIDI S577 n'a été lu avant la prédiction figée. Le split source est **MAESTRO TRAIN**, pas le split officiel TEST ; toutes les branches historiques extérieures au protocole n'ont pas été auditées.

## S578 : deux véritables réseaux de hauteur entraînés sur 28 vieux audios et vrais MIDI

[S578 entraînement + cinq plis bloqués par compositeur SUCCESS #38675146653](https://github.com/Andriamarosoa/note/actions/runs/38675146653). Les 28 anciennes musiques connues ont été réellement auditées et munies de vrais labels MIDI à l'attaque (pour tous les pitchs, même en K0) ; exclusions de groupes co-compositeurs partagés entre contrôle et apprentissage. Tous les vecteurs sont acoustiques temporels **52D**, sans vérité MIDI en entrée. Deux véritables ExtraTrees entraînés et sauvegardés par SHA, aucun modèle ajusté sur S577 :
- **Global28**, toutes les notes 40..88, 52D.
- **Specialist-mid12**, sous-réseau MIDI60..71, 52D.
- Trois routages prédéfinis : Global28 seul ; remplacer seulement les 12 notes médiums ; mélanger 50/50 seulement les 12 notes médiums. Tous laissent graves/aigus et notes BP-born inchangés, même seuil p≥0,70, même dédup100/harmonic0.8.

| S578 validation croisée **rétrospective sur les 28 sources déjà connues**, exact MIDI pitch et attaque ±80ms | TP | FP | FN | F1 NoteOn |
|---|---:|---:|---:|---:|
| Global28 seul | 1676 | 713 | 1309 | **62,37 %** |
| Spécialiste seul au médium | 1618 | 603 | 1367 | 62,16 % |
| **Mélange 50/50 global + spécialiste médium** | 1652 | 626 | 1333 | **62,78 %** |

Le mélange ajoute seulement **+0,40 pp de F1** face au modèle général sur les mêmes 28 chansons, −24 TP et −87 FP. **Ce score est interne, pas un résultat sur nouveaux morceaux.**

## S579–S583 : cinquième vrai test de musique non entraînée

Tout le jeu S577 est traité dans un ordre strict, avec preuve d'immuabilité SHA des fichiers de prédiction antérieure à l'ouverture MIDI :

1. [S579 ORIGINAL V88 WAV SUCCESS #38675784995](https://github.com/Andriamarosoa/note/actions/runs/38675784995) : même pile V88 de freeze_local_combo et mêmes prédictions K0..K6 de source audio, pas le S80 complet ; SHA des véritables WAV, aucun MIDI. Un premier téléchargement WAV trop volumineux / un contrôle de rapport ancien a été corrigé sans résultat MIDI.
2. [S580 original Basic Pitch SUCCESS #38675865231](https://github.com/Andriamarosoa/note/actions/runs/38675865231) : notes BP prédéfinies, sans MIDI.
3. [S581 original 49×11 données WAV SUCCESS #38676271087](https://github.com/Andriamarosoa/note/actions/runs/38676271087) : tous les événements de note acoustiques 40..88 y compris K0. La restriction historique 140 MB de S511 a été remplacée **par la lecture déjà vérifiée 250 MB de S579**, sans changement de classifieur ni d'annotation.
4. [S582 gel des quatre politiques NoteOn SHA SUCCESS #38675724837](https://github.com/Andriamarosoa/note/actions/runs/38675724837) : poids du S569 full23 et les 2 modèles S578 full28 contrôlés par SHA original ; quatre propositions MIDI40..88 nommées et horodatées ont été générées sur les 5 véritables WAVs de S577, puis figées **AVANT** consultation des étiquettes.
5. [S583 vrai MIDI NoteOn SUCCESS #38675826715](https://github.com/Andriamarosoa/note/actions/runs/38675826715) : seulement après gel SHA des quatre propositions, les cinq MIDI source ont été ouverts et toutes les vraies/fausses notes exactes MIDI40..88 et les débuts d'attaque ±30/50/80 ms ont été comptés, avec appariement 1:1. Total **272 vraies notes** : Pachelbel 37, Weber 43, Franck 53, Strauss/Grünfeld 97, Mendelssohn 42. Il existe **1 authentique re-frappe de même note à moins de 100ms** chez Strauss/Grünfeld.

### S583 — résultat IMPORTANT : la petite amélioration du spécialiste ne se généralise pas

| Modèle, politique rigoureusement identique dedup100 + harmonic0.8 | TP | FP | FN | Précision | Rappel | F1 NoteOn ±80ms |
|---|---:|---:|---:|---:|---:|---:|
| S569 old23 control | 161 | 116 | 111 | 58,12 % | 59,19 % | 58,65 % |
| **S578 Global28** | **162** | **112** | **110** | **59,12 %** | **59,56 %** | **59,34 %** |
| S578 remplacer seulement médium | 153 | 94 | 119 | 61,94 % | 56,25 % | 58,96 % |
| S578 mélanger 50/50 seulement médium | 155 | 97 | 117 | 61,51 % | 56,99 % | 59,16 % |

**Seule amélioration externe validée dans cette boucle : S578 Global28 +0,69 pp versus S569 old23 sur ces mêmes cinq pistes** (TP+1/FP−4/FN−1). Le spécialiste lui-même réduit beaucoup de faux positifs, mais abaisse encore davantage le rappel des notes véritables et son F1 face à Global28 : **remplacement −0,38 pp, mélange −0,18 pp**. Aucun ajustement n'a été fait après visualisation de S583.

Sur le médium MIDI60..71 uniquement :
- Global28 TP71, FP41, FN50 ; F1 **60,94 %**.
- Spécialiste remplaçant TP62, FP23, FN59 ; F1 **60,19 %**.
- Mélange 50/50 TP64, FP26, FN57 ; F1 **60,66 %**.

Il n'est pas scientifique de « promouvoir » ce mélange sous prétexte qu'il a obtenu 62,78 % sur le corpus de développement. Une amélioration interne peut cacher une régression sur morceaux inconnus : **rejeté**.

## S584 — preuve indépendant post-hoc MIDI par hauteur musicale

[S584 SHA recheck PASS #38676593804](https://github.com/Andriamarosoa/note/actions/runs/38676593804) : cinq vrais MIDI originaux re-téléchargés et vérifiés SHA, les quatre prédictions NoteOn produites avant vérité MIDI rejouées ; tous les TP/FP/FN des trois registres et de chaque hauteur MIDI60..71 reproduisent exactement S583.

Le mélange comparé à Global28 au médium retire **15 faux positifs, mais perd 7 vraies notes**. Détail des notes perdues :

| Note MIDI (nom occidental) | Variation TP avec mélange | Variation FP |
|---|---:|---:|
| 60 — Do4 | −1 | −5 |
| 63 — Mi♭4 | −1 | 0 |
| 65 — Fa4 | −2 | −3 |
| 67 — Sol4 | −1 | 0 |
| 70 — Si♭4 | −1 | 0 |
| 71 — Si4 | −1 | 0 |

Les autres douze pitchs ne perdent aucune vraie note ; le mélange retire notamment des FP sur 62/Ré4, 64/Mi4, 69/La4. L'accord des têtes réduit les notes fantômes, mais le nouveau contrôleur reste **aveugle à certaines véritables attaques** et perd des notes monophoniques/accords vrais malgré un gain de précision. S584 est un audit après étiquettes exposées, pas une deuxième validation prospective.

## Verdict et suites de développement

- **Architecture expérimentale principale conservée : S569 temporelle 52D** (nouvelle itération S578 Global28 gagne modestement sur la cinquième source, mais cela ne justifie PAS son remplacement comme référence officielle). Le spécialiste médium S578 est **rejeté au regard de son propre nouveau test**.
- Pour l'architecture future, une simple tête de présence de hauteur ne résout pas un **nouveau NoteOn**. Il faut une décision d'événement physique *réattaque vs sustain/résonance*, des features causales d'atténuation/rupture/transitoire temporel sur la même hauteur, et si possible d'autres instrumentations que le seul piano MAESTRO. Un total de seulement 10 re-frappes <100ms dans S569 et 1 dans S583 ne justifie pas encore une règle générale.
- Ne pas profiter de la validation S583 pour choisir après coup les seuils ou les combinaisons et appeler cela un score indépendant. Prochaine évaluation sur **une autre cohorte jamais annotée**, figer AVANT MIDI, rapporter le total NoteOn ±30/50/80ms, le F1 MIDI60..71 ET les gains/régressions des notes individuelles.
- **Ne pas promouvoir/merger dans la référence officielle S80.** Les anciennes valeurs d'Exact-K GuitarSet/IDMT, la mécanique freeze_local_combo et le pipeline Live n'ont pas été modifiés ni intégralement retestés ici. L'objectif **80 % de F1 NoteOn** n'est pas atteint.
