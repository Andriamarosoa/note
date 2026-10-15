# S545–S552 : nouvelle validation MIDI NoteOn sur cinq compositeurs exclus de l'entraînement

**15 octobre 2026 — GitHub Actions SUCCESS : S545, S547, S548, S549, S550, S551, S552. Aucune référence officielle modifiée.**

## Protocole prospectif, contamination à éviter

**S545** ([run #38658066587](https://github.com/Andriamarosoa/note/actions/runs/38658066587)) a choisi d'après le **seul CSV MAESTRO v3** cinq morceaux du split original **MAESTRO TRAIN**, en excluant les huit anciens enregistrements S469/S537 et tous leurs compositeurs ou co-compositeurs (par exemple A/B). Les cinq nouveaux sujets sont également disjoints les uns des autres sur les noms de compositeurs et co-compositeurs. **Ce n'est pas le benchmark officiel MAESTRO TEST.** La recherche intégrale d'éventuels antécédents de ces fichiers dans toutes les branches et archives du projet n'a pas été réalisée ; n'affirmer aucune absence absolue de contact avec tout historique externe. En revanche, les cinq sources ne figurent **dans aucun entraînement S528 ni S546**, et leurs annotations n'ont pas été ouvertes avant gel.

Les cinq sources choisies sans audio ni étiquettes :
1. Wolfgang Amadeus Mozart ;
2. Edvard Grieg ;
3. Pyotr Ilyich Tchaikovsky / Mikhail Pletnev ;
4. Antonio Soler ;
5. Georges Bizet / Vladimir Horowitz.

Les opérations ont eu lieu dans cet ordre d'audit :

- **S547** [SUCCESS #38659307689](https://github.com/Andriamarosoa/note/actions/runs/38659307689) : SHA audio de chaque source et pile V88 originale exécutée sur 5×12 secondes d'authentique WAV piano. Poids source V88 / freeze_local_combo inchangés, MIDI fermé, **ne pas confondre V88 avec S80 complet**.
- **S548** [SUCCESS #38659342120](https://github.com/Andriamarosoa/note/actions/runs/38659342120) : Spotify Basic Pitch 0.4.0 avec seuils identiques au S539, cinq enregistrements WAV SHA-vérifiés, aucune annotation MIDI.
- **S549** [SUCCESS #38659675965](https://github.com/Andriamarosoa/note/actions/runs/38659675965) : observations originales MIDI40..88, par hauteur et événement V88, **49×11 indices acoustiques**, K0 compris. Les anciens essais du workflow échoués à cause de l'ordre des dépendances/artéfacts et d'un identifiant de statut Basic Pitch S539->S548 incorrect ont été corrigés, sans inspection des MIDI ni changement de méthode de détection.
- **S550** [SUCCESS #38659705635](https://github.com/Andriamarosoa/note/actions/runs/38659705635) : **11 véritables modèles de hauteur déjà entraînés** (3 S528, 8 S546) récupérés par SHA de poids. Tous les nouveaux WAV du jeu S545 sont exclus de leur entraînement. Les modèles des sources d'origine ont été moyennés à l'inférence pour obtenir les probabilités de hauteur des nouvelles chansons. Seuil fixe p≥0,70, six politiques préétablies, notes Basic Pitch déjà nées jamais effacées, déduplication 80/100 ms, veto harmonique fort .8 uniquement pour la dernière politique. La totalité des prédictions de notes **a été figée dans un JSONL avec SHA** AVANT consultation du moindre MIDI S545.
- **S551** [SUCCESS #38659753045](https://github.com/Andriamarosoa/note/actions/runs/38659753045) : les cinq MIDI originaux MAESTRO ont été récupérés **uniquement après vérification de tous les SHA de S550**. Les vraies notes ont été comparées à tous les faux positifs, sur nom exact de note/pitch et temps d'attaque ±30/50/80ms, one-to-one. 431 vrais groupes d'attaque contenant **693 notes MIDI 40..88**.
- **S552** [SUCCESS #38659989151](https://github.com/Andriamarosoa/note/actions/runs/38659989151) : audit indépendant post-frozen en trois registres, réouverture du même MIDI SHA et reproduction exacte de tous les 6×3 scores S551. C'est une analyse **post-hoc**, **pas** une nouvelle validation aveugle.

## S551 — scores vérifiés sur les cinq nouveaux WAV/MIDI

Le F1 est celui de la **détection de vraies notes nommées et de leur attaque ±80 ms**, pas du simple nombre Exact-K, ni un score conditionnel à une attaque annotée connue à l'avance. Tous les faux déclenchements émis par le système comptent FP ; toutes les vraies notes manquées comptent FN, même lorsqu'il n'y a pas de candidat V88.

| Système / politique gelée | TP | FP | FN | Précision | Rappel | F1 NoteOn ±80ms |
|---|---:|---:|---:|---:|---:|---:|
| Spotify Basic Pitch naissance seule | 125 | 28 | 568 | 81,70 % | 18,04 % | **29,55 %** |
| S528 3 têtes moyennées, sans dedup | 285 | 332 | 408 | 46,19 % | 41,13 % | **43,51 %** |
| S528 + dedup80 | 282 | 234 | 411 | 54,65 % | 40,69 % | **46,65 %** |
| S546 8 têtes strict-compositeur moyennées, sans dedup | 356 | 472 | 337 | 43,00 % | 51,37 % | **46,81 %** |
| S546 + dedup80 | 353 | 317 | 340 | 52,69 % | 50,94 % | **51,80 %** |
| **S546 + dedup100 + veto harmonique .8** | **352** | **293** | **341** | **54,57 %** | **50,79 %** | **52,62 %** |

Comparaisons **appariées sur les mêmes cinq nouveaux morceaux** :
- Meilleur S546 (52,62) vs S528 avec dedup80 (46,65) = **+5,97 points de F1**, +70 TP, +59 FP, −70 FN, et ~+10,10 points de rappel.
- Contrôle architectural strict S546 dedup80 (51,80) vs S528 dedup80 (46,65) = **+5,15 points**, +71 TP, +83 FP, −71 FN. Gain de F1 positif dans chacun des cinq morceaux : Soler +4,05 ; Grieg +8,00 ; Bizet/Horowitz +9,76 ; Tchaikovsky/Pletnev +2,63 ; Mozart +3,11 points. Le meilleur ensemble *historique de développement* S546 sur 8 WAV a donné 54,38 %, mais cette valeur n'est pas comparable directement à 52,62 % sur les nouveaux morceaux.
- Dedup100+harmony contre S546 dedup80 : +0,82 point F1 global mais −1 TP, −24 FP, +1 FN. Ce veto comporte donc encore un risque de supprimer de vraies notes.

**Conclusion vérifiée : un meilleur modèle entraîné sur huit sources avec exclusion réelle des co-compositeurs apporte une hausse de rappel effective sur de nouveaux morceaux. La stratégie de déduplication compense une partie de sa sur-prédiction.** L'objectif 80 % n'est pas atteint.

## S552 — localisation détaillée des corrections et des nouvelles erreurs

Les 693 notes originales se répartissent en grave **328**, médium **151**, aigu **214**.

| Registre | Gain S546 dedup80 vs S528 dedup80 | Delta TP | Delta FP | F1 delta |
|---|---|---:|---:|---:|
| MIDI 40–59 (grave) | Grande hausse de rappel, coût FP modéré | +45 | +34 | **+10,70 points** |
| MIDI 60–71 (médium) | Plus de vraies notes, trop d'ajouts FP | +10 | +36 | **−0,42 point** |
| MIDI 72–88 (aigu) | Rappel en hausse | +16 | +13 | **+4,19 points** |

Le passage S546 dedup80 → dedup100+harmonie entraîne dans le **grave** −1 TP/−3 FP (F1 quasi identique, −0,006 pp), dans le **médium** 0 TP/−16 FP (+1,78 pp), et dans l'**aigu** 0 TP/−5 FP (+0,85 pp). Il existe au moins une vraie note identique rejouée à moins de 80 ms dans l'extrait Soler ; la déduplication doit être subordonnée à une véritable reconnaissance acoustique de réattaque plutôt qu'à une règle de temps absolue.

## Décision à partir de la nouvelle validation

- **Candidat architectural recommandé pour prochaine validation** : S546, modèle de 49 hauteurs *positive7* avec entraînement source/compositeur disjoint, émission indépendante K0–K6, protection des BP births et déduplication consciente des véritables répétitions. Les anciens poids S80/Exact-K et Live restent intégralement inchangés.
- **Difficulté à résoudre prioritairement** : trop de faux positifs MIDI 60–71 (médium) provenant des sélections de hauteur. Il faut entraîner et valider une tête de discrimination de fondamentales / sous-harmoniques par note et une tête de réattaque, pas régler de nouveaux seuils sur les cinq MIDI S545 déjà ouverts.
- **Rigueur scientifique** : toute nouvelle combinaison de têtes issue de ces 5 musiques constitue désormais un **développement** sur corpus exposé. La prochaine validation indépendante exige de préenregistrer encore d'autres interprètes, morceaux et idéalement instruments, de geler les décisions avant toute annotation, puis de répéter le F1 NoteOn et les tests anti-régression d'Exact-K/S80/GuitarSet/Live. Aucun de ces composants de production n'est modifié dans S545–S552.
