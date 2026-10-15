# S553–S560 — Rejet des portes naïves de médium ; second test inédit NoteOn à 60,02 %

**Daté du 15 octobre 2026. Synthèse SHA-vérifiée des vrais runs GitHub Actions. Aucune promotion S80, Live ou freeze_local_combo.**

## Hypothèse et première boucle S553 (développement, données exposées)

Le précédent S552 a révélé une régression spécifique dans le médium (MIDI60–71). Pour vérifier une idée simple sans toucher aux poids d'origine ni créer une règle « gagnante » non prouvée, nous avons combiné les **six politiques de notes déjà gelées S550** sur les mêmes cinq sources S545 et leurs MIDI désormais connus.

[S553 PASS #38663564188](https://github.com/Andriamarosoa/note/actions/runs/38663564188). Tous les TP/FP/FN par hauteur sont recalculés par appariement note exacte + temps ±80 ms sur **693 vraies notes**, aucun événement inconnu occulté. **Rétrospectif, ne pas confondre avec un holdout inédit :**

| Politique testée | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| **S546 source, 100ms + harmonic** | **352** | **293** | **341** | **52,62 %** |
| S528 médium et S546 graves/aigus | 342 | 273 | 351 | 52,29 % |
| Union des notes S528/S546 au médium | 353 | 326 | 340 | 51,46 % |
| Intersection des notes S528/S546 au médium | 335 | 246 | 358 | 52,59 % |
| S546 medium à 80ms, reste à 100ms | 352 | 309 | 341 | 51,99 % |
| Rétablissement des réattaques seulement si les deux experts RAW prédisent la note | 353 | 408 | 340 | 48,56 % |
| S528 médium + restauration des réattaques par consensus | 344 | 374 | 349 | 48,76 % |

**VERDICT S553 : six essais, aucun nouveau meilleur score.** Les faux positifs des réattaques réintroduites restent bien plus nombreux que les vrais cas sauvés (+115 FP pour +1 TP dans une variante), ce qui invalide la seule co-détection à deux modèles comme preuve d'une nouvelle attaque. Les fusions statiques des experts au médium ne suffisent pas. Rien de ces tests ne justifie une promotion.

## Validation S554–S559 : nouveau jeu jamais entraîné ni évalué par S546/S528

Pour ne pas optimiser éternellement sur S545, [S554 PASS #38663678664](https://github.com/Andriamarosoa/note/actions/runs/38663678664) a choisi **à partir du seul CSV officiel MAESTRO v3** cinq WAV du split officiel **TRAIN** — Beethoven, Orlando Gibbons, Schumann, Haydn, Jean-Philippe Rameau — tous disjoints des **13 précédents enregistrements S469/S537/S545 et de leurs crédits de compositeurs/co-compositeurs**, y compris noms « A / B ». Durée des morceaux dans metadata : 28–600s, suffisamment longs pour le même extrait 12 s qui commence à 15 s. Aucune note ni WAV ou MIDI utilisé pour les choisir. L'historique complet des autres branches non scanné : ne pas prétendre un test sans aucune exposition historique, seulement **disjonction stricte des sources connues d'entraînement S528/S546** et gel prospectif à l'intérieur de ce protocole.

**Ordre strictement contrôlé par workflows GitHub avec artefacts, SHA et vérifications à chaque phase :**
1. [S555 PASS #38663781717](https://github.com/Andriamarosoa/note/actions/runs/38663781717) : vrais WAV indépendants, ancienne pile originale V88 et poids freeze_local_combo sans modification. Pas d'exécution du S80 complet.
2. [S556 PASS #38663840345](https://github.com/Andriamarosoa/note/actions/runs/38663840345) : mêmes notes Spotify Basic Pitch 0.4.0 sur cinq WAV, seuils constants, aucune annotation MIDI.
3. [S557 PASS #38663870685](https://github.com/Andriamarosoa/note/actions/runs/38663870685) : 49 hauteurs MIDI40..88, vecteur original de 11 mesures physiques par note et événement V88 K0..K6, aucun MIDI.
4. [S558 PASS #38663923507](https://github.com/Andriamarosoa/note/actions/runs/38663923507) : 3 modèles S528 et 8 modèles S546 réels récupérés, poids contrôlés SHA, aucune réadaptation à ces cinq chansons. **Les six politiques de NoteOn et leurs SHA sont figées AVANT d'autoriser la lecture des MIDI S554.**
5. [S559 PASS #38663995955](https://github.com/Andriamarosoa/note/actions/runs/38663995955) : uniquement à ce stade, télécharger les vraies annotations et mesurer note exacte MIDI40–88, apparition ±30/50/80ms, appariement 1:1, 100 % des FP et FN comptés. **513 vraies notes** (les cinq extraits sont différents des anciens 693 de S551). Aucun apprentissage ou choix de politique après annotation.
6. [S560 PASS #38664511621](https://github.com/Andriamarosoa/note/actions/runs/38664511621) : réaudit indépendant, SHA MIDI inchangé, somme des compteurs TP/FP/FN sur registres égale strictement à S559 pour chaque politique et tolérance.

### Scores S559 exact pitch / NoteOn ±80ms

| Politique pré-gelée | TP | FP | FN | Précision | Rappel | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Basic Pitch births uniquement | 87 | 26 | 426 | 76,99 % | 16,96 % | 27,80 % |
| S528 old 3-forest RAW | 291 | 269 | 222 | 51,96 % | 56,73 % | 54,24 % |
| S528 + 80ms dedup | 290 | 199 | 223 | 59,30 % | 56,53 % | **57,88 %** |
| S546 8-forest RAW | 333 | 392 | 180 | 45,93 % | 64,91 % | 53,80 % |
| S546 + 80ms dedup | 330 | 285 | 183 | 53,66 % | 64,33 % | **58,51 %** |
| **S546 + 100ms dedup + harmonic veto .8** | **328** | **252** | **185** | **56,55 %** | **63,94 %** | **60,02 %** |

Scores tolérance ±30/50 ms du dernier modèle : **46,11 % / 55,99 %**, contre S528 dedup80 **45,11 % / 54,49 %**. Gain du meilleur dernier modèle contre le même nouveau corpus S528 dedup80 = **+2,13 pp** de F1 ; +38 TP, +53 FP et 38 FN récupérés. Le modèle S546 RAW **régresse** en F1 face à S528 RAW (−0,44 pp) malgré un meilleur rappel. Avec la même dédup80, S546 n'apporte que +0,63 pp ; il régresse **à Gibbons et Haydn** versus l'ancien S528. Ne pas confondre son résultat avec le +5,97 pp de S551 sur les précédents 693 événements. Les deux holdouts ne sont pas des partitions appariées et ne doivent pas être moyennés sans pondération et examen des sources.

### Résultat S560 par registre : défaut persistant et confirmé

Sur les **513 notes MIDI** de cette seconde cohorte, 149 sont graves (40..59), 237 médiums (60..71), 127 aiguës (72..88). Comparaison S546 dedup80 face à S528 dedup80 :

| Registre | Différence TP | Différence FP | F1 S546 − S528 |
|---|---:|---:|---:|
| Grave | +18 | +33 | **+4,36 pp** |
| Médium | +8 | +25 | **−0,97 pp** |
| Aigu | +14 | +28 | **+1,51 pp** |

Le filtre 100ms+harmony sur S546 dedup80 n'affecte pas le grave dans ce test, mais médium −1 TP/−7 FP (F1 +0,57 pp) et aigu −1 TP/−26 FP (F1 +5,15 pp). Encore des véritables notes supprimées ; le régime dit « sans régression » n'est PAS démontré.

Le *même problème* médium est confirmé indépendamment par S552 sur la première cohorte S545 (−0,42pp sous S546 dedup80) et par S560 sur la seconde (−0,97pp). La priorité est donc d'apprendre à reconnaître, depuis l'acoustique elle-même, si une note MIDI60..71 est une fondamentale / véritable frappe ou un harmonique/résonance/double événement ; il faut plus qu'un vote statique entre modèles. La restauration d'une réattaque exige preuve de transitoire, dynamique de l'enveloppe, silence/relâchement de l'énergie et rôle harmonique, sur nouveaux exemples source-held ; ne pas décider sur les mêmes 5 MIDI de validation ouverte.

## Référence, limites et promotion

- **Meilleur candidat observé sur la cohorte entièrement nouvelle S554 : S546 + dedup100/harmonic, F1 NoteOn 60,02 %.** Aucune preuve de F1 80 %. Le gain face à l'ancien modèle est réel sur ce corpus, mais ne se traduit pas partout et ne démontre pas une architecture finale robuste.
- Le modèle officiel **S80 Exact-K, poids freeze_local_combo, GuitarSet, IDMT et Latence Live ne sont pas remplacés/rejoués intégralement par les tâches ici**. Pas de comparaison équivalente entre Exact-K 43,87 % et F1 NoteOn 60,02 %.
- La PR reste **DRAFT, pas de merge, pas de promotion**. Tous les scripts S553–S560 sont conservés pour audit.
