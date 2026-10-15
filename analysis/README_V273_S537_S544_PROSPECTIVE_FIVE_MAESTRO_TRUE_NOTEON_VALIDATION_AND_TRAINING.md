# S537–S544 : véritable test externe de NoteOn puis nouvelle boucle d'apprentissage

**15 octobre 2026 · résultats validés par GitHub Actions · aucune promotion de référence S80/Exact-K/Live.**

## Méthode : empêcher l'optimisation sur les étiquettes du jeu de test

La série S533–S536 avait atteint **55,50 % F1 NoteOn** sur 3 extraits MAESTRO piano historiques déjà utilisés dans le développement (212 vrais positifs, 136 faux positifs, 204 faux négatifs, exact pitch et temps à ±80 ms). Cette valeur n'était pas indépendante. Nous avons donc mis en œuvre, dans l'ordre, un test sur 5 morceaux de **5 compositeurs différents des 3 anciens**, en gelant les notes WAV avant d'ouvrir leurs MIDI.

- [S537 réussi](https://github.com/Andriamarosoa/note/actions/runs/38654030814) : sélection déterministe d'après le seul CSV MAESTRO-v3, en excluant les 3 anciennes compositions et compositeurs S469. CSV SHA256 cb88a64a12971a995dd41852aa3a33879501cf70d01b9a6feb637f4ad8d8ddff ; nouveaux morceaux sélectionnés sans télécharger aucun WAV ou MIDI. Un filtrage de durée 40–240 s a été nécessaire pour obtenir cinq compositeurs distincts, sans utiliser les annotations.
- [S538 réussi](https://github.com/Andriamarosoa/note/actions/runs/38656509448) : les 5 WAV authentiques sont exécutés dans la même pile historique de candidats temporels V88 et de modèle de K, avec les poids freeze_local_combo originaux immuables. Les 5×12 s de candidats temporels ont été enregistrés et leurs sources WAV vérifiées par empreintes. Ce n'est **pas** une exécution du S80 complet.
- [S539 réussi](https://github.com/Andriamarosoa/note/actions/runs/38656730774) : même Spotify Basic Pitch 0.4.0, mêmes seuils, note names et bornes temporelles, sans MIDI.
- [S540 réussi](https://github.com/Andriamarosoa/note/actions/runs/38656791368) : les observations acoustiques originales de 49 hauteurs × 11 indices ont été calculées pour **les 962 candidats V88**, y compris K0, sans ouvrir aucune annotation MIDI.
- [S541 réussi](https://github.com/Andriamarosoa/note/actions/runs/38656969947) : la moyenne des probabilités des **trois modèles acoustiques S528 réellement entraînés sur les anciens morceaux**, avec seuil 0,70, a produit quatre politiques de prédiction de notes ; les 4 ledgers et leur SHA ont été gelés **avant d'accéder à un seul MIDI de test**. Les modèles ont été combinés en moyenne car aucun ancien morceau ne correspond à l'identité d'un nouveau morceau. Cette combinaison était précisée dans S541 avant validation.

Les quatre politiques gelées : Basic Pitch naissances uniquement ; classificateur S528 moyenné sur trois modèles sans déduplication ; même classificateur avec déduplication 80 ms des seules nouvelles notes ; déduplication 100 ms plus veto faible-harmonique. Les notes Basic Pitch existantes ne sont jamais modifiées.

## S542 — mesure indépendante des notes MIDI et de l'attaque

[CI S542 réussi](https://github.com/Andriamarosoa/note/actions/runs/38657068577). Les annotations MIDI officielles ont été téléchargées **seulement après** comparaison des empreintes SHA avec S541. Les cinq extraits contiennent **518 vrais groupes MIDI NOTE_ON et 754 notes de hauteur 40..88**. Toutes les notes prédites sans vraie attaque associée sont comptées faux positifs ; toutes les vraies notes manquées sont comptées faux négatifs. Matching un-à-un par nom de hauteur et par temps, tolérance ±80 ms. Scores arrondis :

| Politique prédéclarée | TP | FP | FN | Précision | Rappel | F1 NoteOn ±80 ms |
|---|---:|---:|---:|---:|---:|---:|
| Basic Pitch birth only | 134 | 56 | 620 | 70,53 % | 17,77 % | **28,39 %** |
| S528 moyenne 3 modèles, p≥0,70, protégé | 280 | 239 | 474 | 53,95 % | 37,14 % | **43,99 %** |
| S533 déduplication 80 ms, protégée | 279 | 160 | 475 | 63,55 % | 37,00 % | **46,77 %** |
| S536 déduplication 100 ms + contrôle harmonique | 279 | 157 | 475 | 63,99 % | 37,00 % | **46,89 %** |

**La correction de doublons généralise : F1 +2,78 points de pourcentage avec 80 ms, +2,90 points avec 100 ms et veto harmonique, relativement au même modèle non filtré. Cependant, les deux corrections perdent 1 TP sur le nouveau corpus (279 contre 280), donc elles ne satisfont pas une contrainte de non-régression absolue.** Les gains de F1 à ±30/50 ms sont également positifs (par exemple 80 ms : 36,21 % et 42,58 % contre 34,72 % et 40,06 % sans correction).

Le gain F1 de la politique 80 ms est positif sur **chacun des cinq compositeurs** (parmi eux Sergei Rachmaninoff, Johann Sebastian Bach, Frédéric Chopin, Franz Schubert/Franz Liszt et Franz Liszt). Malgré cela, le meilleur F1 S536 historique **55,50 %** descend à **46,89 %** sur cinq nouvelles sources : le choix de l'architecture n'est pas suffisamment généralisé pour remplacer la référence. Il existe des répétitions MIDI réelles rapprochées que toute déduplication peut effacer.

## S543 — enquête post-validation et cause de la limite de rappel

[CI S543 réussi](https://github.com/Andriamarosoa/note/actions/runs/38657282732). Analyse **après** ouverture des annotations, donc diagnostique seulement et impropre à régler les mêmes cinq tests :

- V88 a généré **962 candidats temporels** sur ces 60 secondes de piano. **498/518** vrais groupes d'attaques sont appariables un-à-un à un centre V88 dans ±80 ms (plafond d'événements ≈96,14 % sous ce protocole). 20 véritables attaques n'ont pas de candidat temporel unique.
- Parmi ces **498** attaques appariées à V88, **306 ont un K original égal à zéro** (≈61,45 %) ; K0 ne doit donc **pas** servir à supprimer les naissances de notes.
- Seulement **10 des 754 vrais événements de hauteurs** n'ont aucun centre V88 à proximité dans ±80 ms. La faiblesse majeure est donc le **choix de la bonne hauteur et des reprises au bon moment**, non uniquement la détection des centres temporels.
- Une vraie note **F4 (MIDI 65)** à ~**4,952 s** dans l'extrait Schubert/Liszt était correcte avant déduplication et devient FN après ; il faut apprendre à distinguer une **réattaque authentique** d'un doublon de résonance, sans utiliser les étiquettes du test comme règle statique.

### Régressions vs référence

La candidate S536 « 55,50 % » de S533–S536 **n'est pas promue** ; sur le jeu neuf elle tombe à 46,89 %. Le candidat S533 « 54,69 % » tombe à 46,77 %. Le succès démontré est **structurel** : les duplications de pitch naissantes existent et les supprimer améliore la précision sur cinq nouveaux compositeurs, mais la perte d'une vraie attaque et le rappel ~37 % interdisent de mettre ce correctif dans S80 officiel sans tests supplémentaires.

## Après validation : augmenter effectivement la capacité d'apprentissage

La boucle [S544](https://github.com/Andriamarosoa/note/actions/workflows/v273-s544-eight-composer-trained-note-heads.yml) élargit l'entraînement des **3 anciens + 5 nouveaux compositeurs déjà annotés** :

1. Composer-held leave-one-recording-out (8 partitions), pas de vérité du compositeur retenu dans son modèle.
2. Deux architectures, pour **16 vrais modèles entraînés** : 49 hauteurs × 7 descripteurs acoustiques sur vrais onsets connus seulement ; ou 49 hauteurs × 38 descripteurs de concurrence harmonique en introduisant aussi les candidats MIDI temporellement non appariés comme négatifs du benchmark (doublons possibles).
3. Mesurer le **F1 des notes et attaques sur toutes les annotations**, et non un taux d'accords conditionnel, avec et sans déduplication 80 ms.
4. Ce travail utilise maintenant les cinq nouveaux morceaux comme **entraînement/développement**, et **aucun score S544 sur ces morceaux ne sera présenté comme une nouvelle validation prospective**.
5. Toute promotion réelle demandera **encore une cohorte source entièrement différente** sélectionnée et scellée avant sa lecture, plus reproduction complète d'Exact-K S80, GuitarSet/IDMT et latence Live.

**Aucune fusion dans la branche officielle, aucune nouvelle référence, objectif F1 80 % non atteint.**
