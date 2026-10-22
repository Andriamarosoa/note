# Onset : un VRAI réseau de neurones qui prévoit l'erreur AVANT le NoteOn

Branche de recherche : `codex/note-onset-predecision-neural256`. Uniquement Onset ; les cinq autres caractéristiques et le MIDI officiel restent strictement intacts.

## Définition exacte de « avant »

La prédiction n'est **pas** une devinette sur un futur non entendu. Elle est une décision de **risque d'erreur causale au même instant**, calculée **avant** que le sélecteur Onset décide d'émettre le NoteOn.

À chaque 256 nouveaux échantillons PCM à 44 100 Hz (5,805 ms de cadence), le frontal S613 entraîné à 256 et la tête S701 entraînée à 256 fournissent uniquement leurs caractéristiques acoustiques et historiques courantes/passées. Un **nouveau vrai réseau neuronal entraîné** `StandardScaler + MLPClassifier` (63 dimensions, neurones cachés 64 puis 32, sortie sigmoïde binaire) prédit pour les 49 notes MIDI 40…88 `P(attaque correcte | signal disponible)`, et donc `P(erreur)=1−P(correct)`.

Cette probabilité est calculée dans `Stream._step()` **avant** les appels à `self.base.step` et `self.shadow.step`. Un troisième sélecteur **privé Onset** reçoit alors le posterior Onset masqué uniquement pour les notes que le modèle estime douteuses selon un seuil figé sur inner-CAL. Il n'existe **aucune attente après l'attaque**, aucune trame future, aucun buffer d'anticipation, aucune autre architecture importée, et aucune émission dans le MIDI officiel.

Contrairement à un simple seuil dur, le MLP est effectivement entraîné avec des exemples supervisés de vraies et fausses attaques. Le seuil CAL décide ensuite le compromis pour le flux expérimental, mais **le seuil seul n'est pas le modèle**.

## 63 entrées PRE-décision

- 12 variables harmoniques natives : fondamentale de la note, relations −12/−19/−24/−28 demi-tons, voisins ±1, rapports d'harmoniques, indices de K et évolution acoustique courante/passée ;
- 3 postérieurs Onset/Sustain/Release **du frontal Onset privé S613**, 1 score S701 et 7 probabilités de K purement auxiliaires ;
- 1 coordonnée de registre MIDI et **39 variables S701 avec historique causal**.
- Aucune annotation GT/MIDI n'entre dans les 63 dimensions, ni les états des cinq autres décodeurs. Les événements enregistrés pour entraîner le risque ont exactement la même représentation vectorielle que les 49 vecteurs produits à chaque 256 dans le flux LIVE. Les tests comparent cette identité avec les caractéristiques d'origine.

## Entraînement et contrôles

Le workflow entraîne **cinq véritables réseaux MLP** en crossfit sur GuitarSet : pour chaque composition OUTER (16 pistes), trois autres compositions servent à TRAIN (48 pistes), la quatrième à CAL (16 pistes). Le modèle et son scaler sont ajustés exclusivement sur TRAIN ; le seuil d'acceptation des risques est figé sur CAL sans lire OUTER. Chaque MLP est sérialisé avec SHA256, dimensions ABI 63, cadence entraînée `256` et liste des compositions TRAIN/CAL/OUTER. Un MLP expérimental entraîné sur les 80 pistes est également fourni, **sans prétention de test indépendant**.

Les labels d'apprentissage `correcte/fausse` proviennent d'un appariement NoteOn ±80 ms en **post-traitement des WAV d'origine**, jamais durant l'inférence. L'entraînement mesure d'abord une performance candidate par filtrage d'événements ; **cette approximation n'est pas le résultat réel en LIVE** car le veto change le réarmement de l'état. Un second replay du *vrai flux causal*, appelé `scripts/replay_predecision256.py`, vérifie alors en temps réel logique que le MLP est effectivement invoqué avant chaque décision sur **les 306 536 trames PCM256 des 80 pistes**, et mesure à nouveau TP/FP/FN/F1 avec le vrai sélecteur autonome.

Tout score en-dessous de la référence Onset S613 de **67,5874 %** échoue scientifiquement (artefact de preuve conservé). Le seuil historique S708 **73,2520 %** est suivi séparément mais provient de poids entraînés à 1024, exclus du runtime 256.

## Limites essentielles

**Attention à la fuite historique** : S613 a été initialement entraîné sur 20 des 80 prises GuitarSet, S701 sur 48 des 80 (et les 32 autres appartenaient à son HELD). Même si le nouveau réseau risque est entraîné OOF par composition, **l'ensemble S613 + S701 + risque n'a PAS été évalué sur une population externe jamais vue**. Les nouvelles étiquettes en OUTER sont disjointes du nouveau réseau MLP, mais pas forcément du frontal préentraîné. Ne pas présenter cet OOF comme une démonstration de généralisation industrielle.

Le modèle apprend la probabilité qu'une **proposition Onset** soit une fausse attaque **avant son émission**. Il ne prétend pas prédire un futur signal audio non disponible ni créer une attaque à partir du silence sans proposition. Les FN devront être traitées par une tête complémentaire, elle aussi entraînée indépendamment. Un score correct sur GuitarSet ne prouve pas la latence matérielle 5,805 ms : la période des décisions n'est pas le délai d'exécution total.

## Où voir le travail réel

- `src/onset256/predecision_risk.py` : **63D, MLP entraîné**, probabilités P(correcte) par hauteur.
- `src/onset256/runtime.py` : prédiction **avant** la décision, propre état privé, témoins S613/S701 inchangés.
- `scripts/train_onset_predecision256.py` : apprentissage de cinq checkpoints OOF, calibration et audit.
- `scripts/replay_predecision256.py` : score du **vrai** décodeur causal après apprentissage.
- `tests/test_predecision_neural256.py` : contrôle ordre temporel, ABI, vrai entraînement MLP.
- `.github/workflows/onset-predecision-neural256.yml` : tests, audio original SHA, entraînement supervisé, replay, artefacts, absence d'effets sur les autres axes.

**Statut actuel : expérience à exécuter.** Ne jamais inventer d'amélioration F1 avant la fin du workflow et l'examen des rapports. Le modèle n'est pas promu.
